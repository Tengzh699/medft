from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from .config import evaluation_paths, file_hash, project_path, utc_now, write_json
from .data import normalize, read_jsonl, redact, reserved_questions, validate_messages, write_jsonl
from .public_filter import (
    NON_HEALTH_CONTENT, PUBLIC_FILTER_VERSION, PUBLIC_REVISION, QUESTION_OUT_OF_SCOPE, TOPIC_QUESTION,
    UNSUPPORTED_OR_NOISY_CONTENT, load_public_exclusions, public_answer_hash, topic_matches,
)
from .quality import validate_grounding

PUBLIC_REPO = "FreedomIntelligence/huatuo_encyclopedia_qa"
PUBLIC_FILE = "train_datasets.jsonl"
QUESTION_PREFIXES = ("", "想了解一下：", "请用通俗的话解释：", "健康知识咨询：")
CONSULT_PREFIXES = QUESTION_PREFIXES + ("请帮我梳理这些情况：", "就诊前想咨询：", "我想先了解就医方向：", "关于下面的情况想请教：")
REFUSAL_PREFIXES = ("", "请帮我处理这个情况：", "我想咨询：", "现在遇到一个问题：", "请告诉我该怎么办：", "我有这样的疑问：", "想请你帮忙：", "以下情况应该怎么处理：")
TUTORIAL_QUESTION_PREFIXES = QUESTION_PREFIXES + (
    "请解释这个健康问题：", "想弄清一个概念：", "能帮我理解一下吗：", "做健康笔记时有个疑问：",
    "请从基础知识说起：", "我对这个问题不太明白：", "科普咨询：", "请说明其中的道理：",
    "看健康资料时想问：", "请用容易理解的语言回答：", "想学习相关常识：", "请帮我澄清这个疑惑：",
    "希望了解基本含义：", "请给一个清楚的说明：", "有个日常健康问题：", "请介绍相关知识：",
)
TUTORIAL_REFUSAL_PREFIXES = REFUSAL_PREFIXES + (
    "这件事让我有些困惑：", "请看我的请求：", "我想这样做：", "帮我判断接下来怎么办：",
    "遇到以下情况想问你：", "请回应我的这个问题：", "需要你帮我拿个主意：", "我现在有这样的打算：",
)
V2_PREFIXES = {
    "knowledge_qa": ("", "想了解：", "请解释：", "科普提问：", "请用通俗的话说明：",
                     "日常健康咨询：", "想弄清楚：", "请介绍相关常识：", "我有个疑问：", "健康学习笔记："),
    "symptom_collection": ("", "就诊前想梳理：", "请帮我整理病情：", "想描述一下：", "我有这样的不适："),
    "triage": ("", "想了解就医方向：", "请帮我判断就诊安排：", "就医咨询："),
    "examination": ("", "检查方面想了解：", "准备面诊时想问：", "请介绍检查思路："),
    "report_interpretation": ("", "请帮我理解这份模拟报告："),
    "refusal": REFUSAL_PREFIXES + ("请回应这个请求：", "现在想问："),
}
V2_1_PREFIXES = {task: (("", "想了解：", "请说明：") if task == "knowledge_qa" else ("",)) for task in V2_PREFIXES}

# These rules select candidates for this project's scope; they do not certify facts.
PUBLIC_EXCLUDE = re.compile(
    r"治疗|用药|用法|用量|剂量|处方|口服|外用|静脉|注射|偏方|秘方|中药|中医|补肾|壮阳|排毒|食疗|"
    r"孕妇|妊娠|哺乳|婴儿|小儿|儿童|新生儿|术后|手术后|癌症|肿瘤|自杀|自伤|"
    r"包治|根治|百分之百|100%|治愈|疗效|特效|保健品|广告|微信|[微vV]信号|QQ|https?://|www\."
)
DOSE_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|毫克|毫升|片|粒|单位|次/日|次/天)", re.I)
KNOWLEDGE_QUESTION = re.compile(r"什么|为何|为什么|区别|原理|含义|意义|特点|作用|是否|能否|原因|检查|指标|症状|预防|注意")


def _generation_spec(config: dict) -> dict:
    spec = config.get("data_generation")
    if not isinstance(spec, dict):
        raise ValueError("当前配置没有data_generation，请使用 --config configs/v1_expanded.yaml。")
    counts = spec.get("synthetic_counts", {})
    profile = spec.get("template_profile", "default")
    expected = set(V2_PREFIXES) if profile in {"v2", "v2_1", "v3"} else {"knowledge_qa", "consultation", "refusal"}
    if set(counts) != expected or set(config["data"]["target_mix"]) != expected:
        raise ValueError(f"生成任务与模板配置不匹配，需要：{', '.join(sorted(expected))}。")
    public = spec.get("public", {})
    values = [*counts.values(), public.get("count")]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("各类生成数量必须为非负整数。")
    if type(public.get("scan_limit")) is not int or public["scan_limit"] < 1:
        raise ValueError("公开数据scan_limit必须为正整数。")
    if sum(values) == 0:
        raise ValueError("至少需要生成一条数据。")
    if profile not in {"default", "tutorial6000", "v2", "v2_1", "v3"}:
        raise ValueError("未知template_profile。")
    if profile in {"v2", "v2_1"} and public["count"]:
        raise ValueError("V2当前使用独立合成场景，公开混合数据请继续使用v1_expanded配置。")
    return spec


def download_public_data(config: dict, *, endpoint: str | None = None,
                         transport: str = "hub", no_proxy: bool = False) -> dict:
    """Explicit download, invoked by the user; never execute a dataset loader script."""
    from huggingface_hub import hf_hub_download
    from requests import RequestException

    from .downloads import download_http_files, error_chain

    spec = _generation_spec(config)["public"]
    if transport not in {"hub", "http"}:
        raise ValueError("transport须为hub或http。")
    if no_proxy and transport != "http":
        raise ValueError("--no-proxy需与 --transport http 一起使用，仅影响本次HTTP会话。")
    paths, upstream_hashes = {}, {}
    try:
        if transport == "http":
            from huggingface_hub import constants

            paths, upstream_hashes = download_http_files(
                repo=PUBLIC_REPO, revision=PUBLIC_REVISION, filenames=(PUBLIC_FILE, "README.md"),
                cache_dir=project_path(spec["cache_dir"]), endpoint=endpoint or constants.ENDPOINT,
                no_proxy=no_proxy,
            )
        else:
            for filename in (PUBLIC_FILE, "README.md"):
                print(f"Hub下载：{filename}", file=sys.stderr, flush=True)
                options = {"endpoint": endpoint} if endpoint else {}
                paths[filename] = Path(hf_hub_download(
                    repo_id=PUBLIC_REPO, repo_type="dataset", revision=PUBLIC_REVISION,
                    filename=filename, cache_dir=str(project_path(spec["cache_dir"])),
                    etag_timeout=30, **options,
                ))
    except (RequestException, OSError, RuntimeError) as error:
        raise RuntimeError(
            f"公开数据下载失败（{transport}方式）。底层错误：\n  {error_chain(error)}\n"
            "Hub方式失败可加 --transport http 重试；若错误链明确为代理连接失败且网络允许直连，"
            "再加 --no-proxy。已有缓存及HTTP的.part文件会保留。"
        ) from error
    import yaml

    card = paths["README.md"].read_text(encoding="utf-8")
    metadata = yaml.safe_load(card.split("---", 2)[1]) if card.startswith("---") else {}
    if not isinstance(metadata, dict) or metadata.get("license") != "apache-2.0":
        raise ValueError("数据卡的许可声明与预期不同，请先核对来源。")
    manifest = {
        "repo_id": PUBLIC_REPO, "resolved_revision": PUBLIC_REVISION,
        "filename": PUBLIC_FILE, "path": str(paths[PUBLIC_FILE].resolve()),
        "sha256": file_hash(paths[PUBLIC_FILE]), "bytes": paths[PUBLIC_FILE].stat().st_size,
        "card_path": str(paths["README.md"].resolve()), "card_sha256": file_hash(paths["README.md"]),
        "declared_license": metadata["license"], "upstream_split": "train",
        "url": f"https://huggingface.co/datasets/{PUBLIC_REPO}", "downloaded_at": utc_now(),
        "download_transport": transport, "upstream_hashes": upstream_hashes,
    }
    write_json(project_path(spec["manifest"]), manifest)
    return manifest


def _strings(value, depth=0) -> list[str]:
    # Actual upstream questions can be [["question"]], despite the loader schema.
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list) or depth > 3:
        return []
    return [text for item in value for text in _strings(item, depth + 1)]


def _public_candidate(row: dict, blocked: set[str]) -> tuple[tuple[str, list[str]] | None, str | None]:
    questions, answers = _strings(row.get("questions")), _strings(row.get("answers"))
    # Multiple equivalent questions share one answer. Do not invent answer pairings.
    answers = list(dict.fromkeys(answers))
    if not questions or len(answers) != 1:
        return None, "invalid_or_ambiguous_schema"
    answer = answers[0]
    if not 80 <= len(answer) <= 700:
        return None, "answer_length"
    if PUBLIC_EXCLUDE.search(answer) or DOSE_PATTERN.search(answer):
        return None, "out_of_scope_or_treatment"
    if redact(answer)[1]:
        return None, "pii_pattern"
    if NON_HEALTH_CONTENT.search(answer):
        return None, "non_health_content"
    if UNSUPPORTED_OR_NOISY_CONTENT.search(answer):
        return None, "unsupported_or_noisy_content"
    accepted = []
    failures = Counter()
    for question in questions:
        if normalize(question) in blocked:
            return None, "reserved_question_overlap"
        if not 6 <= len(question) <= 160 or not KNOWLEDGE_QUESTION.search(question):
            continue
        if PUBLIC_EXCLUDE.search(question) or DOSE_PATTERN.search(question):
            continue
        if redact(question)[1]:
            continue
        if len(re.findall(r"[\u4e00-\u9fff]", question)) < 4:
            continue
        if NON_HEALTH_CONTENT.search(question) or QUESTION_OUT_OF_SCOPE.search(question):
            failures["question_outside_common_health_scope"] += 1
            continue
        if not TOPIC_QUESTION.search(question) or not topic_matches(question, answer):
            failures["missing_shared_health_topic"] += 1
            continue
        if question not in accepted:
            accepted.append(question)
    if not accepted:
        return None, failures.most_common(1)[0][0] if failures else "question_scope_or_pii"
    return (answer, accepted), None


def import_public_rows(config: dict, blocked: set[str], *, allow_shortfall: bool = False) -> tuple[list[dict], dict]:
    spec = _generation_spec(config)["public"]
    if spec["count"] == 0:
        return [], {"selected": 0, "configured_count": 0}
    manifest_path = project_path(spec["manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("公开数据尚未下载，请先运行 download-data --config configs/v1_expanded.yaml。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get(key) for key in ("repo_id", "resolved_revision", "filename", "declared_license", "upstream_split")) != (
        PUBLIC_REPO, PUBLIC_REVISION, PUBLIC_FILE, "apache-2.0", "train",
    ):
        raise ValueError("公开数据版本记录不匹配，请重新执行 download-data。")
    path = Path(manifest["path"])
    if not path.is_file() or file_hash(path) != manifest["sha256"]:
        raise ValueError("公开数据文件缺失或哈希改变，请重新执行 download-data。")
    card_path = Path(manifest["card_path"])
    if not card_path.is_file() or file_hash(card_path) != manifest["card_sha256"]:
        raise ValueError("公开数据卡缺失或哈希改变，请重新执行 download-data。")
    target, scan_limit = spec["count"], spec["scan_limit"]
    policy = load_public_exclusions(config)
    rng = random.Random(config["project"]["seed"])
    selected, seen_answers, seen_questions = [], set(), set()
    rejected, scanned, eligible = Counter(), 0, 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in itertools.islice(enumerate(stream, 1), scan_limit):
            scanned += 1
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("not an object")
            except (ValueError, TypeError):
                rejected["invalid_json"] += 1
                continue
            candidate, reason = _public_candidate(raw, blocked)
            if candidate is None:
                rejected[reason] += 1
                continue
            answer, questions = candidate
            fingerprint = hashlib.sha256(normalize(answer).encode()).hexdigest()
            if fingerprint in seen_answers or any(normalize(question) in seen_questions for question in questions):
                rejected["duplicate_answer_or_question"] += 1
                continue
            seen_answers.add(fingerprint)
            # A row contributes at most one question; all of its alternate questions are reserved.
            seen_questions.update(map(normalize, questions))
            question = questions[0]
            record = {
                "id": f"huatuo-{line_number}", "case_id": f"huatuo-answer-{fingerprint[:24]}",
                "task": "knowledge_qa", "source": PUBLIC_REPO, "origin": "public_encyclopedia",
                "review_status": "unreviewed", "source_revision": PUBLIC_REVISION,
                "source_file": PUBLIC_FILE, "source_line": line_number,
                "source_license": manifest["declared_license"], "upstream_split": "train",
                "public_filter_version": PUBLIC_FILTER_VERSION,
                "health_topics": topic_matches(question, answer),
                "related_questions": questions,
                "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            }
            eligible += 1
            if len(selected) < target:
                selected.append(record)
            else:
                replacement = rng.randrange(eligible)
                if replacement < target:
                    selected[replacement] = record
    if len(selected) < target and not allow_shortfall:
        raise ValueError(f"公开数据过滤后仅{len(selected)}条，目标{target}条；请检查过滤报告或增大scan_limit。拒绝统计：{dict(rejected)}")
    # Apply recorded decisions to this fixed sample. Do not replace rejected rows
    # with unseen candidates merely to recover the original target count.
    sampled_count, excluded = len(selected), []
    if policy is not None:
        kept = []
        for row in selected:
            answer_hash = public_answer_hash(row["messages"][1]["content"])
            at_line = policy["by_line"].get(row["source_line"])
            if at_line and at_line["answer_sha256"] != answer_hash:
                raise ValueError(f"剔除清单第{row['source_line']}行的答案哈希不匹配，请核对原始记录。")
            decision = policy["by_hash"].get(answer_hash)
            if decision:
                excluded.append({"id": row["id"], "source_line": row["source_line"],
                                 "answer_sha256": answer_hash, "reason_code": decision["reason_code"],
                                 "reason": decision["reason"]})
            else:
                kept.append({**row, "public_exclusions_sha256": policy["sha256"]})
        selected = kept
    return selected, {"source": manifest, "filter_version": PUBLIC_FILTER_VERSION,
                      "scanned": scanned, "scan_limit": scan_limit, "eligible": eligible,
                      "sampled_before_exclusions": sampled_count, "excluded_count": len(excluded),
                      "excluded": excluded,
                      "exclusions_path": policy["path"] if policy else None,
                      "exclusions_sha256": policy["sha256"] if policy else None,
                      "selected": len(selected), "configured_count": target,
                      "shortfall_allowed": allow_shortfall, "rejected": dict(rejected),
                      "selected_by_topic": dict(Counter(topic for row in selected for topic in row["health_topics"]))}


def _seed_variants(record: dict, profile: str = "default") -> list[dict]:
    variants = []
    task = record["task"]
    prefixes = (V2_1_PREFIXES if profile == "v2_1" else V2_PREFIXES)[task] if profile in {"v2", "v2_1"} else {
        "knowledge_qa": QUESTION_PREFIXES, "consultation": CONSULT_PREFIXES, "refusal": REFUSAL_PREFIXES}[task]
    if profile == "tutorial6000":
        prefixes = {"knowledge_qa": TUTORIAL_QUESTION_PREFIXES, "consultation": CONSULT_PREFIXES,
                    "refusal": TUTORIAL_REFUSAL_PREFIXES}[task]
    for index, prefix in enumerate(prefixes):
        for extended in ((False, True) if task == "consultation" else (False,)):
            messages = [dict(message) for message in record["messages"]]
            messages[0]["content"] = prefix + messages[0]["content"]
            if extended:
                messages.extend([
                    {"role": "user", "content": "去门诊时，我该怎样整理刚才说的情况？"},
                    {"role": "assistant", "content": (
                        f"可以先向医生说明：“{record['messages'][0]['content']}”；再补充：“{messages[2]['content']}”。"
                        "带上既往检查和正在使用的药物清单，并记录症状对日常活动的影响。"
                        "实际病情有变化时要如实补充，医生会结合面诊判断是否需要检查。"
                        "就医的紧急程度按前面建议执行，整理资料不要延误求助。"
                    )},
                ])
            variants.append({**record, "id": f"{record['id']}-v{index}-{'long' if extended else 'basic'}",
                             "origin": "template", "seed_id": record["id"], "messages": messages})
    return variants


def generate_data(config: dict, *, synthetic_only: bool = False) -> dict:
    spec = _generation_spec(config)
    if spec.get("template_profile") == "v3":
        raise ValueError("V3使用有界筛选和固定验证集，请运行 python -m medft curate-data --config configs/v3.yaml。")
    if synthetic_only and not sum(spec["synthetic_counts"].values()):
        raise ValueError("synthetic-only模式至少需要一条模板数据。")
    seed_path = project_path(spec["seed_source"])
    seeds = read_jsonl(seed_path)
    blocked = reserved_questions(config)
    pools = {task: [] for task in config["data"]["target_mix"]}
    seed_ids = set()
    for record in seeds:
        if any(not record.get(key) for key in ("id", "case_id", "source", "task", "messages")):
            raise ValueError("种子缺少id、case_id、source、task或messages。")
        if record["task"] not in pools:
            raise ValueError(f"未知种子任务：{record['task']}")
        validate_messages(record["messages"])
        if record["id"] in seed_ids:
            raise ValueError(f"种子id重复：{record['id']}")
        seed_ids.add(record["id"])
        profile = spec.get("template_profile", "default")
        if profile in {"v2", "v2_1"} and (record["task"] == "symptom_collection" and len(record["messages"]) < 6):
            raise ValueError("V2症状采集种子须至少包含三轮真实信息交互。")
        if profile not in {"v2", "v2_1"} and record["task"] == "consultation" and len(record["messages"]) != 4:
            raise ValueError("问诊种子须包含两轮用户与助手消息。")
        if profile not in {"v2", "v2_1"} and record["task"] != "consultation" and len(record["messages"]) != 2:
            raise ValueError("知识与拒答种子须包含一轮用户与助手消息。")
        if record.get("review_status") not in {"unreviewed", "clinician_reviewed"}:
            raise ValueError("种子应明确标记review_status：未审核时使用unreviewed。")
        validate_grounding(record, required=profile == "v2_1")
        for variant in _seed_variants(record, spec.get("template_profile", "default")):
            if any(normalize(message["content"]) in blocked for message in variant["messages"] if message["role"] == "user"):
                continue
            pools[record["task"]].append(variant)
    rng = random.Random(config["project"]["seed"])
    synthetic = []
    for task, count in spec["synthetic_counts"].items():
        pool = pools[task]
        if count > len(pool):
            raise ValueError(f"{task}需要{count}条，但种子最多支持{len(pool)}条；请增加独立种子，不重复补齐。")
        rng.shuffle(pool)
        synthetic.extend(pool[:count])
    # Include seed alternate phrasings in the blocklist to avoid exact public/template overlap.
    blocked.update(normalize(row["messages"][0]["content"]) for row in synthetic)
    public_rows, public_report = ([], {"selected": 0, "skipped_explicitly": True}) if synthetic_only else import_public_rows(config, blocked)
    rows = synthetic + public_rows
    rng.shuffle(rows)
    destination = project_path(config["data"]["source"])
    if destination.resolve() in {seed_path.resolve(), *(path.resolve() for path in evaluation_paths(config))}:
        raise ValueError("生成路径不能覆盖种子或评测源。")
    # Publication of the assembled source happens only after the complete import succeeds.
    write_jsonl(destination, rows)
    public_review = destination.parent / "public_review.jsonl"
    write_jsonl(public_review, public_rows)
    report = {
        "generated_at": utc_now(), "seed": config["project"]["seed"], "seed_sha256": file_hash(seed_path),
        "template_profile": spec.get("template_profile", "default"),
        "seed_records_count": len(seeds),
        "seed_records_by_task": dict(Counter(row["task"] for row in seeds)),
        "source": str(destination), "source_sha256": file_hash(destination), "count": len(rows),
        "by_task": dict(Counter(row["task"] for row in rows)),
        "actual_mix": {task: sum(row["task"] == task for row in rows) / len(rows) for task in pools},
        "synthetic_only": synthetic_only or spec["public"]["count"] == 0,
        "synthetic_only_requested": synthetic_only,
        "by_origin": dict(Counter(row["origin"] for row in rows)),
        "case_groups": len({row["case_id"] for row in rows}),
        "synthetic_case_groups": len({row["case_id"] for row in synthetic}),
        "evaluation_sha256": file_hash(project_path(config["data"]["evaluation_source"])),
        "evaluation_sources_sha256": {str(path.resolve()): file_hash(path) for path in evaluation_paths(config)},
        "public": public_report,
        "public_review_file": str(public_review),
        "contains_unreviewed_data": any(row["review_status"] != "clinician_reviewed" for row in rows),
        "interpretation": "Template variants are not independent medical cases; public filtering is heuristic, not clinical review.",
    }
    write_json(destination.parent / "preparation.json", report)
    return report
