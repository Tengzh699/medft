"""Bounded V3 candidate selection and a matched historical-data control."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import evaluation_paths, file_hash, load_config, project_path, utc_now, write_json
from .corpus import import_public_rows
from .data import (
    _group_keys, assert_no_leakage, build_data, case_groups, clean_records, normalize,
    read_jsonl, reserved_questions, split_records, write_jsonl,
)
from .quality import validate_grounding
from .public_filter import load_public_rewrites, public_answer_hash

CURATION_VERSION = "v3-selection-v1"
PUBLIC_CAUTION = re.compile(
    r"正常值|正常范围|参考值|诊断标准|\d+(?:\.\d+)?\s*(?:%|％|mmol|mg/d|g/L|μmol)|"
    r"排毒|宿便|体内毒素|酸性体质|一定会|一定是|绝对安全|无需就医|不用就医|必须空腹|"
    r"多少钱|价格|收费|费用", re.I,
)


def answer_key(row: dict) -> str:
    text = normalize(row["messages"][-1]["content"])
    return hashlib.sha256(text.encode()).hexdigest()


def _slot(family: str, count: int) -> int:
    return int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % count + 1


def seed_candidates(rows: list[dict]) -> tuple[list[dict], Counter]:
    """Select information states once; never manufacture prefix variants."""
    selected, reports, portions = [], defaultdict(list), defaultdict(list)
    rejected = Counter()
    allowed = {"symptom_collection": {"stable", "changed", "missing_duration"},
               "triage": {"stable", "changed"}, "examination": {"purpose", "prior_normal"}}
    for row in rows:
        validate_grounding(row, required=True)
        task, family = row["task"], row["case_id"]
        keep = False
        if task == "knowledge_qa":
            if re.fullmatch(r"v21-k\d+", row["id"]):
                keep = True
            elif row.get("calculation"):
                portions[family].append(row)
        elif task in allowed:
            match = re.search(r"-(?:collect|triage|exam)-(\d+)-", row["id"])
            keep = bool(match and int(match[1]) == _slot(family, 5) and row.get("branch") in allowed[task])
        elif task == "report_interpretation":
            reports[family].append(row)
        elif task == "refusal":
            # Original frame only; duplicate answers are subsequently capped.
            keep = row.get("request_frame") == 0
        if keep:
            selected.append(row)
        elif not row.get("calculation") and task != "report_interpretation":
            rejected["unselected_information_state_or_prefix"] += 1
    for family, group in sorted(portions.items()):
        selected.append(group[_slot(family, len(group)) - 1])
        rejected["portion_variants_capped"] += len(group) - 1
    for family, group in sorted(reports.items()):
        profile = _slot(family, 3)
        choices = [row for row in group if row["id"].startswith(f"{family}-{profile}-")]
        # Rotate four different branches across report families, rather than
        # duplicating every numerical profile of every report.
        random.Random(f"{CURATION_VERSION}:{family}").shuffle(choices)
        selected.extend(choices[:4])
        rejected["report_variants_capped"] += len(group) - min(4, len(choices))
    return [{**row, "origin": row.get("origin", "template")} for row in selected], rejected


def _shingles(text: str) -> set[str]:
    text = normalize(text)
    return {text[i:i + 5] for i in range(max(1, len(text) - 4))}


def prune_candidates(rows: list[dict], *, maximum: int, threshold: float) -> tuple[list[dict], list[dict]]:
    """Bound exact repetition; detect public text near duplicates with equal numbers.

    This is a character-overlap heuristic, not semantic deduplication or fact checking.
    Numeric signatures prevent collapsing otherwise similar but different results.
    """
    kept, rejected, cases, answers = [], [], Counter(), set()
    public_index = []
    for row in rows:
        reason = None
        key = (row["task"], answer_key(row))
        if cases[row["case_id"]] >= maximum:
            reason = "case_cap"
        elif key in answers:
            reason = "identical_final_answer_within_task"
        elif row.get("origin", "").startswith("public_encyclopedia"):
            text = " ".join(message["content"] for message in row["messages"])
            if PUBLIC_CAUTION.search(text):
                reason = "public_numeric_claim_or_caution_phrase"
            else:
                signature = tuple(re.findall(r"\d+(?:\.\d+)?", text))
                grams = _shingles(text)
                for old_signature, old_grams in public_index:
                    if signature == old_signature and len(grams & old_grams) / len(grams | old_grams) >= threshold:
                        reason = "public_character_near_duplicate"
                        break
                if reason is None:
                    public_index.append((signature, grams))
        if reason:
            rejected.append({"id": row["id"], "reason": reason})
            continue
        # Identical final answers across tasks are kept together during splitting.
        kept.append({**row, "curation_version": CURATION_VERSION, "answer_group": answer_key(row)})
        answers.add(key)
        cases[row["case_id"]] += 1
    return kept, rejected


def matched_control(pool: list[dict], train: list[dict], validation: list[dict], seed: int) -> tuple[list[dict], dict]:
    """Match training row counts by task and exclude all held-out case/opening keys."""
    held_keys = {key for row in validation for key in _group_keys(row)}
    held_answers = {answer_key(row) for row in validation}
    targets, strata, rejected = Counter(row["task"] for row in train), defaultdict(list), Counter()
    for row in pool:
        if any(key in held_keys for key in _group_keys(row)) or answer_key(row) in held_answers:
            rejected["held_out_case_opening_or_answer"] += 1
            continue
        strata[row["task"]].append(row)
    selected = []
    rng = random.Random(seed)
    for task, count in sorted(targets.items()):
        choices = sorted(strata[task], key=lambda row: row["id"])
        if len(choices) < count:
            raise ValueError(f"旧数据对照的{task}只有{len(choices)}条无重叠候选，无法匹配{count}条；不重复补齐。")
        rng.shuffle(choices)
        selected.extend(choices[:count])
    rng.shuffle(selected)
    assert_no_leakage({"train": selected, "validation": validation})
    return selected, {"matched_by_task": dict(targets), "rejected": dict(rejected)}


def rewrite_public_rows(rows: list[dict], config: dict, source_sha256: str | None) -> tuple[list[dict], dict]:
    catalog = load_public_rewrites(config)
    if catalog is None:
        return rows, {"enabled": False}
    if source_sha256 != catalog["source_sha256"]:
        raise ValueError("公开改编清单与下载源文件哈希不匹配。")
    originals = {row["source_line"]: row for row in rows}
    selected = []
    for line, entry in catalog["by_line"].items():
        row = originals.get(line)
        if (row is None or public_answer_hash(row["messages"][-1]["content"]) != entry["original_answer_sha256"]
                or entry["evidence"] not in row["messages"][-1]["content"]):
            raise ValueError(f"公开改编项的原始内容未通过源筛选、哈希或摘录检查：{line}")
        selected.append({**row, "id": row["id"] + "-v3-edit", "origin": "public_encyclopedia_rewritten",
                         "source_original_messages": row["messages"], "related_questions": [entry["question"], *row["related_questions"]],
                         "public_derivation": {"kind": "engineering_rewrite_not_clinical_review",
                                               "catalog_sha256": catalog["sha256"], "reason": entry["reason"], "evidence": entry["evidence"]},
                         "messages": [{"role": "user", "content": entry["question"]},
                                      {"role": "assistant", "content": entry["answer"]}]})
    return selected, {"enabled": True, "catalog_sha256": catalog["sha256"], "rewritten": len(selected),
                      "not_selected_for_rewrite": len(rows) - len(selected), "clinician_reviewed": False}


def _bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + os.linesep for row in rows).encode()


def _check_publication(path: Path, rows: list[dict], protected: set[Path]) -> None:
    if path.resolve() in protected:
        raise ValueError(f"V3输出会覆盖输入或保留评测：{path}")
    if path.exists() and path.read_bytes() != _bytes(rows):
        raise FileExistsError(f"V3目标已有不同内容，请为新实验选择独立目录：{path}")


def curate_data(config: dict) -> dict:
    if config.get("data_generation", {}).get("template_profile") != "v3" or "curation" not in config:
        raise ValueError("curate-data需要V3配置，例如 --config configs/v3.yaml。")
    spec = config["curation"]
    maximum, threshold = spec["max_per_case"], spec["near_duplicate_threshold"]
    if type(maximum) is not int or maximum < 1 or not 0 < threshold <= 1:
        raise ValueError("max_per_case须为正整数，near_duplicate_threshold须在(0,1]内。")
    if config["data"].get("fixed_validation_source") or config["data"].get("split_counts") or config["data"]["test_ratio"]:
        raise ValueError("V3主实验使用病例组比例划分，不设置固定验证源、split_counts或test。")
    seed_path = project_path(config["data_generation"]["seed_source"])
    behavior_path, old_source = project_path(spec["behavior_source"]), project_path(spec["control_source"])
    control = load_config(spec["control_config"])
    if any(config[key] != control[key] for key in ("model", "system_prompt", "generation")):
        raise ValueError("新数据与旧数据对照的模型、提示词和生成条件必须相同。")
    left, right = dict(config["training"]), dict(control["training"])
    left.pop("output_dir")
    right.pop("output_dir")
    if left != right or config["project"]["seed"] != control["project"]["seed"]:
        raise ValueError("数据对照实验只能改变数据，训练设置和随机种子须相同。")
    if any(config["data"][key] != control["data"][key] for key in ("evaluation_source", "evaluation_exclusions", "target_mix")):
        raise ValueError("新旧数据实验须保留相同的评测及任务定义。")
    main_dir = project_path(config["data"]["output_dir"])
    if project_path(control["data"].get("fixed_validation_source", "")).resolve() != (main_dir / "validation.jsonl").resolve():
        raise ValueError("旧数据对照须使用本次V3主实验的validation.jsonl。")

    seeds, seed_rejections = seed_candidates(read_jsonl(seed_path))
    behavior = read_jsonl(behavior_path)
    for row in behavior:
        validate_grounding(row)
        if row.get("behavior_version") != "v3-behavior-v1" or row.get("review_status") != "unreviewed":
            raise ValueError("新增行为数据须标明版本和真实的unreviewed状态。")
    blocked = reserved_questions(config)
    for row in seeds + behavior:
        if any(normalize(message["content"]) in blocked for message in row["messages"] if message["role"] == "user"):
            raise ValueError(f"候选与保留评测问题重叠：{row['id']}")
    public_blocked = blocked | {normalize(row["messages"][0]["content"]) for row in seeds + behavior}
    public, public_report = import_public_rows(config, public_blocked, allow_shortfall=True)
    if not public:
        raise ValueError("当前公开源没有可用候选，请检查下载及过滤记录；不自动退回纯模板。")
    public, rewrite_report = rewrite_public_rows(public, config, public_report.get("source", {}).get("sha256"))
    selected, rejected = prune_candidates(behavior + seeds + public, maximum=maximum, threshold=threshold)
    if not any(row.get("origin", "").startswith("public_encyclopedia") for row in selected):
        raise ValueError("V3追加筛选后无公开候选；请先检查过滤条件与原始内容。")
    random.Random(config["project"]["seed"]).shuffle(selected)
    cleaned, _ = clean_records(selected, config)
    splits = split_records(cleaned, config["project"]["seed"], config["data"]["validation_ratio"], 0)
    validation_groups = len({row["split_group"] for row in splits["validation"]})
    if validation_groups < spec["min_validation_groups"]:
        raise ValueError(f"验证病例组只有{validation_groups}个，低于配置要求；请增加不同病例。")
    control_pool, _ = clean_records(read_jsonl(old_source), control)
    control_rows, control_report = matched_control(control_pool, splits["train"], splits["validation"], config["project"]["seed"])
    for row in control_rows:
        if any(normalize(message["content"]) in blocked for message in row["messages"] if message["role"] == "user"):
            raise ValueError(f"旧数据对照与保留评测问题重叠：{row['id']}")

    destination, control_destination = project_path(config["data"]["source"]), project_path(control["data"]["source"])
    if destination.resolve() == control_destination.resolve() or main_dir.resolve() == project_path(control["data"]["output_dir"]).resolve():
        raise ValueError("新旧数据实验须使用独立的生成和处理目录。")
    protected = {seed_path.resolve(), behavior_path.resolve(), old_source.resolve(),
                 *(path.resolve() for path in evaluation_paths(config))}
    publications = [(destination, selected), (control_destination, control_rows)]
    publications += [(main_dir / f"{name}.jsonl", rows) for name, rows in splits.items()]
    control_groups = {row["id"]: row for group in case_groups(control_rows + splits["validation"]) for row in group}
    expected_control = {"train": [control_groups[row["id"]] for row in control_rows],
                        "validation": [control_groups[row["id"]] for row in splits["validation"]]}
    publications += [(project_path(control["data"]["output_dir"]) / f"{name}.jsonl", rows)
                     for name, rows in expected_control.items()]
    for path, rows in publications:
        _check_publication(path, rows, protected)
    write_jsonl(destination, selected)
    write_jsonl(control_destination, control_rows)
    main_stats, control_stats = build_data(config), build_data(control)
    if main_stats["splits"]["validation"]["sha256"] != control_stats["splits"]["validation"]["sha256"]:
        raise ValueError("新旧实验的验证集字节不一致，请检查固定验证源。")
    report = {
        "prepared_at": utc_now(), "curation_version": CURATION_VERSION,
        "inputs_sha256": {str(path): file_hash(path) for path in (seed_path, behavior_path, old_source)},
        "seed_selection_rejections": dict(seed_rejections), "public": public_report, "public_rewrites": rewrite_report,
        "additional_rejections": dict(Counter(row["reason"] for row in rejected)),
        "count": len(selected), "by_task": main_stats["by_task"], "by_origin": main_stats["by_origin"],
        "case_groups": main_stats["case_groups"], "splits": main_stats["splits"],
        "mean_assistant_chars_by_task": main_stats["mean_assistant_chars_by_task"],
        "control": control_report, "validation_is_identical": True,
        "source_sha256": main_stats["source_sha256"], "control_source_sha256": control_stats["source_sha256"],
        "interpretation": "All candidates remain unreviewed. Character overlap, exact-answer caps and bounded checks do not establish medical correctness or semantic independence. Row and task counts match; supervised token budgets can differ.",
    }
    write_jsonl(destination.parent / "selection_rejections.jsonl", rejected)
    write_jsonl(destination.parent / "public_review.jsonl", [row for row in selected if row.get("origin", "").startswith("public_encyclopedia")])
    write_json(destination.parent / "preparation.json", report)
    return report
