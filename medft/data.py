from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .config import evaluation_paths, file_hash, project_path, write_json
from .public_filter import assert_current_public_filter
from .quality import validate_grounding

PII_PATTERNS = (
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<![\dA-Za-z])\d{17}[\dXx](?![\dA-Za-z])"), "[ID]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
)


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {number} 行不是合法 JSON。") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {number} 行必须为 JSON 对象。")
            records.append(record)
    if not records:
        raise ValueError(f"数据文件为空：{path}")
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()


def reserved_questions(config: dict) -> set[str]:
    return {normalize(message["content"]) for path in evaluation_paths(config) for row in read_jsonl(path)
            for message in row["messages"] if message["role"] == "user"}


def redact(text: str) -> tuple[str, int]:
    changes = 0
    for pattern, replacement in PII_PATTERNS:
        text, count = pattern.subn(replacement, text)
        changes += count
    return text, changes


def validate_messages(messages: list[dict], *, needs_answer: bool = True) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须为非空列表。")
    body = messages[1:] if messages[0].get("role") == "system" else messages
    if not body:
        raise ValueError("对话不能只有 system 消息。")
    for index, message in enumerate(body):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected:
            raise ValueError(f"第 {index + 1} 条正文消息应为 {expected}，对话须交替出现。")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("消息 content 必须为非空字符串。")
    expected_last = "assistant" if needs_answer else "user"
    if body[-1]["role"] != expected_last:
        raise ValueError(f"该对话必须以 {expected_last} 结尾。")


def _opening(messages: list[dict]) -> str:
    return normalize(next(message["content"] for message in messages if message["role"] == "user"))


def _group_keys(record: dict) -> list[tuple[str, str]]:
    # Public alternate questions identify the same knowledge; generic follow-ups do not.
    keys = [("case", record["case_id"]), ("opening", _opening(record["messages"]))] + [
        ("opening", normalize(question)) for question in record.get("related_questions", [])
    ]
    if record.get("answer_group"):
        keys.append(("answer", record["answer_group"]))
    return keys


def clean_records(records: list[dict], config: dict) -> tuple[list[dict], dict]:
    assert_current_public_filter(records, config)
    cleaned, seen_ids, seen_dialogues = [], set(), set()
    stats = {"input_count": len(records), "duplicate_dialogues": 0, "pii_replacements": 0}
    tasks = config["data"]["target_mix"]
    for record in records:
        for key in ("id", "case_id", "task", "messages", "source"):
            if not record.get(key):
                raise ValueError(f"样本缺少 {key}：{record.get('id', '<unknown>')}")
        if record["id"] in seen_ids:
            raise ValueError(f"样本 id 重复：{record['id']}")
        seen_ids.add(record["id"])
        if record["task"] not in tasks:
            raise ValueError(f"未知任务 {record['task']}；请先修改配置中的任务定义。")
        related = record.get("related_questions", [])
        if not isinstance(related, list) or any(not isinstance(q, str) or not q.strip() for q in related):
            raise ValueError("related_questions须为非空问题字符串组成的列表。")
        validate_messages(record["messages"])
        validate_grounding(record, required=config.get("data_generation", {}).get("template_profile") == "v2_1")
        body = record["messages"]
        if body[0]["role"] == "system":
            body = body[1:]
        messages = [{"role": "system", "content": config["system_prompt"]}]
        for message in body:
            content, replacements = redact(message["content"].strip())
            stats["pii_replacements"] += replacements
            messages.append({"role": message["role"], "content": content})
        fingerprint = tuple((message["role"], normalize(message["content"])) for message in messages)
        if fingerprint in seen_dialogues:
            stats["duplicate_dialogues"] += 1
            continue
        seen_dialogues.add(fingerprint)
        cleaned.append({**record, "messages": messages})
    if not cleaned:
        raise ValueError("清洗后没有可用样本。")
    stats["clean_count"] = len(cleaned)
    return cleaned, stats


def _exact_validation_groups(strata: dict, target: int) -> dict:
    # Subset sums within each task stratum keep whole case families together.
    total = sum(len(rows) for groups in strata.values() for rows in groups)
    options = {}
    for task, groups in strata.items():
        reachable = {0: ()}
        for index, rows in enumerate(groups):
            for size, selected in list(reachable.items()):
                size += len(rows)
                if size <= target and size not in reachable:
                    reachable[size] = selected + (index,)
        options[task] = {size: chosen for size, chosen in reachable.items() if len(chosen) < len(groups)}
    # Find an exact total, preferring the closest available per-task proportions.
    states = {0: (0.0, {})}
    for task, groups in strata.items():
        desired = sum(map(len, groups)) * target / total
        next_states = {}
        for subtotal, (cost, choices) in states.items():
            for size, selected in options[task].items():
                combined = subtotal + size
                if combined > target:
                    continue
                score = cost + (size - desired) ** 2 / max(desired, 1)
                if combined not in next_states or score < next_states[combined][0]:
                    next_states[combined] = (score, {**choices, task: set(selected)})
        states = next_states
    if target not in states:
        raise ValueError(f"保持完整病例组时无法精确划出{target}条验证数据，请调整种子组大小或split_counts。")
    return states[target][1]


def case_groups(records: list[dict]) -> list[list[dict]]:
    """Join case IDs, alternate openings and explicitly marked answer families."""
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, record in enumerate(records):
        for key in _group_keys(record):
            if key in owners:
                parents[find(index)] = find(owners[key])
            owners[key] = index
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    result = []
    for rows in groups.values():
        group_id = hashlib.sha256("|".join(sorted({row["case_id"] for row in rows})).encode()).hexdigest()[:16]
        result.append([{**row, "split_group": group_id} for row in rows])
    return result


def split_records(records: list[dict], seed: int, validation_ratio: float, test_ratio: float,
                  split_counts: dict | None = None) -> dict:
    strata = defaultdict(list)
    for rows in case_groups(records):
        # A case can cover several tasks; keep it intact in one stratum.
        key = tuple(sorted({row["task"] for row in rows}))
        strata[key].append(rows)
    result = {name: [] for name in (("train", "validation", "test") if test_ratio else ("train", "validation"))}
    rng = random.Random(seed)
    for task, task_groups in sorted(strata.items()):
        task_groups.sort(key=lambda rows: rows[0]["split_group"])
        rng.shuffle(task_groups)
    if split_counts is not None:
        if test_ratio or split_counts.get("test", 0):
            raise ValueError("精确条数模式目前只支持train/validation两份。")
        if sum(split_counts.values()) != len(records):
            raise ValueError(f"清洗后{len(records)}条，与split_counts总数{sum(split_counts.values())}不一致；不重复补齐或丢弃。")
        if split_counts["train"] <= 0 or split_counts["validation"] <= 0:
            raise ValueError("train和validation的精确条数必须为正整数。")
        chosen = _exact_validation_groups(dict(sorted(strata.items())), split_counts["validation"])
        for task, task_groups in sorted(strata.items()):
            for index, rows in enumerate(task_groups):
                result["validation" if index in chosen[task] else "train"].extend(rows)
    else:
        _ratio_split(strata, result, validation_ratio, test_ratio)
    for rows in result.values():
        rng.shuffle(rows)
    assert_no_leakage(result)
    return result


def _ratio_split(strata: dict, result: dict, validation_ratio: float, test_ratio: float) -> None:
    for task, task_groups in sorted(strata.items()):
        count = len(task_groups)
        minimum = 3 if test_ratio else 2
        if count < minimum:
            raise ValueError(f"任务组合 {task} 只有 {count} 个独立病例组，至少需要 {minimum} 组才能划分。")
        n_val = max(1, round(count * validation_ratio))
        n_test = max(1, round(count * test_ratio)) if test_ratio else 0
        if n_val + n_test >= count:
            raise ValueError(f"任务组合 {task} 的病例组不足以满足划分比例。")
        boundaries = (n_val, n_val + n_test)
        for name, selected in (
            ("validation", task_groups[:boundaries[0]]),
            ("test", task_groups[boundaries[0]:boundaries[1]]),
            ("train", task_groups[boundaries[1]:]),
        ):
            for rows in selected:
                result[name].extend(rows)


def assert_no_leakage(splits: dict[str, list[dict]]) -> None:
    owners = {}
    for name, rows in splits.items():
        for row in rows:
            for kind, key in _group_keys(row):
                previous = owners.setdefault((kind, key), name)
                if previous != name:
                    raise ValueError(f"数据泄漏：{kind} 在 {previous} 和 {name} 中同时出现。")


def build_data(config: dict) -> dict:
    source = project_path(config["data"]["source"])
    raw = read_jsonl(source)
    fixed_value = config["data"].get("fixed_validation_source")
    fixed_path = project_path(fixed_value) if fixed_value else None
    fixed = read_jsonl(fixed_path) if fixed_path else []
    if fixed and (config["data"]["test_ratio"] or config["data"].get("split_counts")):
        raise ValueError("固定验证集不能与test划分或split_counts一起使用。")
    cleaned, stats = clean_records(raw + fixed, config)
    evaluation_source = project_path(config["data"]["evaluation_source"])
    eval_questions = reserved_questions(config)
    for row in cleaned:
        questions = [message["content"] for message in row["messages"] if message["role"] == "user"]
        questions.extend(row.get("related_questions", []))
        if any(normalize(question) in eval_questions for question in questions):
            raise ValueError("训练数据与独立评测题存在相同的用户问题，请先处理污染。")
    if fixed_path:
        if stats["duplicate_dialogues"]:
            raise ValueError("固定验证集与训练源存在重复对话；请先处理污染。")
        fixed_ids = {row["id"] for row in fixed}
        grouped = {row["id"]: row for group in case_groups(cleaned) for row in group}
        # Preserve the external validation order so its byte hash is comparable.
        splits = {"train": [grouped[row["id"]] for row in cleaned if row["id"] not in fixed_ids],
                  "validation": [grouped[row["id"]] for row in fixed]}
        assert_no_leakage(splits)
    else:
        splits = split_records(cleaned, config["project"]["seed"], config["data"]["validation_ratio"],
                               config["data"]["test_ratio"], config["data"].get("split_counts"))
    output = project_path(config["data"]["output_dir"])
    source_files = {source.resolve(), *(path.resolve() for path in evaluation_paths(config))}
    if fixed_path:
        source_files.add(fixed_path.resolve())
    destinations = {output / f"{name}.jsonl" for name in splits} | {output / "stats.json"}
    if any(path.resolve() in source_files for path in destinations):
        raise ValueError("生成路径会覆盖源数据，请修改 output_dir。")
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    stats.update({
        "seed": config["project"]["seed"],
        "source_sha256": file_hash(source),
        "evaluation_sha256": file_hash(evaluation_source),
        "evaluation_sources_sha256": {str(path.resolve()): file_hash(path) for path in evaluation_paths(config)},
        "target_mix": config["data"]["target_mix"],
        "requested_split_counts": config["data"].get("split_counts"),
        "by_task": dict(Counter(row["task"] for row in cleaned)),
        "by_source": dict(Counter(row["source"] for row in cleaned)),
        "by_origin": dict(Counter(row.get("origin", "unspecified") for row in cleaned)),
        "by_review_status": dict(Counter(row.get("review_status", "unspecified") for row in cleaned)),
        "by_urgency": dict(Counter(row["urgency"] for row in cleaned if row.get("urgency"))),
        "by_department": dict(Counter(row["department"] for row in cleaned if row.get("department"))),
        "case_groups": len({row["split_group"] for rows in splits.values() for row in rows}),
        "assistant_turns_per_dialogue": dict(Counter(
            sum(message["role"] == "assistant" for message in row["messages"]) for row in cleaned)),
        "mean_assistant_chars_by_task": {task: round(
            sum(lengths) / len(lengths), 1) for task in config["data"]["target_mix"]
            if (lengths := [len(message["content"]) for row in cleaned if row["task"] == task
                            for message in row["messages"] if message["role"] == "assistant"])},
        "contains_unreviewed_data": any(row.get("review_status") != "clinician_reviewed" for row in cleaned),
        "splits": {name: {
            "count": len(rows), "by_task": dict(Counter(row["task"] for row in rows)),
            "by_source": dict(Counter(row["source"] for row in rows)),
            "case_groups": len({row["split_group"] for row in rows}),
            "sha256": file_hash(output / f"{name}.jsonl"),
        } for name, rows in splits.items()},
        "pii_scope": "Only mobile phone, 18-character ID and email patterns; not comprehensive de-identification.",
    })
    if fixed_path:
        stats["fixed_validation_source"] = str(fixed_path.resolve())
        stats["fixed_validation_sha256"] = file_hash(fixed_path)
    write_json(output / "stats.json", stats)
    return stats
