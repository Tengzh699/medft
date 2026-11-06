"""Bounded consistency checks for structured teaching records, not clinical review."""
from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal


QUALITY_VERSION = "v2.1-grounding-v1"


def number(value) -> str:
    text = format(Decimal(str(value)), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def report_facts(report: dict) -> dict[str, str]:
    """Derive statements from supplied fields; do not invent reference ranges."""
    value = report["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("报告数值须为有限数字。")
    low, high = report.get("reference_low"), report.get("reference_high")
    if (low is None) != (high is None):
        raise ValueError("参考上下界须同时给出或同时缺失。")
    if low is not None and (not all(type(n) in (int, float) and math.isfinite(n) for n in (low, high)) or low > high):
        raise ValueError("参考范围无效。")
    previous = report.get("previous")
    if previous is not None and (type(previous) not in (int, float) or not math.isfinite(previous)):
        raise ValueError("既往数值须为有限数字。")
    unit = report.get("unit")
    facts = {"unit": f"本次单位已提供，为{unit}" if unit else "本次单位未提供"}
    if low is None:
        direction = "unknown"
        facts["direction"] = "本次参考范围未提供，不能判断相对本报告范围的位置"
    elif not unit or report.get("reference_unit") != unit:
        direction = "unknown"
        facts["direction"] = "本次单位与参考范围的对应关系尚未确认，暂不判断高低"
    else:
        direction = "below" if value < low else "above" if value > high else "within"
        relation = {"below": "低于", "above": "高于", "within": "处于"}[direction]
        facts["direction"] = f"本次{number(value)}{relation}本报告参考范围{number(low)}至{number(high)}"
    if previous is None:
        trend = "unknown"
        facts["trend"] = "既往数值未提供，不能判断变化趋势"
    elif not report.get("previous_unit"):
        trend = "unknown"
        facts["trend"] = "上次单位未提供，暂不能直接比较趋势"
    elif not unit or report["previous_unit"] != unit:
        trend = "unknown"
        facts["trend"] = "两次数值的单位对应关系尚未确认，暂不能直接比较趋势"
    elif report.get("same_method") is not True:
        trend = "unknown"
        facts["trend"] = "两次检测方法是否相同未确认，不能直接把数字差异当作病情变化"
    else:
        trend = "down" if value < previous else "up" if value > previous else "unchanged"
        facts["trend"] = {"down": "数字较上次降低", "up": "数字较上次升高", "unchanged": "数字与上次相同"}[trend]
    return {**facts, "direction_label": direction, "trend_label": trend}


def validate_grounding(record: dict, *, required: bool = False) -> None:
    """Check explicit facts and assertions; this does not validate free medical text."""
    if record.get("behavior_version") is not None:
        if record["behavior_version"] != "v3-behavior-v1":
            raise ValueError(f"未知行为样本版本：{record.get('id')}")
        checks = record.get("behavior_checks")
        if (not isinstance(checks, dict) or set(checks) != {"required", "forbidden"}
                or any(not isinstance(values, list) or any(not isinstance(text, str) or not text for text in values)
                       for values in checks.values())):
            raise ValueError(f"行为样本约束格式无效：{record.get('id')}")
        answer = record["messages"][-1]["content"]
        if (any(text not in answer for text in checks["required"])
                or any(text in answer for text in checks["forbidden"])):
            raise ValueError(f"行为样本约束检查失败：{record.get('id')}")
    if record.get("seed_asset_version") != QUALITY_VERSION:
        if required:
            raise ValueError(f"V2.1样本版本缺失或过期：{record.get('id')}")
        return
    answer = record["messages"][-1]["content"]
    checks = record.get("grounding_checks", {})
    if not isinstance(checks, dict) or any(
        not isinstance(checks.get(key, []), list) or any(not isinstance(text, str) or not text for text in checks.get(key, []))
        for key in ("required", "forbidden")
    ):
        raise ValueError(f"信息一致性约束格式无效：{record['id']}")
    for fragment in checks.get("required", []):
        if fragment not in answer:
            raise ValueError(f"信息一致性检查失败：{record['id']} 缺少 {fragment}")
    for fragment in checks.get("forbidden", []):
        if fragment in answer:
            raise ValueError(f"信息一致性检查失败：{record['id']} 出现矛盾断言 {fragment}")
    if record["task"] == "report_interpretation":
        report = record["report"]
        facts = report_facts(report)
        question = next(m["content"] for m in record["messages"] if m["role"] == "user")
        current = f"本次{number(report['value'])}，单位{report.get('unit') or '未提供'}"
        reference = ("本次参考范围未提供" if report.get("reference_low") is None else
                     f"本报告参考范围{number(report['reference_low'])}至{number(report['reference_high'])}{report.get('reference_unit')}（含边界）")
        history = ("没有既往数值" if report.get("previous") is None else
                   f"上次{number(report['previous'])}，上次单位{report.get('previous_unit') or '未提供'}")
        method = "两次同一方法" if report.get("same_method") is True else "两次方法是否相同尚不清楚"
        if any(fragment not in question for fragment in (f"教学模拟报告：{report['item']}，", current, reference, history, method)):
            raise ValueError(f"报告字段与输入数值/单位/既往信息不一致：{record['id']}")
        if any(facts[key] not in answer for key in ("unit", "direction", "trend")):
            raise ValueError(f"报告单位、方向或趋势与字段不符：{record['id']}")
        if record["report"].get("unit") and any(word in answer for word in ("本次单位未提供", "本次缺少单位", "缺少单位和方法")):
            raise ValueError(f"已给本次单位却声称缺失：{record['id']}")
        if facts["direction_label"] != "unknown":
            relation = {"below": "低于", "above": "高于", "within": "处于"}[facts["direction_label"]]
            other = {"低于", "高于", "处于"} - {relation}
            if any(phrase in answer for word in other for phrase in (
                f"本次{number(report['value'])}{word}", f"本次{word}参考范围", f"一次{word}参考范围")):
                raise ValueError(f"报告中存在相反方向断言：{record['id']}")
        if record.get("direction_label") != facts["direction_label"] or record.get("trend_label") != facts["trend_label"]:
            raise ValueError(f"报告标签与数值不符：{record['id']}")
    calculation = record.get("calculation")
    if calculation:
        if any(type(calculation.get(key)) not in (int, float) or not math.isfinite(calculation[key]) or calculation[key] < 0
               for key in ("per_100", "portion", "result")) or calculation["portion"] == 0:
            raise ValueError(f"份量换算数字无效：{record['id']}")
        expected = Decimal(str(calculation["per_100"])) * Decimal(str(calculation["portion"])) / 100
        question = next(m["content"] for m in record["messages"] if m["role"] == "user")
        if (f"{number(calculation['per_100'])}{calculation['unit']}，这次称取{number(calculation['portion'])}克" not in question
                or Decimal(str(calculation["result"])) != expected or f"本次摄入{number(expected)}{calculation['unit']}" not in answer):
            raise ValueError(f"份量换算不一致：{record['id']}")


def audit_data(config: dict) -> dict:
    from .config import file_hash, project_path, write_json
    from .training import load_training_data

    rows, stats = load_training_data(config)
    all_rows = [row for split in rows.values() for row in split]
    required = config.get("data_generation", {}).get("template_profile") == "v2_1"
    for record in all_rows:
        validate_grounding(record, required=required)
    reports = [row for row in all_rows if row.get("seed_asset_version") == QUALITY_VERSION and row["task"] == "report_interpretation"]
    versioned = [row for row in all_rows if row.get("seed_asset_version") == QUALITY_VERSION]
    result = {
        "checks_passed": True, "quality_version": QUALITY_VERSION if required else None,
        "count": len(all_rows), "splits": {name: len(split) for name, split in rows.items()},
        "versioned_records_checked": len(versioned),
        "report_records_checked": len(reports),
        "portion_calculations_checked": sum(bool(row.get("calculation")) for row in versioned),
        "explicit_text_constraints_checked": sum(bool(row.get("grounding_checks", {}).get("required") or row.get("grounding_checks", {}).get("forbidden")) for row in versioned),
        "behavior_records_checked": sum(row.get("behavior_version") == "v3-behavior-v1" for row in all_rows),
        "behavior_text_constraints_checked": sum(bool(row.get("behavior_checks", {}).get("required") or row.get("behavior_checks", {}).get("forbidden")) for row in all_rows),
        "by_task": dict(Counter(row["task"] for row in all_rows)),
        "by_origin": dict(Counter(row.get("origin", "unspecified") for row in all_rows)),
        "report_directions": dict(Counter(report_facts(row["report"])["direction_label"] for row in reports)),
        "report_trends": dict(Counter(report_facts(row["report"])["trend_label"] for row in reports)),
        "source_sha256": file_hash(project_path(config["data"]["source"])),
        "splits_sha256": {name: spec["sha256"] for name, spec in stats["splits"].items()},
        "interpretation": "Structured numeric and specified text checks only; not comprehensive hallucination detection or medical review.",
    }
    write_json(project_path(config["data"]["output_dir"]) / "quality.json", result)
    return result
