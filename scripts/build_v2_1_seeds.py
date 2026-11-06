"""Reproducible V2.1 teaching candidates; no API calls or clinical-review claim."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medft.quality import QUALITY_VERSION, number, report_facts, validate_grounding
from scripts.build_v2_seeds import REPORTS, build_seeds as old_seeds, scenarios

COUNTS = {"knowledge_qa": 600, "symptom_collection": 2250, "triage": 1800,
          "examination": 900, "report_interpretation": 1350, "refusal": 900}
SAME_DAY = {21, 22, 24, 27, 34, 37, 38, 43}
PROFILES = [(31, "5天", 0), (46, "10天", 1), (58, "4天", 0), (72, "3周", 1), (27, "2天", 0)]


def record(identifier, family, task, texts, *, required=(), forbidden=(), **meta):
    return {"id": identifier, "case_id": family, "task": task,
            "source": "author_created_v2_1_teaching_seed", "review_status": "unreviewed",
            "seed_asset_version": QUALITY_VERSION, **meta,
            "grounding_checks": {"required": list(required), "forbidden": list(forbidden)},
            "messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": text}
                         for i, text in enumerate(texts)]}


def urgent(event, department, index, style=0):
    if index == 19:
        return ("现在先远离可能伤害自己的物品，让可信任的人立即到身边陪同，不要独处。"
                "存在即时行动风险时立即联系120或110，说明当前位置和刚才的情况。")
    if index in SAME_DAY:
        return (f"你补充的“{event}”使原先的等待安排不再适用。今天尽快到{department}或急诊评估；"
                "如果状态迅速变差、晕厥或明显呼吸困难，立即联系120。")
    action = ["请立即联系120或急诊，不要继续等待普通门诊。",
              "这些新变化需要立即求助，请联系120，不等待资料整理完。",
              "现在应立即联系120获取急救帮助，请身边人陪同。 "][style % 3].strip()
    timing = "记录首次出现或最后正常的时间，并告诉急救人员。" if any(w in event for w in ("言语", "说话", "一侧无力", "口角")) else "说明发生时间和已知变化，按急救人员指导行动。"
    return f"你刚说明“{event}”。{action}{timing}不要自行驾车。"


def knowledge_records(old):
    rows = []
    for index, seed in enumerate((r for r in old if r["task"] == "knowledge_qa"), 1):
        family = f"v21-k{index:03}"
        rows.append(record(family, family, "knowledge_qa", [m["content"] for m in seed["messages"]], derived_from_seed_id=seed["id"]))
    foods = ["谷物片", "面包", "酸奶", "饼干", "豆制品", "蔬菜脆", "坚果混合物", "汤料", "米制品", "玉米制品"]
    portions = [12, 25, 35, 60, 75, 110, 150]
    for nutrient_index, (nutrient, unit) in enumerate((("钠", "毫克"), ("蛋白质", "克"), ("膳食纤维", "克"))):
        for food_index, food in enumerate(foods):
            family = f"v21-label-{nutrient_index}-{food_index}"
            per_100 = Decimal(80 + food_index * 17) if nutrient_index == 0 else Decimal("1.2") + Decimal(food_index) * Decimal("0.7")
            for slot, portion in enumerate(portions):
                result = per_100 * portion / 100
                question = (f"模拟{food}包装写每100克含{nutrient}{number(per_100)}{unit}，这次称取{portion}克。"
                            "请按实际份量换算，只解释这份标签，不判断我的病情。")
                answer = (f"实际份量为{portion}克，按{number(per_100)}×{portion}÷100计算，本次摄入{number(result)}{unit}{nutrient}。"
                          "这是模拟标签的份量换算，不能单凭这一份食物判断全天摄入或个人疾病。")
                rows.append(record(f"{family}-{slot}", family, "knowledge_qa", [question, answer],
                                   calculation={"per_100": float(per_100), "portion": portion, "result": float(result), "unit": unit},
                                   required=[f"实际份量为{portion}克"]))
    # These cards share the clinical scenario family, so they cannot leak across splits.
    for index, scene in enumerate(scenarios()[:30], 1):
        family = f"v21-s{index:03}"
        symptom, _, contexts, *_ = scene
        for slot in range(7):
            age, duration = 32 + index, f"{slot + 2}天"
            known = [f"主诉为{symptom}"]
            unknown = []
            if slot not in (0, 3):
                known.append(f"年龄为{age}岁")
            else:
                unknown.append("年龄")
            if slot not in (1, 3):
                known.append(f"持续时间为{duration}")
            else:
                unknown.append("持续时间")
            if slot in (2, 4, 6):
                known.append(f"补充情况为{contexts[slot % 2]}")
            else:
                unknown.append("诱因")
            if slot in (4, 5):
                known.append("用药情况已明确为未使用新药")
            else:
                unknown.append("用药情况")
            question = ("这是一张模拟就诊信息卡：" + "；".join(known) + "。其他字段没有提供。"
                        "请分清已经知道和还没知道的信息，不补写疾病、危险表现或服药史。")
            answer = "已知：" + "；".join(known) + "。尚未提供：" + "、".join(unknown) + "。未提供不能解释为已经排除，整理时保留未知字段。"
            rows.append(record(f"{family}-facts-{slot}", family, "knowledge_qa", [question, answer],
                               known_fields=known, unknown_fields=unknown, required=known + ["尚未提供：" + "、".join(unknown)]))
    return rows


def scenario_records():
    rows = []
    branches = ("stable", "changed", "missing_duration", "missing_age", "unknown_history",
                "corrected_age", "corrected_duration", "corrected_context", "persistent", "conditional_risk")
    for index, (symptom, question, contexts, advice, flags, department, exam, event) in enumerate(scenarios(), 1):
        family = f"v21-s{index:03}"
        for slot, (age, duration, ci) in enumerate(PROFILES, 1):
            context = contexts[ci]
            summary = f"{age}岁，{symptom}持续{duration}，{context}。"
            for branch in branches:
                opening = f"{symptom}，我想先把情况说清楚。"
                detail = f"我{age}岁，持续{duration}，{context}；目前较轻，没有你问到的危险表现，没有其他新用药。"
                plan = f"目前你说明的是{duration}、{context}且没有上述危险表现。{advice}"
                urgency, required = "outpatient", []
                if branch == "stable":
                    last = "今天休息后有减轻，仍没有新的不适，后面怎样记录？"
                    answer = f"减轻和没有新症状是这次更新的信息。可记录发生时间、活动关系和功能影响，{advice}若之后出现{flags}，应提前就医。"
                elif branch == "changed":
                    last, answer = f"情况变了：{event}。", urgent(event, department, index, slot)
                    urgency = "same_day" if index in SAME_DAY else "emergency"
                elif branch == "missing_duration":
                    detail = f"我{age}岁，{context}，起病时间记不清，还没有确认病程。"
                    plan = f"年龄和{context}已提供，但持续时间尚未提供；还需要明确病程和有无{flags}。"
                    last = "开始时间仍不清楚，能把它写成两周吗？"
                    answer = f"持续时间尚未提供，不能把两周当成既定病史。可回想首次注意到不适的活动或日期，并确认有无{flags}，再安排就诊。"
                    urgency, required = "undetermined", ["持续时间尚未提供"]
                elif branch == "missing_age":
                    detail = f"替家人咨询，{symptom}持续{duration}，{context}，但年龄还没问到。"
                    plan = f"病程和{context}已给出，年龄尚未提供，不把咨询者年龄当成家人年龄。还要核实有无{flags}。"
                    last = "家人的年龄仍没核实，是否可以按我自己的年龄记？"
                    answer = "年龄尚未提供，不能把你的年龄替代为家人年龄。先向本人核实，暂时保留为未知；同时按实际情况补充症状和用药。"
                    urgency, required = "undetermined", ["年龄尚未提供"]
                elif branch == "unknown_history":
                    detail = f"{summary}症状较轻，不过病史和药物名称我不清楚。"
                    plan = f"你已说明{duration}和{context}，但病史与用药尚未确认；不能据此记录为没有基础病或没有服药。"
                    last = "病史和药物还没问清，能先写没有吗？"
                    answer = "病史与用药尚未确认，应保留未知而不是写没有。可核对药盒、既往资料并向本人询问；若有急迫变化，先就医，不等资料齐全。"
                    urgency, required = "undetermined", ["病史与用药尚未确认"]
                elif branch == "corrected_age":
                    last = f"更正一下，患者年龄是{age + 4}岁，前面报错了，其余情况没变。"
                    answer = f"患者年龄更正为{age + 4}岁，采用这次核实的信息，不沿用先前年龄。{advice}可把更正和发生时间记入就诊资料。"
                    required = [f"年龄更正为{age + 4}岁"]
                elif branch == "corrected_duration":
                    last = f"确认过了，实际持续{slot + 6}天，之前的病程说错，其余情况没变。"
                    answer = f"病程更正为{slot + 6}天，应以这次确认的信息记录。{advice}是否需要加快就诊还应结合症状程度及有无{flags}。"
                    required = [f"病程更正为{slot + 6}天"]
                elif branch == "corrected_context":
                    last = f"刚问清楚，实际情况是{contexts[1 - ci]}，不是前面所说的情况；症状仍较轻。"
                    answer = f"补充情况更正为“{contexts[1 - ci]}”。时间上的先后不等于已确定病因。{advice}后续用更正后的资料向医生说明。"
                    required = [contexts[1 - ci]]
                elif branch == "persistent":
                    last = "这些天没有明显加重，但反复影响日常活动，还能一直只记录吗？"
                    answer = f"反复影响日常活动时，不宜一直只靠记录，可安排{department}或全科面诊。带上病程和活动影响，{advice}"
                else:
                    last = f"假如以后出现{flags}应怎么办？现在没有这些表现。"
                    answer = f"你说明现在没有这些表现，不能把假设写成已发生的病史。若之后出现{flags}，应及时就医；明显危急时联系120。现在仍按已知情况记录和评估。"
                    required = ["现在没有这些表现"]
                rows.append(record(f"{family}-collect-{slot}-{branch}", family, "symptom_collection",
                                   [opening, question, detail, plan, last, answer], branch=branch, required=required,
                                   department=department, urgency=urgency,
                                   forbidden=["你已出现意识变化", "你已经昏迷", "你正在服抗凝药"]))
            # Eight distinct information/decision states, rather than four prefixes.
            for branch in ("stable", "changed", "missing_age", "missing_duration", "unknown_history", "risk_unknown", "prior_normal", "third_party"):
                q = summary
                if branch == "stable":
                    q += "症状较轻且无上述危险表现，可以先从什么科室评估？"
                    a = f"可先到{department}或全科门诊评估。{advice}目前没有描述危险表现，不代表之后的变化无需重评。"
                    urgency = "outpatient"
                elif branch == "changed":
                    q += f"现在新出现{event}，之前的门诊安排是否还合适？"
                    a = urgent(event, department, index, slot + 1)
                    urgency = "same_day" if index in SAME_DAY else "emergency"
                else:
                    additional = {"missing_age": "家人转述，年龄未核实，前面只是咨询者的年龄。",
                                  "missing_duration": "更正：开始时间其实没有核实。",
                                  "unknown_history": "基础病与用药尚不清楚。",
                                  "risk_unknown": "是否存在危险表现还没有问过。",
                                  "prior_normal": "过去一次检查未发现明显异常，但这次还没有面诊。",
                                  "third_party": "是朋友转述，哪些情况仍存在还未向患者确认。"}[branch]
                    q += additional + "能保证不着急并直接固定一个科室吗？"
                    reason = {"missing_age": "患者年龄尚未提供", "missing_duration": "病程尚未核实",
                              "unknown_history": "病史与用药尚未确认", "risk_unknown": "危险表现尚未核实",
                              "prior_normal": "既往一次结果不能排除现在的问题", "third_party": "转述内容需要向本人核实"}[branch]
                    a = f"{reason}，不能直接保证不紧急或固定科室。先核实症状变化及有无{flags}。没有危险表现且情况稳定时，可从{department}或全科评估；出现急迫变化先及时求助。"
                    urgency = "undetermined"
                rows.append(record(f"{family}-triage-{slot}-{branch}", family, "triage", [q, a], branch=branch,
                                   department=department, urgency=urgency))
            for branch in ("purpose", "all_tests", "prior_normal", "urgent_change"):
                if branch == "urgent_change":
                    q = summary + f"现在出现{event}，能先等预约检查吗？"
                    a = urgent(event, department, index, slot + 2)
                else:
                    focus = {"purpose": "想了解医生先评估什么，以及检查分别解决什么问题。",
                             "all_tests": "是不是应该直接把影像和化验项目全部做一遍？",
                             "prior_normal": "以前一次检查正常，这次是否就不必再评估？"}[branch]
                    q = summary + focus
                    a = exam
                    if branch == "all_tests":
                        a += "不是所有人都需要同一套检查；先说明最需解决的问题，再结合病史和查体选择。"
                    elif branch == "prior_normal":
                        a += "以前一次正常结果不代表当前所有问题已排除，需结合这次病程和变化判断。"
                rows.append(record(f"{family}-exam-{slot}-{branch}", family, "examination", [q, a], branch=branch))
    return rows


def decimal_value(value):
    return float(value.quantize(Decimal("0.0001")))


def report_records():
    rows = []
    kinds = ("within_range", "below_range", "above_range", "at_low", "at_high", "missing_unit",
             "missing_reference", "missing_previous_unit", "no_previous", "conflicting_summary")
    for index, (name, unit, raw_low, raw_high, _, _, meaning) in enumerate(REPORTS, 1):
        family = f"v21-report{index:03}"
        low, high = Decimal(str(raw_low)), Decimal(str(raw_high))
        span = high - low
        step = max(span * Decimal("0.035"), Decimal("0.0001"))
        for profile, fraction in enumerate((Decimal("0.23"), Decimal("0.47"), Decimal("0.71")), 1):
            for kind in kinds:
                value = low + span * fraction
                if kind == "below_range":
                    value = max(Decimal(0), low - step * profile)
                elif kind == "above_range":
                    value = high + step * profile
                elif kind == "at_low":
                    value = low
                elif kind == "at_high":
                    value = high
                elif kind == "conflicting_summary":
                    value = high + step * (profile + 1) if profile % 2 else max(Decimal(0), low - step)
                previous = max(Decimal(0), value + step * (profile - 2))
                if kind == "below_range" and low == 0:
                    # A nonnegative measurement cannot fall below a zero lower bound.
                    # Keep a real difference in supplied history, not a cosmetic prefix.
                    previous = step * (profile + 5)
                if kind == "at_low":
                    previous = value + step * (profile + 2)
                report = {"item": name, "value": decimal_value(value), "unit": None if kind == "missing_unit" else unit,
                          "reference_low": None if kind == "missing_reference" else raw_low,
                          "reference_high": None if kind == "missing_reference" else raw_high,
                          "reference_unit": None if kind == "missing_reference" else unit,
                          "previous": None if kind == "no_previous" else decimal_value(previous),
                          "previous_unit": None if kind in {"missing_previous_unit", "no_previous"} else unit,
                          "same_method": profile != 3, "fictional_reference_range": True}
                facts = report_facts(report)
                current = f"本次{number(report['value'])}，单位{'未提供' if report['unit'] is None else unit}"
                reference = "本次参考范围未提供" if kind == "missing_reference" else f"本报告参考范围{number(raw_low)}至{number(raw_high)}{unit}（含边界）"
                history = "没有既往数值" if report["previous"] is None else f"上次{number(report['previous'])}，上次单位{'未提供' if report['previous_unit'] is None else unit}"
                method = "两次同一方法" if report["same_method"] else "两次方法是否相同尚不清楚"
                request = "请只根据已给出的字段说明数值位置和比较局限。"
                if kind == "conflicting_summary":
                    wrong = "低于" if facts["direction_label"] == "above" else "高于"
                    request = f"我把本次写成{wrong}参考范围，对吗？请用题中数字核对，不沿用我的判断。"
                question = f"教学模拟报告：{name}，{current}；{reference}；{history}；{method}。{request}"
                answer = f"{facts['unit']}。{facts['direction']}；{facts['trend']}。"
                if kind == "conflicting_summary":
                    answer += "应按数值与上下界的关系更正刚才的判断。"
                elif kind in {"below_range", "above_range"}:
                    answer += meaning
                else:
                    answer += ["这里比较的是已给出的报告字段，不据此确定个人疾病。", "模拟参考范围不能替代个人诊断或治疗目标。", "未确认的信息应保留为未知，不自行补写。"][profile - 1]
                rows.append(record(f"{family}-{profile}-{kind}", family, "report_interpretation", [question, answer],
                                   report=report, branch="lower_boundary" if kind == "below_range" and low == 0 else kind,
                                   direction_label=facts["direction_label"], trend_label=facts["trend_label"]))
    return rows


def refusal_records(old):
    rows = []
    for seed in (r for r in old if r["task"] == "refusal"):
        family = seed["case_id"].replace("v2-", "v21-")
        question, answer = [m["content"] for m in seed["messages"]]
        requests = [question, f"有人提出这样的要求：{question}请说明能否照做及原因。",
                    question + "如果不合适，请说明正规替代途径。", "我以为只是试一下：" + question,
                    "别人说可以这样做：" + question + "请帮我判断这个要求。",
                    question + "请区分可以提供的一般信息和不应代做的事情。",
                    question + "不要补写我没说过的受伤、药名或剂量。",
                    question + "请先明确回应，再说可行的替代方式。",
                    "我准备把这个想法告诉家人：" + question + "你会如何解释风险？",
                    question + "请简明说明边界和下一步。"]
        for frame, text in enumerate(requests):
            rows.append(record(f"{seed['id'].replace('v2-', 'v21-')}-frame{frame}", family, "refusal", [text, answer],
                               derived_from_seed_id=seed["id"], request_frame=frame))
    return rows


def build_seeds():
    old = old_seeds()
    rows = knowledge_records(old) + scenario_records() + report_records() + refusal_records(old)
    assert Counter(r["task"] for r in rows) == COUNTS
    assert len({r["id"] for r in rows}) == len(rows)
    for row in rows:
        validate_grounding(row, required=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "assets/v2_1_seeds.jsonl")
    args = parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in (ROOT / "assets").glob("*.jsonl")} - {(ROOT / "assets/v2_1_seeds.jsonl").resolve()}:
        raise SystemExit("不能覆盖已有其他版本的种子或评测文件。")
    seeds = build_seeds()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeds), encoding="utf-8")
    print(f"{args.output}: {len(seeds)} seeds, {len({r['case_id'] for r in seeds})} families; unreviewed.")
