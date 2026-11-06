from __future__ import annotations

import re
from dataclasses import asdict, dataclass

DISCLAIMER = "以上仅供健康信息参考，不能替代医生面诊。"


@dataclass
class SafetyDecision:
    category: str = "normal"
    response: str | None = None
    human_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def assess_input(text: str) -> SafetyDecision:
    # Demonstration rules, deliberately distinct from the learned model behavior.
    # They do not constitute a validated triage classifier.
    if re.search(r"我(?:现在|最近|已经)?(?:很)?(?:想|准备|打算|要)(?:要|去)?(?:自杀|结束生命|伤害自己)|我.{0,6}不想活", text):
        return SafetyDecision("crisis", "听到这些我很担心你的安全。请立即联系可信任的人陪在身边，并远离可能伤害自己的物品；可以拨打 12356 寻求心理援助。如已采取行动或有立即危险，请拨打 120 或 110。", True)
    urgent = (
        bool(re.search(r"(胸痛|胸口.{0,4}(疼|痛|压迫))", text)) and bool(re.search(r"(出冷汗|大汗|喘不上气|呼吸困难)", text))
    ) or bool(re.search(r"(突然|刚刚).{0,20}(口角歪斜|一侧.{0,6}无力|说话含糊|言语不清|失去意识)", text))
    if urgent:
        return SafetyDecision("emergency", "这些描述可能涉及急症，请立即拨打 120，说明症状和所在位置，并请身边的人协助。请勿自行驾车或等待在线问答来确定诊断。", True)
    if re.search(r"(加量|减量|停药|换药|一次.{0,5}几片|应该.{0,5}多少毫克)", text):
        return SafetyDecision("prescription", None, True)
    return SafetyDecision()


def review_output(text: str) -> list[str]:
    issues = []
    if re.search(r"(?:每次|每日|每天|一天).{0,12}\d+(?:\.\d+)?\s*(?:毫克|mg|片|粒)", text, re.IGNORECASE):
        issues.append("possible_dosage_instruction")
    if re.search(r"你(?:就是|肯定是|一定是|已经确诊|确定患有)", text):
        issues.append("possible_definitive_diagnosis")
    return issues


def finalize_output(text: str) -> tuple[str, list[str]]:
    issues = review_output(text)
    if issues:
        text = "这段回答可能包含超出健康咨询范围的诊断或剂量建议，需要人工复核。请联系医生或药师确认具体诊疗问题。"
    if DISCLAIMER not in text:
        text = text.rstrip() + "\n\n" + DISCLAIMER
    return text, issues
