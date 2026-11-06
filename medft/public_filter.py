"""Public candidate scope and recorded quality exclusions; no clinical certification."""
from __future__ import annotations

import hashlib
import json
import re

from .config import project_path

PUBLIC_SOURCE = "FreedomIntelligence/huatuo_encyclopedia_qa"
PUBLIC_REVISION = "2989899dd5bed83cf9bd17cf9fff9889705436e2"
PUBLIC_FILTER_VERSION = "common-health-v2"

# Restrict the first batch to identifiable, common health concepts. A generic word
# such as 医生/医学/症状 is insufficient evidence that a row belongs to this scope.
TOPIC_PATTERNS = {
    name: re.compile(pattern, re.I) for name, pattern in {
        "blood_pressure": r"高血压|低血压|血压",
        "blood_glucose": r"血糖",
        "diabetes": r"糖尿病",
        "hba1c": r"糖化血红蛋白|\bHbA1c\b",
        "blood_lipids": r"血脂|高脂血症|胆固醇|甘油三酯|低密度脂蛋白|高密度脂蛋白",
        "blood_count": r"血常规|(?<!糖化)血红蛋白|白细胞|红细胞|血小板|贫血",
        "inflammation_markers": r"C反应蛋白|C.?反应蛋白|\bCRP\b|血沉",
        "liver_tests": r"肝功能|转氨酶|胆红素|白蛋白",
        "kidney_tests": r"肾功能|肌酐|尿常规|尿蛋白",
        "uric_acid": r"尿酸|痛风",
        "thyroid": r"甲状腺功能减退|甲状腺功能亢进|甲状腺功能|促甲状腺激素|\bTSH\b|甲亢|甲减",
        "ecg": r"心电图",
        "ultrasound": r"超声检查|B超|B型超声",
        "imaging": r"磁共振|核磁|\bMRI\b|\bCT\b|CT检查|CT扫描|X线检查|X光检查",
        "endoscopy": r"胃镜|肠镜",
        "respiratory_infections": r"感冒|流感|咽炎|肺炎|支气管炎",
        "allergic_rhinitis": r"鼻炎",
        "asthma": r"哮喘",
        "skin_allergy": r"湿疹|荨麻疹|皮肤过敏",
        "gastritis": r"胃炎|胃溃疡|胃食管反流|反流性食管炎",
        "helicobacter": r"幽门螺杆菌",
        "bowel_habits": r"便秘|腹泻",
        "sleep": r"失眠|睡眠呼吸暂停|打鼾|鼾症",
        "bones_joints": r"骨质疏松|关节炎|颈椎病|腰椎间盘突出",
        "eyes": r"近视|远视|散光|白内障|青光眼|干眼",
        "dental": r"牙周炎|龋齿",
        "ears": r"中耳炎|耳鸣|眩晕",
    }.items()
}
TOPIC_QUESTION = re.compile(
    r"^(?:(?:请解释|请描述|请介绍|请问|什么是|什么叫|为什么|为何|怎样|怎么|如何|关于|"
    r"普通|慢性|急性|原发性|继发性|老年人|老年|成人|男性|女性|老人|女人|男人|"
    r"二型|2型|一型|1型|直立性|空腹|餐后)[：:\s]*)*(?:"
    + "|".join(pattern.pattern for pattern in TOPIC_PATTERNS.values())
    + r")(?:病)?(?=[?？:：，,]|$|的|是|有|和|与|为什么|为何|怎么|怎样|如何|"
    r"应|该|能|会|可|注意|检查|正常|意义|含义|参考|异常|临床|症状|原因|概述|简介|定义|患者|病人)",
    re.I,
)
NON_HEALTH_CONTENT = re.compile(
    r"进士|乡试|会试|曾祖|太医院御医|慈侍下|毕业于|出生于|医学博士|教授|"
    r"帕金森定理|帕金森定律|嫁妆死亡|医科大学|基金会|委员会|疫情|疫潮|"
    r"灌木|乔木|草本|木本|一年生|多年生|花萼|花冠|模式产地|植物|"
    r"兽医|畜禽|家畜|病牛|病猪|病畜|患畜|生猪|雏鸡"
)
QUESTION_OUT_OF_SCOPE = re.compile(
    r"历史|发展史|背景|是谁|生平|简历|履历|传记|聚集性疫情|"
    r"诊断|确诊|鉴别|检查方法|实验室检查|"
    r"吃什么|喝什么|食物|食谱|食疗|饮食|忌口|水果|蔬菜|"
    r"药理|药物|服用|服药|胶囊|颗粒|口服液|注射液|滴眼液|乳胶剂|"
    r"缓释片|分散片|片(?:的|药理|禁忌|副作用|不良反应|是什么)|禁忌|用什么|"
    r"手术|拔牙|怀孕|孕期|孩子|宝宝|幼儿|患儿|正常值|是不是正常|是否正常|正常吗|多少钱|多久|"
    r"怎么检查|如何检查|哪种枕头|什么枕头|看什么科|偏高|偏低|很高|好转反应|桥本|促甲状腺激素释放激素"
)
UNSUPPORTED_OR_NOISY_CONTENT = re.compile(
    r"寒邪|肝气|脾胃虚弱|湿热|气血|活血|凉血|散瘀|温经|健脾|补气|中成药|"
    r"汤药|经络|阴虚|阳虚|风寒|风热|祛湿|煎服|煎煮|紫苏|"
    r"排除毒素|身体毒素|植物酵素|抗癌食物|酸性体质|碱性体质|"
    r"妹纸|小编|朋友们|广大患者|广大朋友|&(?:gt|lt|nbsp);|<[^>]+>|\ufffd|"
    r"纳米立体|火箭制导|空间火箭|大血球|甲状腺功效|高励磁系统|自然环境日益破坏|"
    r"3型糖尿病|三型糖尿病|不是什么大病|生命没有危险|"
    r"胃癌是由幽门螺杆菌引起|幽门螺旋杆菌阴性排除|大便形式的减少"
)


def topic_matches(question: str, answer: str) -> list[str]:
    """Require the same explicit health topic on both sides; this is not entailment."""
    return [name for name, pattern in TOPIC_PATTERNS.items() if pattern.search(question) and pattern.search(answer)]


def public_answer_hash(answer: str) -> str:
    """Hash the stripped upstream answer, before chat formatting or redaction."""
    return hashlib.sha256(answer.strip().encode("utf-8")).hexdigest()


def load_public_exclusions(config: dict) -> dict | None:
    path = config.get("data_generation", {}).get("public", {}).get("exclusions")
    if not path:
        return None
    path = project_path(path)
    content = path.read_bytes()
    policy = json.loads(content.decode("utf-8-sig"))
    if not isinstance(policy, dict) or (
        policy.get("repo_id"), policy.get("source_revision"), policy.get("review_kind")
    ) != (PUBLIC_SOURCE, PUBLIC_REVISION, "engineering_quality_triage"):
        raise ValueError("公开数据剔除清单的来源、版本或复核类型不匹配。")
    entries = policy.get("entries")
    if not isinstance(entries, list):
        raise ValueError("公开数据剔除清单缺少entries列表。")
    by_hash, by_line = {}, {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("公开数据剔除项必须是对象。")
        answer_hash, line = entry.get("answer_sha256"), entry.get("source_line")
        if (not isinstance(answer_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", answer_hash)
                or type(line) is not int or line < 1
                or any(not isinstance(entry.get(key), str) or not entry[key].strip()
                       for key in ("question", "reason_code", "reason", "evidence"))):
            raise ValueError("公开数据剔除项需有原始行号、完整答案SHA-256、问题、理由与证据。")
        if answer_hash in by_hash or line in by_line:
            raise ValueError("公开数据剔除清单包含重复答案或行号。")
        by_hash[answer_hash] = entry
        by_line[line] = entry
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
            "by_hash": by_hash, "by_line": by_line}


def assert_current_public_filter(records, config: dict | None = None) -> None:
    policy = load_public_exclusions(config) if config is not None else None
    rewrites = load_public_rewrites(config) if config is not None else None
    for row in records:
        if row.get("source") != PUBLIC_SOURCE:
            continue
        if row.get("public_filter_version") != PUBLIC_FILTER_VERSION:
            raise ValueError(
                "公开数据来自旧筛选规则，请先运行 generate-data --config configs/v1_expanded.yaml，"
                "再运行 build-data --config configs/v1_expanded.yaml。"
            )
        derivation = row.get("public_derivation")
        if rewrites is not None or derivation is not None:
            entry = rewrites["by_line"].get(row.get("source_line")) if rewrites else None
            original = row.get("source_original_messages", [])
            if (not isinstance(derivation, dict) or entry is None or len(original) != 2
                    or derivation.get("catalog_sha256") != rewrites["sha256"]
                    or public_answer_hash(original[-1]["content"]) != entry["original_answer_sha256"]
                    or [message["content"] for message in row["messages"] if message["role"] != "system"] != [entry["question"], entry["answer"]]):
                raise ValueError("公开数据改编清单缺失、过期或与正文不符，请重新运行curate-data。")
        if policy is not None:
            answer = next(message["content"] for message in row["messages"] if message["role"] == "assistant")
            if (row.get("public_exclusions_sha256") != policy["sha256"]
                    or public_answer_hash(answer) in policy["by_hash"]):
                raise ValueError(
                    "公开数据未应用当前质量剔除清单，请先运行 generate-data --config configs/v1_expanded.yaml，"
                    "再运行 build-data --config configs/v1_expanded.yaml。"
                )


def load_public_rewrites(config: dict) -> dict | None:
    path = config.get("data_generation", {}).get("public", {}).get("rewrites")
    if not path:
        return None
    path = project_path(path)
    content = path.read_bytes()
    catalog = json.loads(content.decode("utf-8-sig"))
    if (not isinstance(catalog, dict) or
            (catalog.get("repo_id"), catalog.get("source_revision"), catalog.get("review_kind")) !=
            (PUBLIC_SOURCE, PUBLIC_REVISION, "engineering_rewrite_not_clinical_review")
            or not re.fullmatch(r"[0-9a-f]{64}", str(catalog.get("source_sha256", "")))):
        raise ValueError("公开改编清单须绑定固定来源、原文件哈希和工程改编类型。")
    entries, by_line = catalog.get("entries"), {}
    if not isinstance(entries, list) or not entries:
        raise ValueError("公开改编清单需要非空entries。")
    for entry in entries:
        if (not isinstance(entry, dict) or type(entry.get("source_line")) is not int or entry["source_line"] < 1
                or entry["source_line"] in by_line
                or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("original_answer_sha256", "")))
                or any(not isinstance(entry.get(key), str) or not entry[key].strip()
                       for key in ("question", "answer", "evidence", "reason"))):
            raise ValueError("公开改编项需有唯一行号、原答案哈希、完整问答、原文摘录及修改理由。")
        by_line[entry["source_line"]] = entry
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
            "source_sha256": catalog["source_sha256"], "by_line": by_line}
