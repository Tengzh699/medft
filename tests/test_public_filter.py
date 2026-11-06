import copy
import unittest

from medft.config import load_config
from medft.corpus import _public_candidate
from medft.data import clean_records, read_jsonl
from medft.config import project_path
from medft.public_filter import PUBLIC_FILTER_VERSION, PUBLIC_SOURCE


BIOGRAPHY = (
    "蒋敷，字宗德，应天府江宁县人，医籍，治《礼记》，年三十岁中式正统十三年戊辰科第三甲第二名进士。"
    "正月初六日生，行一，曾祖蒋义甫；祖蒋子中；父蒋彦升，太医院御医；母周氏。慈侍下，妻史氏，弟敌；叙。"
    "由太医院医生中式顺天府乡试第三十一名举人，会试中式第十七名。"
)
GLUCOSE = (
    "空腹血糖反映符合空腹条件时采血当时的血糖浓度，不能代表全天的平均血糖。"
    "糖化血红蛋白通常反映近两到三个月的平均血糖情况，而不是单次进餐后的即时变化。"
    "解读结果应保留检测日期、单位和实验室参考范围，并结合个人病史、症状及其他检查，由医生综合评估。"
)


class PublicScopeTests(unittest.TestCase):
    def candidate(self, question, answer):
        return _public_candidate({"questions": [[question]], "answers": [answer]}, set())

    def test_actual_biography_contamination_is_rejected(self):
        candidate, reason = self.candidate("蒋敷是什么?", BIOGRAPHY)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "non_health_content")

    def test_plant_and_veterinary_entries_do_not_pass_as_health_questions(self):
        for question, answer in (
            ("侧金盏花的形态是什么？", "侧金盏花是一种多年生草本植物，花朵具有不同的颜色，叶片和茎的形态随品种变化。"),
            ("口蹄疫的症状是什么？", "病牛精神沉郁，口腔有水疱，病畜出现流涎和蹄部不适，需要由兽医根据畜群情况进行评估。"),
        ):
            with self.subTest(question=question):
                # Pad with neutral context to exercise relevance, not the length limit.
                candidate, reason = self.candidate(question, answer + "这段条目描述对象的外观与一般特点，内容仅供记录和分类使用。" * 3)
                self.assertIsNone(candidate)
                self.assertEqual(reason, "non_health_content")

    def test_common_knowledge_is_retained_but_generic_medical_words_are_insufficient(self):
        self.assertIsNotNone(self.candidate("空腹血糖和糖化血红蛋白有什么区别？", GLUCOSE)[0])
        self.assertIsNotNone(self.candidate("餐后血糖和空腹血糖有什么区别？", GLUCOSE)[0])
        unrelated = BIOGRAPHY.replace("进士", "考生").replace("乡试", "考试").replace("会试", "考试")
        self.assertIsNone(self.candidate("一位医生的经历是什么？", unrelated)[0])
        self.assertIsNone(self.candidate("血常规是什么？", GLUCOSE)[0])

    def test_incidental_topic_in_an_unrelated_subject_is_not_enough(self):
        answer = "此条目介绍一种特殊耳部改变，名称中虽然含有胆固醇，但这里并不是讨论血脂检查。" + GLUCOSE
        self.assertIsNone(self.candidate("蓝鼓膜与胆固醇肉芽肿是什么？", answer)[0])

    def test_diagnosis_diet_drug_and_pseudoscience_are_outside_this_batch(self):
        for question in ("糖尿病的诊断是什么？", "糖尿病吃什么水果？", "血糖偏高是否正常？", "血压药物有什么作用？"):
            with self.subTest(question=question):
                self.assertIsNone(self.candidate(question, GLUCOSE + "血压波动还可能受测量条件等因素影响。" * 3)[0])
        candidate, reason = self.candidate("超声检查是什么？", "多普勒纳米立体超声是空间火箭制导数字定位技术在医学中的应用。" + GLUCOSE)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "unsupported_or_noisy_content")

    def test_old_public_rows_require_regeneration(self):
        config = load_config()
        row = copy.deepcopy(read_jsonl(project_path(config["data"]["source"]))[0])
        row["source"] = PUBLIC_SOURCE
        with self.assertRaisesRegex(ValueError, "旧筛选规则"):
            clean_records([row], config)
        row["public_filter_version"] = PUBLIC_FILTER_VERSION
        self.assertEqual(len(clean_records([row], config)[0]), 1)


if __name__ == "__main__":
    unittest.main()
