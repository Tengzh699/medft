import copy
import tempfile
import unittest
from pathlib import Path

from medft.config import ROOT, write_json
from medft.data import read_jsonl
from medft.evaluation import compare_reports, score_response
from medft.safety import DISCLAIMER, assess_input, finalize_output


class SafetyEvaluationTests(unittest.TestCase):
    def test_emergency_and_prescription_routes(self):
        self.assertEqual(assess_input("我胸口疼，还在出冷汗").category, "emergency")
        self.assertTrue(assess_input("降压药我想自行加量").human_review)
        self.assertEqual(assess_input("我现在想伤害自己").category, "crisis")

    def test_educational_questions_not_crisis(self):
        self.assertEqual(assess_input("如何帮助朋友预防自杀？").category, "normal")
        self.assertEqual(assess_input("我想了解如何预防自杀。").category, "normal")

    def test_disclaimer_not_duplicated(self):
        first, issues = finalize_output("可以记录体温和症状。")
        second, _ = finalize_output(first)
        self.assertFalse(issues)
        self.assertEqual(second.count(DISCLAIMER), 1)

    def test_dosage_output_marked_for_review(self):
        text, issues = finalize_output("你每天服用20毫克即可。")
        self.assertIn("possible_dosage_instruction", issues)
        self.assertNotIn("20毫克", text)

    def test_evaluation_dosage_pattern_matches_digits(self):
        cases = read_jsonl(ROOT / "assets/evaluation_v1.jsonl")
        case = next(case for case in cases if case["id"] == "eval-10")
        self.assertTrue(score_response(case, "今晚20毫克。医生。")["forbidden_pattern_hit"])

    def test_incomparable_reports_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {"model": {"repo_id": "same"}, "evaluation_sha256": "same", "system_prompt": "same",
                    "generation": {"max_new_tokens": 256}, "mode": "raw", "seed": 42,
                    "keyword_match_rate": 0.5, "forbidden_pattern_hits": 0, "empty_outputs": 0,
                    "interpretation": "heuristic", "adapter": None}
            write_json(root / "base/summary.json", base)
            changed = copy.deepcopy(base)
            changed["mode"] = "guarded"
            write_json(root / "adapter/summary.json", changed)
            with self.assertRaisesRegex(ValueError, "mode"):
                compare_reports(str(root / "base"), str(root / "adapter"), str(root / "comparison.json"))


if __name__ == "__main__":
    unittest.main()
