import copy
import tempfile
import unittest
from pathlib import Path

from medft.config import load_config
from medft.data import assert_no_leakage, build_data, clean_records, read_jsonl, redact, split_records, validate_messages, write_jsonl


class DataTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.records = read_jsonl(Path(__file__).resolve().parent.parent / "assets/v1_demo_cases.jsonl")

    def test_build_is_reproducible_and_keeps_task_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config["data"]["output_dir"] = directory
            first = build_data(config)
            contents = {name: (Path(directory) / f"{name}.jsonl").read_bytes() for name in ("train", "validation", "test")}
            second = build_data(config)
            self.assertEqual(first, second)
            self.assertEqual([first["splits"][name]["count"] for name in ("train", "validation", "test")], [40, 5, 5])
            self.assertEqual(first["splits"]["train"]["by_task"], {"knowledge_qa": 16, "consultation": 16, "refusal": 8})
            for name, expected in contents.items():
                self.assertEqual((Path(directory) / f"{name}.jsonl").read_bytes(), expected)

    def test_same_case_and_same_opening_stay_together(self):
        variant = copy.deepcopy(self.records[0])
        variant.update(id="variant", case_id="different-case-id")
        variant["messages"][-1]["content"] += "这是不同的回答版本。"
        rows, _ = clean_records(self.records + [variant], self.config)
        splits = split_records(rows, 42, 0.1, 0.1)
        containing = [name for name, records in splits.items() if any(row["id"] == "variant" for row in records)]
        original = [name for name, records in splits.items() if any(row["id"] == self.records[0]["id"] for row in records)]
        self.assertEqual(containing, original)
        assert_no_leakage(splits)

    def test_same_case_with_different_openings_stays_together(self):
        variant = copy.deepcopy(self.records[0])
        variant["id"] = "variant"
        variant["messages"][0]["content"] = "这个病例的另一种问法。"
        rows, _ = clean_records(self.records + [variant], self.config)
        splits = split_records(rows, 42, 0.1, 0.1)
        group = [name for name, records in splits.items() if any(row["id"] == "variant" for row in records)][0]
        self.assertTrue(any(row["id"] == self.records[0]["id"] for row in splits[group]))

    def test_duplicates_removed_without_oversampling(self):
        duplicate = copy.deepcopy(self.records[0])
        duplicate["id"] = "duplicate"
        rows, stats = clean_records(self.records + [duplicate], self.config)
        self.assertEqual(len(rows), 50)
        self.assertEqual(stats["duplicate_dialogues"], 1)

    def test_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "id 重复"):
            clean_records(self.records + [self.records[0]], self.config)

    def test_invalid_role_order_fails(self):
        with self.assertRaises(ValueError):
            validate_messages([{"role": "user", "content": "问"}, {"role": "user", "content": "问"}])

    def test_redaction_preserves_medical_words_and_numbers(self):
        content, count = redact("电话13812345678，证件110101199001011234，邮箱a@example.com。心前区不适，血糖5.6 mmol/L，2026年。")
        self.assertEqual(count, 3)
        self.assertIn("[PHONE]", content)
        self.assertIn("[ID]", content)
        self.assertIn("[EMAIL]", content)
        self.assertIn("心前区", content)
        self.assertIn("5.6 mmol/L", content)
        self.assertIn("2026年", content)

    def test_evaluation_contamination_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config["data"]["output_dir"] = directory
            config["data"]["evaluation_source"] = config["data"]["source"]
            with self.assertRaisesRegex(ValueError, "独立评测"):
                build_data(config)

    def test_followup_and_public_alternate_cannot_reuse_evaluation_question(self):
        question = "这是保留的独立评测问题吗？"
        for use_alternate in (False, True):
            with self.subTest(use_alternate=use_alternate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = copy.deepcopy(self.config)
                config["data"].update(source=str(root / "source.jsonl"),
                                      evaluation_source=str(root / "eval.jsonl"), output_dir=str(root / "processed"))
                rows = copy.deepcopy(self.records)
                if use_alternate:
                    rows[0]["related_questions"] = [question]
                else:
                    rows[0]["messages"].extend([{"role": "user", "content": question},
                                                {"role": "assistant", "content": "回答内容。"}])
                write_jsonl(root / "source.jsonl", rows)
                write_jsonl(root / "eval.jsonl", [{"messages": [{"role": "user", "content": "另一个首轮问题。"},
                                                                 {"role": "assistant", "content": "追问。"},
                                                                 {"role": "user", "content": question}]}])
                with self.assertRaisesRegex(ValueError, "独立评测"):
                    build_data(config)

    def test_public_alternate_questions_link_case_groups(self):
        rows = copy.deepcopy(self.records)
        rows[0]["related_questions"] = [rows[1]["messages"][0]["content"]]
        splits = split_records(rows, 42, .1, .1)
        locations = {row["id"]: name for name, records in splits.items() for row in records}
        self.assertEqual(locations[rows[0]["id"]], locations[rows[1]["id"]])
        assert_no_leakage(splits)


if __name__ == "__main__":
    unittest.main()
