import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from medft.__main__ import main
from medft.config import load_config, project_path, split_names
from medft.corpus import generate_data
from medft.data import assert_no_leakage, build_data, read_jsonl, split_records
from medft.training import load_training_data
from scripts.build_v1_6000_seeds import build_seeds


class Tutorial6000Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config("configs/v1_6000.yaml")
        self.config["data"]["source"] = str(self.root / "generated/corpus.jsonl")
        self.config["data"]["output_dir"] = str(self.root / "processed")

    def test_seed_asset_matches_reproducible_builder_and_keeps_slot_families(self):
        rows = read_jsonl(project_path(self.config["data_generation"]["seed_source"]))
        self.assertEqual(rows, build_seeds())
        self.assertEqual(Counter(row["task"] for row in rows),
                         {"knowledge_qa": 120, "consultation": 150, "refusal": 75})
        families = Counter(row["case_id"] for row in rows)
        self.assertEqual(Counter(families.values()), {1: 120, 6: 25, 3: 25})
        self.assertTrue(all(row["review_status"] == "unreviewed" for row in rows))

    def test_full_generation_exact_split_reproducibility_and_training_preflight(self):
        first = generate_data(self.config)
        self.assertEqual(first["count"], 6000)
        self.assertEqual(first["by_task"], {"knowledge_qa": 2400, "consultation": 2400, "refusal": 1200})
        self.assertEqual(first["actual_mix"], self.config["data"]["target_mix"])
        self.assertEqual(first["seed_records_count"], 345)
        self.assertEqual(first["case_groups"], 170)
        self.assertEqual(first["public"]["selected"], 0)
        second = generate_data(self.config)
        self.assertEqual(first["source_sha256"], second["source_sha256"])
        stats = build_data(self.config)
        self.assertEqual(stats["duplicate_dialogues"], 0)
        self.assertEqual(stats["clean_count"], 6000)
        self.assertEqual(stats["case_groups"], 170)
        self.assertEqual({name: spec["count"] for name, spec in stats["splits"].items()},
                         {"train": 5600, "validation": 400})
        self.assertEqual(stats["splits"]["validation"]["by_task"],
                         {"knowledge_qa": 160, "consultation": 192, "refusal": 48})
        self.assertFalse((self.root / "processed/test.jsonl").exists())
        rows, checked_stats = load_training_data(self.config)
        self.assertEqual(json.loads(json.dumps(stats)), checked_stats)
        assert_no_leakage(rows)
        self.assertEqual(set(rows), {"train", "validation"})
        self.assertEqual(stats, build_data(self.config))
        families = {row["case_id"] for split in rows.values() for row in split}
        for family in families:
            self.assertEqual(sum(any(row["case_id"] == family for row in split) for split in rows.values()), 1)
        changed = copy.deepcopy(self.config)
        changed["data"]["split_counts"].update(train=5500, validation=500)
        with self.assertRaisesRegex(ValueError, "split_counts"):
            load_training_data(changed)

    def test_count_mismatch_does_not_publish_splits(self):
        generate_data(self.config)
        self.config["data"]["split_counts"]["train"] -= 1
        with self.assertRaisesRegex(ValueError, "不一致"):
            build_data(self.config)
        self.assertFalse((self.root / "processed").exists())

    def test_unreachable_exact_count_does_not_break_case_groups(self):
        rows = [{"id": f"{family}-{i}", "case_id": f"case-{family}", "task": "knowledge_qa",
                 "messages": [{"role": "user", "content": f"问题{family}-{i}"}]}
                for family in range(4) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "无法精确"):
            split_records(rows, 42, 0.1, 0, {"train": 10, "validation": 2, "test": 0})
        splits = split_records(rows, 42, 0.1, 0, {"train": 9, "validation": 3, "test": 0})
        self.assertEqual({name: len(selected) for name, selected in splits.items()}, {"train": 9, "validation": 3})
        assert_no_leakage(splits)

    def test_ratio_mode_can_also_disable_test(self):
        demo = read_jsonl(project_path("assets/v1_demo_cases.jsonl"))
        splits = split_records(demo, 42, .1, 0)
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 45, "validation": 5})
        self.assertEqual(split_names(self.config["data"]), ("train", "validation"))
        assert_no_leakage(splits)

    def test_preview_rejects_disabled_split_even_with_stale_file(self):
        from contextlib import redirect_stderr
        from io import StringIO

        with redirect_stderr(StringIO()) as error:
            result = main(["preview", "--config", "configs/v1_6000.yaml", "--split", "test"])
        self.assertEqual(result, 1)
        self.assertIn("未启用test", error.getvalue())


if __name__ == "__main__":
    unittest.main()
