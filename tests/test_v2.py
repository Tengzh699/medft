import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from medft.config import load_config, project_path
from medft.corpus import generate_data
from medft.data import assert_no_leakage, build_data, read_jsonl, write_jsonl
from medft.evaluation import compare_reports, evaluate, score_response
from medft.training import checkpoint_schedule, load_training_data
from scripts.build_v2_seeds import build_seeds
from scripts.build_v2_evaluation import build_cases


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config("configs/v2.yaml")
        self.config["data"].update(source=str(self.root / "generated/corpus.jsonl"), output_dir=str(self.root / "processed"))

    def test_seed_and_evaluation_assets_are_reproducible(self):
        seeds = read_jsonl(project_path("assets/v2_seeds.jsonl"))
        self.assertEqual(seeds, build_seeds())
        self.assertEqual(len(seeds), 2070)
        self.assertEqual(len({row["case_id"] for row in seeds}), 300)
        cases = read_jsonl(project_path("assets/evaluation_v2.jsonl"))
        self.assertEqual(cases, build_cases())
        self.assertEqual(Counter(row["task"] for row in cases), {task: 6 for task in self.config["data"]["target_mix"]})
        self.assertTrue(all(row["review_status"] == "unreviewed" for row in seeds + cases))

    def test_9000_rows_grouped_across_tasks_exact_splits_and_hash_preflight(self):
        first = generate_data(self.config)
        self.assertEqual(first["count"], 9000)
        self.assertEqual(first["by_task"], self.config["data_generation"]["synthetic_counts"])
        self.assertTrue(first["synthetic_only"])
        self.assertEqual(first["source_sha256"], generate_data(self.config)["source_sha256"])
        stats = build_data(self.config)
        self.assertEqual(stats["clean_count"], 9000)
        self.assertEqual(stats["duplicate_dialogues"], 0)
        self.assertEqual(stats["case_groups"], 300)
        self.assertEqual({name: spec["count"] for name, spec in stats["splits"].items()}, {"train": 8400, "validation": 600})
        self.assertEqual(stats["splits"]["validation"]["by_task"],
                         {"knowledge_qa": 120, "symptom_collection": 150, "triage": 120,
                          "examination": 60, "report_interpretation": 90, "refusal": 60})
        self.assertEqual(len(stats["evaluation_sources_sha256"]), 2)
        rows, _ = load_training_data(self.config)
        assert_no_leakage(rows)
        self.assertEqual(stats, build_data(self.config))
        for selected in rows.values():
            for family in {row["case_id"] for row in selected if row["task"] == "symptom_collection"}:
                members = [row for row in selected if row["case_id"] == family]
                self.assertEqual(set(row["task"] for row in members), {"symptom_collection", "triage", "examination"})
                branches = {row["branch"] for row in members if row["task"] == "symptom_collection"}
                self.assertEqual(branches, {"stable", "changed"})

    def test_real_third_turn_changes_action_and_preserves_report_information(self):
        seeds = build_seeds()
        stable = next(row for row in seeds if row["id"] == "v2-s001-collect-1-stable")
        changed = next(row for row in seeds if row["id"] == "v2-s001-collect-1-changed")
        self.assertEqual(stable["messages"][:4], changed["messages"][:4])
        self.assertNotEqual(stable["messages"][4:], changed["messages"][4:])
        self.assertEqual(stable["urgency"], "outpatient")
        self.assertEqual(changed["urgency"], "emergency")
        self.assertIn("120", changed["messages"][-1]["content"])
        self.assertIn("24岁", stable["messages"][3]["content"])
        report = next(row for row in seeds if row["id"] == "v2-report001-1-outside_range")
        self.assertIn("g/L", report["messages"][-1]["content"])
        self.assertIn("低于", report["messages"][-1]["content"])
        self.assertIn("较上次降低", report["messages"][-1]["content"])
        missing = next(row for row in seeds if row["id"] == "v2-report001-1-missing_unit")
        self.assertIsNone(missing["report"]["unit"])
        self.assertIn("补充单位", missing["messages"][-1]["content"])

    def test_original_development_questions_also_remain_reserved(self):
        generate_data(self.config)
        source = read_jsonl(Path(self.config["data"]["source"]))
        excluded = self.root / "reserved.jsonl"
        write_jsonl(excluded, [{"messages": [{"role": "user", "content": source[0]["messages"][0]["content"]}]}])
        self.config["data"]["evaluation_exclusions"] = [str(excluded)]
        with self.assertRaisesRegex(ValueError, "独立评测"):
            build_data(self.config)
        self.assertFalse((self.root / "processed").exists())

    def test_changed_exclusion_list_stops_training_preflight(self):
        generate_data(self.config)
        build_data(self.config)
        extra = self.root / "another_eval.jsonl"
        write_jsonl(extra, [{"messages": [{"role": "user", "content": "另一道保留题。"}]}])
        self.config["data"]["evaluation_exclusions"] = [str(extra)]
        with self.assertRaisesRegex(ValueError, "保留评测"):
            load_training_data(self.config)

    def test_all_required_groups_must_match_and_old_scores_stay_compatible(self):
        case = build_cases()[0]
        weak = score_response(case, "连续用眼是否感到疲劳？")
        self.assertFalse(weak["required_groups_pass"])
        strong = score_response(case, "可调整照明，避免反光，并安排间歇看远休息。")
        self.assertTrue(strong["required_groups_pass"])
        old = read_jsonl(project_path("assets/evaluation_v1.jsonl"))[0]
        self.assertNotIn("required_groups_pass", score_response(old, "不能只凭一个数值。"))
        with self.assertRaisesRegex(ValueError, "required_groups"):
            score_response({"required_groups": {"invalid": []}}, "")

    def test_evaluation_summaries_include_all_group_and_task_results(self):
        generate_data(self.config)
        with patch("medft.evaluation.Generator") as generator:
            generator.return_value.manifest = {"repo_id": "fixture", "resolved_revision": "fixture"}
            generator.return_value.generate.return_value = "仅有用眼两个字，不能因此当作完整建议。"
            base = evaluate(self.config, adapter=None, output_dir=str(self.root / "base"))
            adapter = evaluate(self.config, adapter="fixture-adapter", output_dir=str(self.root / "adapter"))
        self.assertEqual(base["required_groups_case_count"], 36)
        self.assertEqual(base["required_groups_pass_rate"], 0)
        self.assertEqual(base["required_groups_pass_by_task"],
                         {task: {"count": 6, "passed": 0} for task in self.config["data"]["target_mix"]})
        compared = compare_reports(str(self.root / "base"), str(self.root / "adapter"), str(self.root / "comparison.json"))
        self.assertEqual(compared["base"]["required_groups_pass_rate"], 0)

    def test_followup_overlap_is_rejected_before_model_load(self):
        generate_data(self.config)
        source = read_jsonl(Path(self.config["data"]["source"]))
        followup = next(row["messages"][2]["content"] for row in source if len(row["messages"]) > 2)
        path = self.root / "overlap.jsonl"
        write_jsonl(path, [{"id": "reserved", "messages": [{"role": "user", "content": "新首问"},
                                                          {"role": "assistant", "content": "追问"},
                                                          {"role": "user", "content": followup}]}])
        self.config["data"]["evaluation_source"] = str(path)
        with patch("medft.evaluation.Generator") as generator:
            with self.assertRaisesRegex(ValueError, "重叠"):
                evaluate(self.config, adapter=None, output_dir=str(self.root / "evaluation"))
        generator.assert_not_called()

    def test_checkpoint_schedule_keeps_validation_and_saving_in_sync(self):
        from transformers import TrainingArguments

        schedule = checkpoint_schedule(self.config["training"])
        args = TrainingArguments(output_dir=str(self.root / "trainer"), report_to=[],
                                 load_best_model_at_end=True, **schedule)
        self.assertEqual(args.eval_steps, 150)
        self.assertEqual(args.save_steps, 150)
        self.assertEqual(checkpoint_schedule({}, smoke=False), {"eval_strategy": "epoch", "save_strategy": "epoch"})
        self.assertEqual(checkpoint_schedule({"eval_steps": 150}, smoke=True), {"eval_strategy": "no", "save_strategy": "no"})
        with self.assertRaisesRegex(ValueError, "eval_steps"):
            checkpoint_schedule({"eval_steps": 0})


if __name__ == "__main__":
    unittest.main()
