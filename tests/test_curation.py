import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import yaml

from medft.config import file_hash, load_config, project_path, write_json
from medft.corpus import generate_data
from medft.curation import matched_control, prune_candidates, seed_candidates, curate_data, rewrite_public_rows
from medft.data import assert_no_leakage, build_data, read_jsonl, split_records, write_jsonl
from medft.public_filter import PUBLIC_FILTER_VERSION, PUBLIC_SOURCE, PUBLIC_REVISION, public_answer_hash
from medft.quality import validate_grounding
from medft.training import encode_splits, load_training_data, plan_training
from scripts.build_v3_behavior_seeds import build_seeds


def record(key, *, task="knowledge_qa", family=None, question=None, answer=None):
    return {"id": key, "case_id": family or key, "task": task, "source": "test_fixture", "review_status": "unreviewed",
            "messages": [{"role": "user", "content": question or f"测试问题{key}是什么？"},
                         {"role": "assistant", "content": answer or f"这是测试回答{key}，不代表医学资料。"}]}


class CurationTests(unittest.TestCase):
    def test_behavior_seeds_are_reproducible_and_constraints_catch_regression(self):
        rows = build_seeds()
        self.assertEqual(rows, read_jsonl(project_path("assets/v3_behavior_seeds.jsonl")))
        self.assertTrue(all(row["review_status"] == "unreviewed" for row in rows))
        changed = next(row for row in rows if row["id"] == "v3-behavior-meal-changed")
        changed["messages"][-1]["content"] = "先问一下情况稳定时是否可以等几天。"
        with self.assertRaisesRegex(ValueError, "行为样本约束"):
            validate_grounding(changed)

    def test_state_selection_reduces_templates_and_omits_information_cards(self):
        rows, rejected = seed_candidates(read_jsonl(project_path("assets/v2_1_seeds.jsonl")))
        self.assertGreater(rejected["unselected_information_state_or_prefix"], 0)
        self.assertFalse(any(row.get("known_fields") for row in rows))
        self.assertEqual(sum(row["task"] == "report_interpretation" for row in rows), 180)
        self.assertLessEqual(max(Counter(row["case_id"] for row in rows).values()), 7)

    def test_case_caps_exact_answers_and_numeric_near_duplicates(self):
        rows = [record("a", family="same"), record("b", family="same"), record("c", answer="这是测试回答a， 不代表医学资料！")]
        selected, rejected = prune_candidates(rows, maximum=1, threshold=0.88)
        self.assertEqual([row["id"] for row in selected], ["a"])
        self.assertEqual(Counter(row["reason"] for row in rejected), {"case_cap": 1, "identical_final_answer_within_task": 1})
        text = "这是用于测试字符相似度的长段落，涉及检查记录和原始资料，不提供任何医学结论。" * 5
        public = [record("p1", question="血常规第10号是什么？", answer=text + "测试编号10。"),
                  record("p2", question="血常规第11号是什么？", answer=text + "测试编号11。"),
                  record("p3", question="请问血常规第10号是什么？", answer=text + "测试编号10，附注。")]
        for row in public:
            row["origin"] = "public_encyclopedia"
        selected, rejected = prune_candidates(public, maximum=8, threshold=0.88)
        self.assertEqual([row["id"] for row in selected], ["p1", "p2"])
        self.assertEqual(rejected[0]["reason"], "public_character_near_duplicate")

    def test_answer_families_stay_together_across_tasks(self):
        rows = [record(f"k{i}") for i in range(10)] + [record(f"s{i}", task="triage") for i in range(10)]
        rows[0]["answer_group"] = rows[10]["answer_group"] = "same-supervised-answer"
        rows[1]["answer_group"] = rows[11]["answer_group"] = "another-shared-answer"
        splits = split_records(rows, 42, 0.2, 0)
        locations = {row["id"]: split for split, group in splits.items() for row in group}
        self.assertEqual(locations["k0"], locations["s0"])
        assert_no_leakage(splits)

    def test_matched_control_excludes_case_alternate_question_and_answer(self):
        val = [record("v", question="保留问题？", answer="保留回答。")]
        pool = [record("case", family="v"), record("opening", question="保留问题？"),
                record("answer", answer="保留回答。"), record("alternate"), record("ok")]
        pool[3]["related_questions"] = ["保留问题？"]
        rows, report = matched_control(pool, [record("target")], val, 42)
        self.assertEqual([row["id"] for row in rows], ["ok"])
        self.assertEqual(report["rejected"]["held_out_case_opening_or_answer"], 4)
        with self.assertRaisesRegex(ValueError, "无法匹配"):
            matched_control(pool, [record("t1"), record("t2")], val, 42)

    def test_public_rewrites_bind_original_evidence_and_detect_stale_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/v3.yaml")
            path = Path(directory) / "rewrites.json"
            config["data_generation"]["public"].update(rewrites=str(path), exclusions=None)
            original = record("public", answer="这是可核对的原文摘录，还有不应直接导入的内容。")
            original.update(source=PUBLIC_SOURCE, source_line=10, public_filter_version=PUBLIC_FILTER_VERSION,
                            related_questions=[original["messages"][0]["content"]])
            catalog = {"repo_id": PUBLIC_SOURCE, "source_revision": PUBLIC_REVISION, "source_sha256": "a" * 64,
                       "review_kind": "engineering_rewrite_not_clinical_review", "entries": [{
                           "source_line": 10, "original_answer_sha256": public_answer_hash(original["messages"][-1]["content"]),
                           "question": "改编问题是什么？", "answer": "候选回答仍待专业复核。", "evidence": "可核对的原文摘录", "reason": "删除错误延伸。"}]}
            write_json(path, catalog)
            edited, report = rewrite_public_rows([original], config, "a" * 64)
            self.assertFalse(report["clinician_reviewed"])
            self.assertEqual(edited[0]["source_original_messages"], original["messages"])
            from medft.data import clean_records
            self.assertEqual(len(clean_records(edited, config)[0]), 1)
            with self.assertRaisesRegex(ValueError, "改编清单"):
                clean_records([original], config)
            with self.assertRaisesRegex(ValueError, "哈希不匹配"):
                rewrite_public_rows([original], config, "b" * 64)
            catalog["entries"][0]["answer"] += "修改。"
            write_json(path, catalog)
            with self.assertRaisesRegex(ValueError, "改编清单"):
                clean_records(edited, config)
            catalog["entries"][0]["evidence"] = "原文没有的摘录"
            write_json(path, catalog)
            with self.assertRaisesRegex(ValueError, "摘录检查"):
                rewrite_public_rows([original], config, "a" * 64)

    def test_full_preparation_keeps_identical_validation_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = load_config("configs/v3.yaml")
            control = load_config("configs/v3_control.yaml")
            main["data"].update(source=str(root / "new/corpus.jsonl"), output_dir=str(root / "new-processed"))
            main["curation"].update(control_source=str(root / "historical/corpus.jsonl"),
                                    control_config=str(root / "control.yaml"), min_validation_groups=10)
            control["data"].update(source=str(root / "control/corpus.jsonl"), output_dir=str(root / "control-processed"),
                                   fixed_validation_source=str(root / "new-processed/validation.jsonl"))
            for config in (main, control):
                config["data_generation"]["public"].update(exclusions=None, rewrites=None)
            (root / "control.yaml").write_text(yaml.safe_dump(control, allow_unicode=True), encoding="utf-8")
            old = load_config("configs/v2_1.yaml")
            old["data"]["source"] = main["curation"]["control_source"]
            generate_data(old)
            old_hash = file_hash(old["data"]["source"])
            public = [record(f"public-{i}", question=f"血糖测试资料第{i}号有什么意义？",
                             answer=f"这是独立格式测试第{i}号。" + "血糖记录需要保留检测日期和完整原始报告。" * 6) for i in range(80)]
            for row in public:
                row.update(source=PUBLIC_SOURCE, origin="public_encyclopedia", public_filter_version=PUBLIC_FILTER_VERSION)
            with patch("medft.curation.import_public_rows", return_value=(public, {"selected": len(public)})):
                first = curate_data(main)
                second = curate_data(main)
            self.assertEqual(first["source_sha256"], second["source_sha256"])
            self.assertEqual(file_hash(old["data"]["source"]), old_hash)
            left, _ = load_training_data(main)
            right, _ = load_training_data(control)
            self.assertEqual(Counter(row["task"] for row in left["train"]), Counter(row["task"] for row in right["train"]))
            self.assertEqual(left["validation"], right["validation"])
            self.assertEqual((root / "new-processed/validation.jsonl").read_bytes(),
                             (root / "control-processed/validation.jsonl").read_bytes())
            self.assertGreater(first["splits"]["validation"]["case_groups"], 16)
            # An external validation edit invalidates training before loading weights.
            with (root / "new-processed/validation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "固定验证源"):
                load_training_data(control)

    def test_fixed_validation_overlap_and_output_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config()
            rows = read_jsonl(project_path(config["data"]["source"]))
            config["data"].update(source=str(root / "train.jsonl"), fixed_validation_source=str(root / "val.jsonl"),
                                   output_dir=str(root / "out"), test_ratio=0)
            write_jsonl(root / "train.jsonl", rows[:30])
            overlap = copy.deepcopy(rows[0])
            overlap["id"] = "overlap"
            overlap["messages"][-1]["content"] += "不同回答。"
            write_jsonl(root / "val.jsonl", [overlap])
            with self.assertRaisesRegex(ValueError, "数据泄漏"):
                build_data(config)
            config["data"]["output_dir"] = str(root)
            write_jsonl(root / "val.jsonl", rows[30:])
            config["data"]["fixed_validation_source"] = str(root / "validation.jsonl")
            write_jsonl(root / "validation.jsonl", rows[30:])
            original = (root / "validation.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "覆盖源数据"):
                build_data(config)
            self.assertEqual(original, (root / "validation.jsonl").read_bytes())


class PlanTests(unittest.TestCase):
    def test_strict_length_check_prevents_unequal_experiment_data(self):
        from test_formatting import TinyTemplateTokenizer
        rows = {name: [record("long", answer="很长" * 100)] for name in ("train", "validation")}
        with self.assertRaisesRegex(ValueError, "不允许静默丢样本|没有符合长度"):
            encode_splits(rows, TinyTemplateTokenizer(), {"max_length": 5, "require_all_samples": True})

    def test_plan_never_loads_model_and_counts_pinned_scheduler_steps(self):
        config = load_config("configs/v3.yaml")
        summary = {"train": {"used": 83}, "validation": {"used": 20}}
        stats = {"source_sha256": "source", "splits": {"validation": {"sha256": "val"}}}
        with patch("medft.training.load_training_data", return_value=({}, stats)), \
                patch("medft.training.load_tokenizer"), \
                patch("medft.training.encode_splits", return_value=({}, summary)), \
                patch("medft.training.load_quantized_model", side_effect=AssertionError("weights loaded")):
            result = plan_training(config)
        self.assertEqual(result["planned_max_steps"], 10)
        self.assertEqual(result["evaluation_steps"], [10])
        self.assertFalse(result["loads_model_weights"])


if __name__ == "__main__":
    unittest.main()
