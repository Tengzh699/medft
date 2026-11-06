import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from medft.config import file_hash, load_config, project_path, write_json
from medft.corpus import generate_data
from medft.data import assert_no_leakage, build_data, read_jsonl, write_jsonl
from medft.quality import QUALITY_VERSION, audit_data, report_facts, validate_grounding
from medft.training import checkpoint_schedule, load_training_data, training_callbacks
from scripts.build_v2_1_seeds import COUNTS, build_seeds


class V21DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seeds = build_seeds()
        cls.by_id = {row["id"]: row for row in cls.seeds}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config("configs/v2_1.yaml")
        self.config["data"].update(source=str(self.root / "generated/corpus.jsonl"), output_dir=str(self.root / "processed"))

    def test_reproducible_seeds_keep_versions_groups_and_review_state(self):
        self.assertEqual(self.seeds, read_jsonl(project_path("assets/v2_1_seeds.jsonl")))
        self.assertEqual(Counter(row["task"] for row in self.seeds), COUNTS)
        self.assertEqual(len({row["case_id"] for row in self.seeds}), 330)
        self.assertTrue(all(row["review_status"] == "unreviewed" and row["seed_asset_version"] == QUALITY_VERSION for row in self.seeds))

    def test_full_build_is_reproducible_exact_and_groups_all_related_tasks(self):
        first = generate_data(self.config)
        self.assertEqual(first["source_sha256"], generate_data(self.config)["source_sha256"])
        self.assertEqual(first["count"], 9000)
        self.assertEqual(first["by_task"], self.config["data_generation"]["synthetic_counts"])
        stats = build_data(self.config)
        self.assertEqual(stats["duplicate_dialogues"], 0)
        self.assertEqual(stats["splits"]["train"]["count"], 8400)
        self.assertEqual(stats["splits"]["validation"]["count"], 600)
        self.assertEqual(stats["splits"]["validation"]["by_task"], {
            "knowledge_qa": 120, "symptom_collection": 150, "triage": 120,
            "examination": 60, "report_interpretation": 90, "refusal": 60})
        rows, _ = load_training_data(self.config)
        assert_no_leakage(rows)
        families = {}
        for split, selected in rows.items():
            for row in selected:
                self.assertEqual(families.setdefault(row["case_id"], split), split)
        for selected in rows.values():
            tasks = {row["task"] for row in selected if row["case_id"] == "v21-s001"}
            if tasks:
                self.assertEqual(tasks, {"knowledge_qa", "symptom_collection", "triage", "examination"})
        checked = audit_data(self.config)
        self.assertTrue(checked["checks_passed"])
        self.assertEqual(checked["versioned_records_checked"], 9000)
        self.assertEqual(checked["report_records_checked"], 1350)
        self.assertEqual(checked["portion_calculations_checked"], 630)
        self.assertEqual(stats, build_data(self.config))

    def test_invalid_new_seed_version_and_public_mix_rejected_before_publication(self):
        self.config["data_generation"]["public"]["count"] = 1
        with self.assertRaisesRegex(ValueError, "公开混合"):
            generate_data(self.config)
        self.config["data_generation"]["public"]["count"] = 0
        broken = copy.deepcopy(self.seeds[0])
        broken.pop("seed_asset_version")
        path = self.root / "seeds.jsonl"
        write_jsonl(path, [broken])
        self.config["data_generation"]["seed_source"] = str(path)
        with self.assertRaisesRegex(ValueError, "版本缺失"):
            generate_data(self.config)
        self.assertFalse(Path(self.config["data"]["source"]).exists())
        # Historical records keep their original contract.
        validate_grounding(broken)

    def test_unit_history_method_and_boundary_cases(self):
        for branch, expected in (("below_range", "below"), ("above_range", "above"),
                                 ("at_low", "within"), ("at_high", "within"),
                                 ("missing_reference", "unknown"), ("missing_unit", "unknown")):
            row = self.by_id[f"v21-report001-1-{branch}"]
            self.assertEqual(report_facts(row["report"])["direction_label"], expected)
            validate_grounding(row, required=True)
        previous_unit = self.by_id["v21-report001-1-missing_previous_unit"]
        facts = report_facts(previous_unit["report"])
        self.assertEqual(facts["unit"], "本次单位已提供，为g/L")
        self.assertEqual(facts["trend"], "上次单位未提供，暂不能直接比较趋势")
        method = report_facts(self.by_id["v21-report001-3-within_range"]["report"])
        self.assertEqual(method["trend_label"], "unknown")
        no_previous = report_facts(self.by_id["v21-report001-1-no_previous"]["report"])
        self.assertIn("既往数值未提供", no_previous["trend"])
        clamped = [row for row in self.seeds if row.get("branch") == "lower_boundary"]
        self.assertEqual(len(clamped), 18)
        self.assertTrue(all(row["direction_label"] == "within" and row["report"]["value"] == 0 for row in clamped))

    def test_incorrect_report_fields_and_contradictory_answers_are_rejected(self):
        original = self.by_id["v21-report001-1-below_range"]
        for field, value in (("value", 107), ("unit", "mg/dL"), ("previous", 1),
                             ("reference_low", 80), ("previous_unit", None), ("same_method", False)):
            with self.subTest(field=field):
                broken = copy.deepcopy(original)
                broken["report"][field] = value
                with self.assertRaisesRegex(ValueError, "字段与输入"):
                    validate_grounding(broken, required=True)
        for contradiction in ("本次缺少单位", "缺少单位和方法", "一次高于参考范围"):
            with self.subTest(contradiction=contradiction):
                broken = copy.deepcopy(original)
                broken["messages"][-1]["content"] += contradiction
                with self.assertRaises(ValueError):
                    validate_grounding(broken, required=True)
        broken = copy.deepcopy(original)
        broken["messages"][-1]["content"] = broken["messages"][-1]["content"].replace("低于", "高于")
        with self.assertRaisesRegex(ValueError, "方向或趋势"):
            validate_grounding(broken, required=True)

    def test_portion_calculation_rejects_changed_numbers_and_wrong_result(self):
        original = next(row for row in self.seeds if row.get("calculation"))
        validate_grounding(original, required=True)
        for field, value in (("per_100", 95), ("portion", 20), ("result", 123)):
            broken = copy.deepcopy(original)
            broken["calculation"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "份量换算"):
                validate_grounding(broken, required=True)
        broken = copy.deepcopy(original)
        broken["messages"][-1]["content"] = "本次摄入123毫克钠。实际份量为12克。"
        with self.assertRaisesRegex(ValueError, "份量换算"):
            validate_grounding(broken, required=True)

    def test_new_information_changes_response_without_inventing_history(self):
        base = "v21-s001-collect-1-"
        stable, changed = (self.by_id[base + name] for name in ("stable", "changed"))
        self.assertEqual(stable["messages"][:4], changed["messages"][:4])
        self.assertNotEqual(stable["messages"][4:], changed["messages"][4:])
        self.assertIn("120", changed["messages"][-1]["content"])
        for branch, fragment in (("missing_age", "年龄尚未提供"), ("missing_duration", "持续时间尚未提供"),
                                 ("unknown_history", "病史与用药尚未确认"), ("corrected_age", "年龄更正为35岁"),
                                 ("corrected_duration", "病程更正为7天"), ("conditional_risk", "现在没有这些表现")):
            row = copy.deepcopy(self.by_id[base + branch])
            self.assertIn(fragment, row["messages"][-1]["content"])
            row["messages"][-1]["content"] = row["messages"][-1]["content"].replace(fragment, "省略")
            with self.subTest(branch=branch), self.assertRaisesRegex(ValueError, "信息一致性"):
                validate_grounding(row, required=True)

    def test_rehashing_a_corrupted_split_cannot_bypass_quality_preflight(self):
        generate_data(self.config)
        build_data(self.config)
        path = self.root / "processed/train.jsonl"
        rows = read_jsonl(path)
        row = next(row for row in rows if row["task"] == "report_interpretation" and row["report"]["unit"])
        row["messages"][-1]["content"] += "本次缺少单位。"
        write_jsonl(path, rows)
        stats_path = self.root / "processed/stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["splits"]["train"]["sha256"] = file_hash(path)
        write_json(stats_path, stats)
        with self.assertRaisesRegex(ValueError, "已给本次单位"):
            load_training_data(self.config)

    def test_changed_source_is_rejected_with_unchanged_splits(self):
        generate_data(self.config)
        build_data(self.config)
        with Path(self.config["data"]["source"]).open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "源数据已"):
            load_training_data(self.config)


class V21TrainingTests(unittest.TestCase):
    def test_early_stopping_parameters_and_smoke_semantics(self):
        spec = load_config("configs/v2_1.yaml")["training"]
        self.assertEqual(checkpoint_schedule(spec)["eval_steps"], 25)
        self.assertEqual(checkpoint_schedule(spec)["save_steps"], 25)
        self.assertEqual(training_callbacks(spec)[0].early_stopping_patience, 3)
        self.assertEqual(training_callbacks(spec, smoke=True), [])
        self.assertEqual(training_callbacks({}), [])
        for invalid in (0, -1, True, 1.5):
            with self.subTest(patience=invalid), self.assertRaisesRegex(ValueError, "patience"):
                training_callbacks({"early_stopping_patience": invalid})
        for invalid in (-1, True, float("nan"), float("inf")):
            with self.subTest(threshold=invalid), self.assertRaisesRegex(ValueError, "threshold"):
                training_callbacks({"early_stopping_patience": 2, "early_stopping_threshold": invalid})
        with self.assertRaisesRegex(ValueError, "一起使用"):
            training_callbacks({"early_stopping_threshold": 0.0})

    def test_tiny_cpu_lora_stops_restores_best_weights_and_resumes_patience(self):
        """Exercise the real Trainer path, including saved callback state; no 7B model or download."""
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainerCallback, TrainingArguments
        from medft.formatting import AssistantOnlyCollator

        class ControlledValidation(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics, **kwargs):
                metrics["eval_loss"] = 1.0 + 0.2 * (state.global_step - 1)

        def model():
            torch.manual_seed(42)
            base = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                bos_token_id=1, eos_token_id=2, pad_token_id=0, use_sliding_window=False, sliding_window=None))
            return get_peft_model(base, LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))

        features = [{"input_ids": [1, 5, 6, 7, 8, 2], "labels": [-100, -100, -100, 7, 8, 2]}] * 8
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory() as directory:
            def trainer():
                args = TrainingArguments(
                    output_dir=directory, use_cpu=True, report_to=[], per_device_train_batch_size=1,
                    max_steps=5, learning_rate=0.01, lr_scheduler_type="constant", optim="adamw_torch",
                    eval_strategy="steps", save_strategy="steps", eval_steps=1, save_steps=1,
                    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                    restore_callback_states_from_checkpoint=True, label_names=["labels"], remove_unused_columns=False,
                    dataloader_pin_memory=False, disable_tqdm=True, seed=42)
                return Trainer(model=model(), args=args, train_dataset=features, eval_dataset=features,
                               data_collator=AssistantOnlyCollator(0), callbacks=[ControlledValidation(), *training_callbacks({"early_stopping_patience": 2})])

            first = trainer()
            first.train()
            self.assertEqual(first.state.global_step, 3)
            self.assertEqual(first.state.best_metric, 1.0)
            self.assertEqual(Path(first.state.best_model_checkpoint).name, "checkpoint-1")
            best = load_file(str(Path(directory) / "checkpoint-1/adapter_model.safetensors"))
            last = load_file(str(Path(directory) / "checkpoint-3/adapter_model.safetensors"))
            current = get_peft_model_state_dict(first.model)
            self.assertEqual(set(current), set(best))
            self.assertTrue(all(torch.equal(current[key], best[key]) for key in best))
            self.assertTrue(any(not torch.equal(best[key], last[key]) for key in best))
            state = json.loads((Path(directory) / "checkpoint-2/trainer_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"]["early_stopping_patience_counter"], 1)
            resumed = trainer()
            resumed.train(resume_from_checkpoint=str(Path(directory) / "checkpoint-2"))
            self.assertEqual(resumed.state.global_step, 3)
            self.assertEqual(Path(resumed.state.best_model_checkpoint).name, "checkpoint-1")
            self.assertTrue(all(torch.equal(get_peft_model_state_dict(resumed.model)[key], best[key]) for key in best))


if __name__ == "__main__":
    unittest.main()
