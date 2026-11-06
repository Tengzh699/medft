import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from medft.config import file_hash, load_config, write_json
from medft.corpus import (
    PUBLIC_FILE, PUBLIC_REPO, PUBLIC_REVISION, _public_candidate,
    download_public_data, generate_data, import_public_rows,
)
from medft.data import assert_no_leakage, build_data, clean_records, normalize, read_jsonl, write_jsonl
from medft.public_filter import public_answer_hash


def public_example(number):
    return {
        "questions": [[f"血常规的第{number}号检查指标有什么意义？"], [f"请解释血常规的第{number}号指标意义？"]],
        "answers": [f"这是用于格式测试的第{number}号记录。" +
                    "血常规报告的检查指标需要结合项目、单位、实验室参考范围和既往结果来理解。"
                    "单次结果可能受采样条件和身体状态影响，不应只凭一个箭头判断疾病。"
                    "可保留完整报告和检查日期，向接诊医生说明症状及相关病史，由医生结合整体情况评估。"],
    }


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config("configs/v1_expanded.yaml")
        self.config["data"]["source"] = str(self.root / "generated/corpus.jsonl")
        self.config["data"]["output_dir"] = str(self.root / "processed")
        self.config["data_generation"]["public"]["manifest"] = str(self.root / "manifest.json")
        self.config["data_generation"]["public"]["exclusions"] = None

    def make_public_file(self, rows):
        source = self.root / PUBLIC_FILE
        write_jsonl(source, rows)
        card = self.root / "README.md"
        card.write_text("---\nlicense: apache-2.0\n---\nTest fixture only.\n", encoding="utf-8")
        manifest = {
            "repo_id": PUBLIC_REPO, "resolved_revision": PUBLIC_REVISION,
            "filename": PUBLIC_FILE, "path": str(source), "sha256": file_hash(source),
            "declared_license": "apache-2.0", "upstream_split": "train",
            "card_path": str(card), "card_sha256": file_hash(card),
        }
        write_json(self.root / "manifest.json", manifest)
        return manifest

    def make_exclusions(self, raw):
        path = self.root / "exclusions.json"
        policy = {
            "repo_id": PUBLIC_REPO, "source_revision": PUBLIC_REVISION,
            "review_kind": "engineering_quality_triage",
            "entries": [{"source_line": 1, "answer_sha256": public_answer_hash(raw["answers"][0]),
                         "question": raw["questions"][0][0], "reason_code": "fixture_rejection",
                         "reason": "Test fixture: exclude this answer.", "evidence": "Test fixture only."}],
        }
        write_json(path, policy)
        self.config["data_generation"]["public"]["exclusions"] = str(path)
        return path, policy

    def test_800_candidates_reproducible_and_grouped(self):
        self.make_public_file([public_example(i) for i in range(260)])
        first = generate_data(self.config)
        source = Path(self.config["data"]["source"])
        expected = source.read_bytes()
        second = generate_data(self.config)
        self.assertEqual(source.read_bytes(), expected)
        self.assertEqual(first["source_sha256"], second["source_sha256"])
        self.assertEqual(first["count"], 800)
        self.assertEqual(first["by_task"], {"knowledge_qa": 320, "consultation": 320, "refusal": 160})
        self.assertEqual(first["actual_mix"], {"knowledge_qa": .4, "consultation": .4, "refusal": .2})
        self.assertEqual(first["case_groups"], 270)
        self.assertEqual(first["synthetic_case_groups"], 70)
        self.assertEqual(len(read_jsonl(first["public_review_file"])), 200)
        rows = read_jsonl(source)
        self.assertTrue(all(row["review_status"] == "unreviewed" for row in rows))
        consultations = [row for row in rows if row["task"] == "consultation"]
        self.assertEqual(Counter(len(row["messages"]) for row in consultations), {4: 160, 6: 160})
        stats = build_data(self.config)
        self.assertEqual(stats["clean_count"], 800)
        self.assertEqual(stats["case_groups"], 270)
        self.assertEqual(stats["duplicate_dialogues"], 0)
        splits = {name: read_jsonl(self.root / f"processed/{name}.jsonl")
                  for name in ("train", "validation", "test")}
        assert_no_leakage(splits)
        self.assertEqual(sum(map(len, splits.values())), 800)
        for rows in splits.values():
            self.assertEqual(set(row["task"] for row in rows), {"knowledge_qa", "consultation", "refusal"})

    def test_synthetic_only_is_explicit_and_does_not_require_download(self):
        report = generate_data(self.config, synthetic_only=True)
        self.assertEqual(report["count"], 600)
        self.assertTrue(report["synthetic_only"])
        self.assertEqual(report["case_groups"], 70)
        self.assertAlmostEqual(report["actual_mix"]["knowledge_qa"], .2)
        self.assertNotEqual(report["actual_mix"], self.config["data"]["target_mix"])

    def test_nested_questions_keep_pairing_and_provenance(self):
        self.config["data_generation"]["public"]["count"] = 1
        raw = public_example(1)
        self.make_public_file([raw])
        rows, report = import_public_rows(self.config, set())
        self.assertEqual(report["selected"], 1)
        self.assertEqual(rows[0]["messages"][0]["content"], raw["questions"][0][0])
        self.assertEqual(rows[0]["messages"][1]["content"], raw["answers"][0])
        self.assertEqual(len(rows[0]["related_questions"]), 2)
        self.assertEqual(rows[0]["source_revision"], PUBLIC_REVISION)
        self.assertEqual(rows[0]["upstream_split"], "train")
        self.assertEqual(rows[0]["source_line"], 1)

    def test_filters_reject_dose_pii_ambiguity_and_reserved_alternate(self):
        for suffix in ("建议口服治疗。", "每天服用5mg。", "电话13812345678。", "邮箱a@example.com。"):
            with self.subTest(suffix=suffix):
                raw = public_example(1)
                raw["answers"][0] += suffix
                self.assertIsNone(_public_candidate(raw, set())[0])
        raw = public_example(1)
        blocked = {normalize(raw["questions"][1][0])}
        self.assertEqual(_public_candidate(raw, blocked)[1], "reserved_question_overlap")
        raw["answers"].append("另一个不同答案，无法确定问题答案配对。")
        self.assertEqual(_public_candidate(raw, set())[1], "invalid_or_ambiguous_schema")

    def test_import_deduplicates_answers_and_alternate_questions(self):
        self.config["data_generation"]["public"]["count"] = 2
        rows = [public_example(1), public_example(2)]
        duplicate_answer = copy.deepcopy(rows[0])
        duplicate_answer["questions"] = [["血常规的结果为什么需要结合病史？"]]
        duplicate_question = public_example(3)
        duplicate_question["questions"].append(rows[0]["questions"][1])
        self.make_public_file(rows + [duplicate_answer, duplicate_question])
        selected, report = import_public_rows(self.config, set())
        self.assertEqual(len(selected), 2)
        self.assertEqual(report["eligible"], 2)
        self.assertEqual(report["rejected"]["duplicate_answer_or_question"], 2)

    def test_recorded_exclusions_reduce_actual_count_without_backfilling(self):
        raw = [public_example(1), public_example(2)]
        self.make_public_file(raw)
        self.config["data_generation"]["public"]["count"] = 2
        path, _ = self.make_exclusions(raw[0])
        report = generate_data(self.config)
        self.assertEqual(report["public"]["sampled_before_exclusions"], 2)
        self.assertEqual(report["public"]["excluded_count"], 1)
        self.assertEqual(report["public"]["selected"], 1)
        self.assertEqual(report["count"], 601)
        self.assertAlmostEqual(report["actual_mix"]["knowledge_qa"], 121 / 601)
        public = read_jsonl(report["public_review_file"])
        self.assertEqual(public[0]["source_line"], 2)
        self.assertEqual(public[0]["public_exclusions_sha256"], file_hash(path))
        self.assertEqual(public[0]["review_status"], "unreviewed")
        self.assertEqual(build_data(self.config)["clean_count"], 601)

    def test_exclusion_hash_mismatch_does_not_publish_new_corpus(self):
        raw = public_example(1)
        self.make_public_file([raw])
        self.config["data_generation"]["public"]["count"] = 1
        path, policy = self.make_exclusions(raw)
        policy["entries"][0]["answer_sha256"] = "0" * 64
        write_json(path, policy)
        source = Path(self.config["data"]["source"])
        source.parent.mkdir(parents=True)
        source.write_text("previous source", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "答案哈希不匹配"):
            generate_data(self.config)
        self.assertEqual(source.read_text(encoding="utf-8"), "previous source")

    def test_old_or_stale_public_rows_require_current_exclusions(self):
        raw = [public_example(1), public_example(2)]
        self.make_public_file(raw)
        self.config["data_generation"]["public"]["count"] = 2
        old_rows, _ = import_public_rows(self.config, set())
        path, policy = self.make_exclusions(raw[0])
        with self.assertRaisesRegex(ValueError, "质量剔除清单"):
            clean_records(old_rows, self.config)
        current, _ = import_public_rows(self.config, set())
        self.assertEqual(len(clean_records(current, self.config)[0]), 1)
        # A policy change invalidates prepared rows even when their answers remain.
        policy["entries"][0]["reason"] = "Updated triage decision."
        write_json(path, policy)
        with self.assertRaisesRegex(ValueError, "质量剔除清单"):
            clean_records(current, self.config)

    def test_exclusions_are_scoped_to_the_pinned_dataset_revision(self):
        raw = public_example(1)
        self.make_public_file([raw])
        self.config["data_generation"]["public"]["count"] = 1
        path, policy = self.make_exclusions(raw)
        policy["source_revision"] = "different-revision"
        write_json(path, policy)
        with self.assertRaisesRegex(ValueError, "剔除清单的来源、版本"):
            import_public_rows(self.config, set())

    def test_failed_import_preserves_existing_generated_source(self):
        source = Path(self.config["data"]["source"])
        source.parent.mkdir(parents=True)
        source.write_text("existing result", encoding="utf-8")
        self.make_public_file([public_example(1)])
        with self.assertRaisesRegex(ValueError, "目标200"):
            generate_data(self.config)
        self.assertEqual(source.read_text(encoding="utf-8"), "existing result")

    def test_tampered_download_or_wrong_split_fails(self):
        self.config["data_generation"]["public"]["count"] = 1
        manifest = self.make_public_file([public_example(1)])
        Path(manifest["path"]).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "哈希"):
            import_public_rows(self.config, set())
        manifest = self.make_public_file([public_example(1)])
        manifest["upstream_split"] = "test"
        write_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "版本"):
            import_public_rows(self.config, set())

    def test_capacity_and_old_profile_have_actionable_errors(self):
        self.config["data_generation"]["synthetic_counts"]["knowledge_qa"] = 121
        with self.assertRaisesRegex(ValueError, "独立种子"):
            generate_data(self.config, synthetic_only=True)
        with self.assertRaisesRegex(ValueError, "v1_expanded.yaml"):
            generate_data(load_config())

    def test_explicit_download_pins_revision_and_only_uses_train_file_and_card(self):
        manifest = self.make_public_file([public_example(1)])

        def local_download(**kwargs):
            self.assertEqual(kwargs["revision"], PUBLIC_REVISION)
            self.assertEqual(kwargs["repo_type"], "dataset")
            self.assertEqual(kwargs["endpoint"], "https://hf-mirror.com")
            return manifest["path"] if kwargs["filename"] == PUBLIC_FILE else manifest["card_path"]

        with patch("huggingface_hub.hf_hub_download", side_effect=local_download) as download:
            result = download_public_data(self.config, endpoint="https://hf-mirror.com")
        self.assertEqual([call.kwargs["filename"] for call in download.call_args_list], [PUBLIC_FILE, "README.md"])
        self.assertEqual(result["declared_license"], "apache-2.0")
        self.assertEqual(result["sha256"], file_hash(result["path"]))

    def test_hub_failure_preserves_root_cause_and_retry_hint(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        from requests.exceptions import ProxyError

        error = LocalEntryNotFoundError("Cannot locate file on Hub")
        error.__cause__ = ProxyError("proxy connection refused")
        with patch("huggingface_hub.hf_hub_download", side_effect=error):
            with self.assertRaises(RuntimeError) as captured:
                download_public_data(self.config)
        self.assertIn("ProxyError", str(captured.exception))
        self.assertIn("proxy connection refused", str(captured.exception))
        self.assertIn("--transport http", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
