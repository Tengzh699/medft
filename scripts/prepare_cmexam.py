"""Pinned public exam data, isolated official splits and a frozen final test.

This is a knowledge experiment, not evidence of better clinical consultation.
All source labels remain unreviewed by this project. Never import test answers
into the training corpus or choose examples using model test performance.
"""
from __future__ import annotations

import copy
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from medft.config import file_hash, load_config, project_path, utc_now, write_json
from medft.data import build_data, normalize, read_jsonl
from medft.models import load_tokenizer
from medft.formatting import encode_conversation, SequenceTooLong
from scripts.build_complete_data import safe_publish

REVISION = "fadb22c89beb1b7115dc36460ba792eb96b7b972"
PREFIX = "以下是医学考试学习题，不是患者诊疗请求。请选择唯一最合适的选项并解释依据。"
FILES = {
    "train.csv": "3d6ca5e2499c510956534740c95179c483d9d47d2531e39fc5069333ffd23724",
    "val.csv": "58cc9776b2f17fdc3a235ceec9ae153ee3f617703293dda661c6decbffb1f190",
    "test_with_annotations.csv": "e998f69b3f8108bf08bcb8cc9a2a6ed276afeabde6ba3caa3015789b1f6fd407",
}


def options_of(text):
    lines = text.strip().splitlines()
    if len(lines) != 5:
        raise ValueError("Not five separately labelled options")
    result = {}
    for label, line in zip("ABCDE", lines):
        match = re.fullmatch(label + r"[\s.、．:：]+(.+)", line.strip())
        if not match:
            raise ValueError("Invalid option labels")
        result[label] = match[1].strip()
    if len({normalize(v) for v in result.values()}) != 5:
        raise ValueError("Repeated options")
    return result


def prompt_of(row):
    return PREFIX + "\n题目：" + row["question"] + "\n" + "\n".join(
        k + " " + v for k, v in row["options"].items())


def group_key(question):
    # Conservative: exclude a whole stem, also when only numbers have changed.
    return re.sub(r"[0-9]+(?:[.][0-9]+)?", "#", normalize(question))


def rank(row, salt):
    return hashlib.sha256((salt + row["id"]).encode()).hexdigest()


def known_source_defect(row):
    body = row["question"] + "\n" + "\n".join(row["options"].values())
    if re.search(r"接触角.{0,30}℃", body):
        return "angle_mislabelled_as_temperature"
    # These labels can denote the same condition; do not teach an arbitrary
    # unique choice between them from a historical source explanation.
    opts = {normalize(v) for v in row["options"].values()}
    for left, right in [("脓毒性休克", "感染性休克"), ("低血钾", "低钾血症")]:
        if left in opts and right in opts:
            return "synonymous_options"
    return None


class NearStemIndex:
    """Conservative character-trigram comparison, independent of model scores."""
    def __init__(self, stems):
        self.features = []
        self.index = defaultdict(list)
        for stem in sorted(set(stems)):
            if len(stem) < 12:
                continue
            fs = {stem[i:i+3] for i in range(len(stem)-2)}
            idx = len(self.features)
            self.features.append(fs)
            for sh in fs:
                self.index[sh].append(idx)

    def matches(self, stem, threshold=0.90):
        if len(stem) < 12:
            return False
        fs = {stem[i:i+3] for i in range(len(stem)-2)}
        counts = Counter(idx for sh in fs for idx in self.index.get(sh, ()))
        return any(n/(len(fs)+len(self.features[idx])-n) >= threshold for idx,n in counts.items())


def load_source(filename):
    path = project_path("data/raw/cmexam/" + filename)
    if not path.is_file() or file_hash(path) != FILES[filename]:
        raise ValueError("Missing or changed official source: " + str(path))
    records, rejected = [], Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, raw in enumerate(csv.DictReader(stream), 2):
            try:
                opts = options_of(raw["Options"])
                gold = raw["Answer"].strip()
                if gold not in opts or not raw["Question"].strip():
                    raise ValueError("Missing stem or single answer")
            except ValueError:
                rejected["schema"] += 1
                continue
            records.append({"id": f"cmexam-{filename.split('.')[0]}-{line}",
                            "question": raw["Question"].strip(), "options": opts,
                            "answer": gold, "explanation": raw.get("Explanation", "").strip(),
                            "source_file": filename, "source_line": line,
                            "stem_group": group_key(raw["Question"])})
    return records, dict(rejected)


def as_training(row, system):
    return {"id": row["id"], "case_id": "cmexam-stem-" + hashlib.sha256(row["stem_group"].encode()).hexdigest()[:24],
            "task": "medical_exam", "source": "williamliujl/CMExam", "source_revision": REVISION,
            "source_file": row["source_file"], "source_line": row["source_line"],
            "source_sha256": FILES[row["source_file"]], "source_license": "Apache-2.0; README research-use restrictions retained",
            "origin": "public_exam", "review_status": "unreviewed", "gold_answer": row["answer"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt_of(row)},
                         {"role": "assistant", "content": "答案：" + row["answer"] + "。\n解析：" + row["explanation"]}]}


def unique_stems(records, forbidden, salt):
    selected, seen = [], set(forbidden)
    for row in sorted(records, key=lambda r: rank(r, salt)):
        if row["stem_group"] not in seen:
            selected.append(row)
            seen.add(row["stem_group"])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict-near-dedup', action='store_true')
    args = parser.parse_args()
    config = copy.deepcopy(load_config("configs/v3_4_natural_grouped.yaml"))
    tokenizer = load_tokenizer(config)
    train, train_reject = load_source("train.csv")
    val, val_reject = load_source("val.csv")
    test, test_reject = load_source("test_with_annotations.csv")
    # Test selection never uses predictions or explanations. Keep official test
    # questions out of *all* training and validation, including numeric variants.
    test_stems = {r["stem_group"] for r in test}
    all_heldout_stems = set()
    for filename in ["val.csv", "test_with_annotations.csv"]:
        with project_path("data/raw/cmexam/" + filename).open(encoding="utf-8-sig", newline="") as stream:
            all_heldout_stems.update(group_key(r["Question"]) for r in csv.DictReader(stream)
                                     if r.get("Question", "").strip())
    near_index = NearStemIndex(all_heldout_stems) if args.strict_near_dedup else None
    frozen_test = unique_stems(test, set(), "test-20261106")[:1000]
    frozen_dev = unique_stems(val, test_stems, "dev-20261106")[:200]
    for row in frozen_test + frozen_dev:
        row["messages"] = [{"role": "user", "content": prompt_of(row)}]
    accepted, reject = [], Counter()
    for row in unique_stems(train, all_heldout_stems, "train-20261106"):
        if near_index is not None and near_index.matches(row['stem_group']):
            reject['near_official_heldout_stem'] += 1
            continue
        defect = known_source_defect(row)
        if defect:
            reject[defect] += 1
            continue
        exp = row["explanation"]
        if not 35 <= len(exp) <= 450 or len(prompt_of(row)) > 550:
            reject["length"] += 1
            continue
        if any(s in exp for s in ["http://", "https://", "暂无解析", "解析略", "<img", "见图"]):
            reject["incomplete_or_external"] += 1
            continue
        # Accept explanations that explicitly identify the source's correct
        # option, including negative questions ("X错，为本题正确答案").
        gold = row["answer"]
        if not (re.search(r"[（(]" + gold + r"对", exp) or
                re.search(r"[（(]" + gold + r"错[，,]为本题正确答案", exp)):
            reject["no_explicit_label_support"] += 1
            continue
        record = as_training(row, config["system_prompt"])
        try:
            encode_conversation(tokenizer, record["messages"], 1024)
        except SequenceTooLong:
            reject["token_length"] += 1
            continue
        accepted.append(record)
        if len(accepted) == 2000:
            break
    assert len(accepted) == 2000 and len(frozen_dev) == 200 and len(frozen_test) == 1000
    # Fixed validation: 100 official dev questions (hash-selected), plus the
    # original grouped natural validation. Remaining 100 are generation dev,
    # never seen by SFT or checkpoint loss.
    val_training = [as_training(r, config["system_prompt"]) for r in frozen_dev[:100]]
    for row in val_training:
        encode_conversation(tokenizer, row["messages"], 2048)
    natural = read_jsonl(project_path("data/generated/v3-4-natural-grouped/corpus.jsonl"))
    natural_val = read_jsonl(project_path("data/processed/v3-4-natural-grouped/validation.jsonl"))
    for r in natural + natural_val:
        r.pop("split_group", None)
    fixed_natural_ids = {r["id"] for r in natural_val}
    corpus = accepted + [r for r in natural if r["id"] not in fixed_natural_ids]
    fixed_val = val_training + natural_val
    variant = "v4-1-cmexam" if args.strict_near_dedup else "v4-cmexam"
    out = project_path("data/generated/" + variant)
    safe_publish(out / "corpus.jsonl", corpus)
    safe_publish(out / "fixed_validation.jsonl", fixed_val)
    safe_publish(project_path("assets/cmexam_dev_frozen.jsonl"), frozen_dev[100:])
    safe_publish(project_path("assets/cmexam_test_frozen.jsonl"), frozen_test)
    config["project"]["name"] = "medft-" + variant
    config["data"].update(source="data/generated/" + variant + "/corpus.jsonl",
                          fixed_validation_source="data/generated/" + variant + "/fixed_validation.jsonl",
                          output_dir="data/processed/" + variant)
    config["data"]["evaluation_exclusions"] += ["assets/cmexam_dev_frozen.jsonl", "assets/cmexam_test_frozen.jsonl"]
    config["data"]["target_mix"]["medical_exam"] = 0.90
    for task in list(config["data"]["target_mix"]):
        if task != "medical_exam":
            config["data"]["target_mix"][task] *= 0.10
    config["training"].update(output_dir="outputs/" + variant, max_length=2048, epochs=1,
                              learning_rate=5e-5, lora_r=16, lora_alpha=32,
                              batch_size=2, gradient_accumulation_steps=4,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                              eval_steps=92, early_stopping_patience=3)
    cfgpath = project_path("configs/" + ("v4_1_cmexam.yaml" if args.strict_near_dedup else "v4_cmexam.yaml"))
    if cfgpath.exists():
        assert load_config(cfgpath) == config
    else:
        cfgpath.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    stats = build_data(config)
    train_keys = {group_key(r["messages"][1]["content"].split("\n")[1][3:]) for r in accepted}
    assert not train_keys & all_heldout_stems
    protocol = {"created_at": utc_now(), "revision": REVISION, "source_files_sha256": FILES,
                "near_stem_filter": {"enabled": args.strict_near_dedup,
                                     "method": "numeric-normalized character-trigram Jaccard >=0.90, stem length >=12"},
                "raw_counts": {"train": len(train), "val": len(val), "test": len(test)},
                "schema_rejections": [train_reject, val_reject, test_reject], "train_screening_rejections": dict(reject),
                "public_training_count": len(accepted), "natural_count": len(natural),
                "official_stem_groups_excluded_from_train": len(all_heldout_stems),
                "dev_count": 100, "test_count": 1000,
                "dev_sha256": file_hash(project_path("assets/cmexam_dev_frozen.jsonl")),
                "test_sha256": file_hash(project_path("assets/cmexam_test_frozen.jsonl")),
                "source_sha256": file_hash(out / "corpus.jsonl"),
                "public_train_labels": dict(Counter(r["gold_answer"] for r in accepted)),
                "evaluation_protocol": {"primary": "argmax_next_token_among_A_to_E_after_answer_prefix",
                                        "assistant_prefix": "答案：", "batch_size": 8,
                                        "shuffle_options_secondary": "fixed_seed_on_options_without_letter_or_position_references",
                                        "option_permutation_seed": "options-20261106:",
                                        "test_policy": "One final base/selected-adapter comparison after dev model selection. Never select a checkpoint from test accuracy."},
                "limitations": ["Official source answers are unreviewed by this project.",
                                "Public test contamination in base pretraining is unknown.",
                                "Question-stem and numeric-template overlap excluded; semantic overlap remains possible.",
                                "Exam accuracy measures knowledge choices, not diagnosis, clinical safety or explanation correctness.",
                                "Repository Apache-2.0 and README research-only restrictions are both retained."],
                "data_stats": stats}
    write_json(out / "preparation.json", protocol)
    print(json.dumps({k:v for k,v in protocol.items() if k != "data_stats"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
