"""Frozen next-option scoring; identical prompts and batches for both models.

Answers and explanations never enter the input. This constrained accuracy is
not the correctness of a freely generated explanation or a clinical benchmark.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import copy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medft.config import load_config, project_path, file_hash, write_json, utc_now
from medft.data import read_jsonl, write_jsonl, normalize
from medft.inference import Generator
from scripts.prepare_cmexam import prompt_of, group_key


def wilson(correct, total):
    z = 1.959963984540054
    p = correct / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    margin = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [center-margin, center+margin]


def can_permute(row):
    # Conservative exclusion: "all above", choice-letter references, or other
    # positional language can change meaning when options are reordered.
    return not any(re.search(r"[A-EＡ-Ｅ]|以上|下列|上述|前述|均|都|这些|其余|第[一二三四五]", v)
                   for v in row["options"].values())


def permute_options(row, seed):
    if not can_permute(row):
        raise ValueError("Option text can depend on its position")
    row = copy.deepcopy(row)
    order = list("ABCDE")
    random.Random(seed + row["id"]).shuffle(order)
    original = row["options"]
    original_gold = row["answer"]
    row["options"] = {new: original[old] for new,old in zip("ABCDE",order)}
    row["answer"] = "ABCDE"[order.index(original_gold)]
    return row


def main():
    import torch
    parser = argparse.ArgumentParser()
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--base", action="store_true")
    selector.add_argument("--adapter")
    parser.add_argument("--config", default="configs/v4_cmexam.yaml")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare-base")
    parser.add_argument("--permuted", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    dest = project_path(args.output_dir)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(dest)
    prep_path = project_path(config['data']['source']).parent / 'preparation.json'
    preparation = json.loads(prep_path.read_text(encoding="utf-8"))
    source = project_path(f"assets/cmexam_{args.split}_frozen.jsonl")
    assert file_hash(source) == preparation[args.split + "_sha256"]
    assert file_hash(project_path(config["data"]["source"])) == preparation["source_sha256"]
    cases = read_jsonl(source)
    if args.permuted:
        seed = preparation["evaluation_protocol"]["option_permutation_seed"]
        cases = [permute_options(r, seed) for r in cases if can_permute(r)]
        if not cases:
            raise ValueError("No position-independent options")
    train = read_jsonl(project_path(config['data']['output_dir']) / 'train.jsonl')
    train_stems = {group_key(r["messages"][1]["content"].split("\n")[1][3:]) for r in train if r["task"] == "medical_exam"}
    assert not train_stems & {r["stem_group"] for r in cases}
    generator = Generator(config, args.adapter)
    tokenizer = generator.tokenizer
    tokenizer.padding_side = "left"
    labels = list("ABCDE")
    tokens = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    assert all(len(ids) == 1 for ids in tokens)
    option_ids = [ids[0] for ids in tokens]
    batch_size = preparation["evaluation_protocol"]["batch_size"]
    predictions = []
    for offset in range(0, len(cases), batch_size):
        batch = cases[offset:offset+batch_size]
        prompts = [tokenizer.apply_chat_template([
            {"role":"system", "content":config["system_prompt"]},
            {"role":"user", "content":prompt_of(r)}], tokenize=False, add_generation_prompt=True) + "答案：" for r in batch]
        # Check the answer-prefix boundary really uses the same next option token.
        for prompt in prompts:
            prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
            for label, ids in zip(labels, tokens):
                assert tokenizer.encode(prompt+label, add_special_tokens=False) == prefix_ids+ids
        inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(generator.model.device)
        assert inputs["input_ids"].shape[1] <= config["generation"]["max_input_tokens"]
        with torch.inference_mode():
            logits = generator.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, option_ids].float().cpu()
        assert torch.isfinite(logits).all()
        chosen = logits.argmax(dim=-1).tolist()
        for row, scores, index in zip(batch, logits.tolist(), chosen):
            prediction = labels[index]
            predictions.append({"id":row["id"], "gold":row["answer"], "prediction":prediction,
                                "correct":prediction == row["answer"],
                                "option_logits":dict(zip(labels, scores))})
        if len(predictions) % 80 == 0 or len(predictions) == len(cases):
            print(f"CMExam {args.split}: {len(predictions)}/{len(cases)}", flush=True)
    correct = sum(r["correct"] for r in predictions)
    dest.mkdir(parents=True, exist_ok=True)
    write_jsonl(dest/"predictions.jsonl", predictions)
    summary = {"completed_at":utc_now(), "model":generator.manifest,
               "adapter":str(project_path(args.adapter)) if args.adapter else None,
               "evaluation_sha256":file_hash(source), "source_sha256":preparation["source_sha256"],
               "protocol":preparation["evaluation_protocol"], "system_prompt":config["system_prompt"],
               "seed":config["project"]["seed"], "split":args.split,
               "view":"fixed_permuted_eligible_subset" if args.permuted else "original",
               "case_count":len(cases), "correct":correct, "accuracy":correct/len(cases),
               "wilson95":wilson(correct,len(cases)),
               "gold_distribution":dict(Counter(r["gold"] for r in predictions)),
               "prediction_distribution":dict(Counter(r["prediction"] for r in predictions)),
               "interpretation":"Constrained medical exam choice accuracy; not free-generation explanation accuracy, clinical consultation quality or safety. Public pretraining contamination is unknown."}
    if args.compare_base:
        base_dir = project_path(args.compare_base)
        base_summary = json.loads((base_dir/"summary.json").read_text(encoding="utf-8"))
        for key in ["model", "evaluation_sha256", "source_sha256", "protocol", "system_prompt", "seed", "split", "view"]:
            assert summary[key] == base_summary[key], key
        base_preds = read_jsonl(base_dir/"predictions.jsonl")
        assert [(r["id"],r["gold"]) for r in predictions] == [(r["id"],r["gold"]) for r in base_preds]
        gained = sum(a["correct"] and not b["correct"] for b,a in zip(base_preds,predictions))
        lost = sum(b["correct"] and not a["correct"] for b,a in zip(base_preds,predictions))
        from scripts.paired_statistics import exact_paired_p
        discordant = gained+lost
        summary["paired_comparison"] = {
            "base_correct":base_summary["correct"], "adapter_correct":correct,
            "delta_percentage_points":100*(correct-base_summary["correct"])/len(cases),
            "gained":gained, "lost":lost,
            "mcnemar_exact_two_sided_p":exact_paired_p(gained, lost),
            "note":"One paired comparison. Dev model selection and public pretraining overlap limit interpretation."}
    write_json(dest/"summary.json", summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2), flush=True)


if __name__ == "__main__":
    main()
