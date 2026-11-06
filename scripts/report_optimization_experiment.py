"""Verify an experiment against preserved artifacts and make paired reviews."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medft.config import load_config, project_path, file_hash, write_json, utc_now
from medft.data import read_jsonl, normalize
from medft.training import load_training_data
from medft.evaluation import compare_reports
from scripts.verify_complete_results import check_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--original-dir', required=True)
    parser.add_argument('--extension-dir')
    parser.add_argument('--base-dir', default='outputs/evaluations/v2-base')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    dest = project_path(args.output_dir)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(dest)
    start = json.loads(project_path('outputs/optimization/optimization-start.json').read_text(encoding='utf-8'))
    changed = [p for p, h in start['protected_files'].items()
               if not project_path(p).is_file() or file_hash(project_path(p)) != h]
    assert not changed, changed
    config = load_config(args.config)
    splits, stats = load_training_data(config)
    run = json.loads((project_path(config['training']['output_dir']) / 'run.json').read_text(encoding='utf-8'))
    assert run.get('completed_at') and run['config'] == config and run['data'] == stats
    weights = project_path(run['adapter_dir']) / 'adapter_model.safetensors'
    best_weights = Path(run['selection']['best_model_checkpoint']) / 'adapter_model.safetensors'
    assert file_hash(weights) == file_hash(best_weights)
    from safetensors import safe_open
    with safe_open(str(weights), framework='pt', device='cpu') as stream:
        keys = list(stream.keys())
        parameters = sum(math.prod(stream.get_slice(k).get_shape()) for k in keys)
    assert parameters == run['trainable_parameters']
    train = splits['train']
    val = splits['validation']
    collisions = {}
    for key in ['case_id', 'split_group']:
        collisions[key] = sorted({r[key] for r in train} & {r[key] for r in val})
    for role in ['user', 'assistant']:
        # Final answers are grouped; common earlier assistant questions can recur.
        def values(rows):
            if role == 'assistant':
                return {normalize(r['messages'][-1]['content']) for r in rows}
            return {normalize(m['content']) for r in rows for m in r['messages'] if m['role'] == role}
        collisions[role] = sorted(values(train) & values(val))
    assert not any(collisions.values()), collisions
    evaluations = {}
    comparisons = {}
    paired_changes = {}
    pairs = [('original', args.base_dir, args.original_dir, 'assets/evaluation_v2.jsonl')]
    if args.extension_dir:
        pairs.append(('extension', 'outputs/evaluations/complete-frozen-base', args.extension_dir, 'assets/evaluation_complete_frozen.jsonl'))
    for label, base_dir, tuned_dir, source in pairs:
        baseline, base_audit = check_predictions(project_path(base_dir), project_path(source))
        tuned, tuned_audit = check_predictions(project_path(tuned_dir), project_path(source))
        evaluations[label] = {'base': base_audit, 'adapter': tuned_audit}
        comparisons[label] = compare_reports(base_dir, tuned_dir, str(dest / f'{label}-comparison.json'))
        gains, losses = [], []
        markdown = ['# 原始回答逐题对照', '',
                    '开发回归；保留原题和规则。表中的通过只表示关键词要点命中，内容仍需阅读。', '']
        for b, a in zip(baseline, tuned):
            bp, ap = b['scores']['required_groups_pass'], a['scores']['required_groups_pass']
            if ap and not bp: gains.append(a['id'])
            if bp and not ap: losses.append(a['id'])
            markdown += [f"## {a['id']} / {a['task']}", '',
                         '\n\n'.join(m['role'] + ': ' + m['content'] for m in a['messages']), '',
                         f'规则：基模 {bp} / adapter {ap}', '', '### 基模', '', b['raw_output'], '',
                         '### adapter', '', a['raw_output'], '']
        paired_changes[label] = {'gained': gains, 'lost': losses}
        (dest / f'{label}-paired-review.md').write_text('\n'.join(markdown), encoding='utf-8')
    result = {'verified_at': utc_now(), 'config': args.config,
              'protected_files_checked': len(start['protected_files']), 'changed_historical_files': changed,
              'data_sha256': stats['source_sha256'], 'split_counts': {k: len(v) for k, v in splits.items()},
              'exact_split_collisions': collisions, 'selection': run['selection'],
              'adapter_sha256': file_hash(weights), 'adapter_tensors': len(keys), 'adapter_parameters': parameters,
              'evaluations': evaluations, 'comparisons': comparisons, 'paired_changes': paired_changes,
              'limitation': 'Both are development evaluations. Curriculum and extension have the same author; neither is a blind clinical test. Semantic/template overlap remains possible.'}
    write_json(dest / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
