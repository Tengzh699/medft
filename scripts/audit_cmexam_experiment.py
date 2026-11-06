"""File-only verification of source labels, selected weights and exam scores."""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medft.config import load_config, project_path, file_hash, utc_now, write_json
from medft.data import read_jsonl, normalize
from medft.training import load_training_data
from scripts.prepare_cmexam import FILES, options_of, prompt_of, group_key
from scripts.evaluate_cmexam import wilson, can_permute, permute_options


def check_scores(folder, config, preparation):
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    source = project_path(f"assets/cmexam_{summary['split']}_frozen.jsonl")
    assert summary['evaluation_sha256'] == file_hash(source) == preparation[summary['split']+'_sha256']
    assert summary['source_sha256'] == preparation['source_sha256']
    assert summary['system_prompt'] == config['system_prompt']
    assert summary['protocol'] == preparation['evaluation_protocol']
    cases = read_jsonl(source)
    if summary['view'] == 'fixed_permuted_eligible_subset':
        cases = [permute_options(r, summary['protocol']['option_permutation_seed']) for r in cases if can_permute(r)]
    else:
        assert summary['view'] == 'original'
    rows = read_jsonl(folder/'predictions.jsonl')
    assert len(cases) == len(rows) == summary['case_count']
    for case, row in zip(cases, rows):
        assert row['id'] == case['id'] and row['gold'] == case['answer']
        scores = row['option_logits']
        assert set(scores) == set('ABCDE') and all(math.isfinite(v) for v in scores.values())
        assert row['prediction'] == max('ABCDE', key=lambda k:scores[k])
        assert row['correct'] == (row['prediction'] == row['gold'])
    correct = sum(r['correct'] for r in rows)
    assert summary['correct'] == correct
    assert summary['accuracy'] == correct/len(rows)
    assert summary['wilson95'] == wilson(correct,len(rows))
    if 'paired_comparison' in summary:
        paired = summary['paired_comparison']
        assert paired['adapter_correct'] == correct
        assert paired['gained']-paired['lost'] == correct-paired['base_correct']
    return {'correct':correct, 'count':len(rows), 'predictions_sha256':file_hash(folder/'predictions.jsonl'),
            'paired_comparison':summary.get('paired_comparison')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/v4_1_cmexam.yaml')
    parser.add_argument('--evaluation-dir', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    splits, stats = load_training_data(config)
    preparation = json.loads((project_path(config['data']['source']).parent/'preparation.json').read_text(encoding='utf-8'))
    freeze = json.loads(project_path('outputs/optimization/cmexam-strict-pretraining-freeze.json').read_text(encoding='utf-8'))
    amendments = json.loads(project_path('outputs/optimization/cmexam-statistics-amendment.json').read_text(encoding='utf-8'))
    for p,h in freeze['files'].items():
        if p == 'scripts/evaluate_cmexam.py':
            assert amendments['original_evaluator_sha256'] == h
            assert file_hash(project_path(amendments['original_evaluator_copy'])) == h
            assert file_hash(project_path(p)) == amendments['updated_evaluator_sha256']
        else:
            assert file_hash(project_path(p)) == h, p
    assert file_hash(project_path('scripts/paired_statistics.py')) == amendments['statistics_helper_sha256']
    history = json.loads(project_path('outputs/optimization/optimization-start.json').read_text(encoding='utf-8'))
    assert all(project_path(p).is_file() and file_hash(project_path(p)) == h for p,h in history['protected_files'].items())
    sources = {}
    for name, sha in FILES.items():
        path = project_path('data/raw/cmexam/'+name)
        assert file_hash(path) == sha
        with path.open(encoding='utf-8-sig', newline='') as f:
            sources[name] = list(csv.DictReader(f))
    public_count = 0
    reconstructed_count = 0
    train_stems = set()
    heldout_stems = {group_key(r['Question']) for name in ['val.csv','test_with_annotations.csv'] for r in sources[name]}
    for split, rows in splits.items():
        for row in rows:
            if row['task'] != 'medical_exam':
                continue
            reconstructed_count += 1
            raw = sources[row['source_file']][row['source_line']-2]
            reconstructed = {'question':raw['Question'].strip(),'options':options_of(raw['Options'])}
            assert row['messages'][1]['content'] == prompt_of(reconstructed)
            assert row['gold_answer'] == raw['Answer'].strip()
            assert row['messages'][-1]['content'] == '答案：'+raw['Answer'].strip()+'。\n解析：'+raw['Explanation'].strip()
            if split == 'train':
                assert row['source_file'] == 'train.csv'
                public_count += 1
                train_stems.add(group_key(raw['Question']))
    assert reconstructed_count == 2100 and public_count == len(train_stems) == 2000 and not train_stems & heldout_stems
    overlaps = {}
    for role in ['user','assistant']:
        def texts(rows):
            return {normalize(m['content']) for r in rows for m in ([r['messages'][-1]] if role=='assistant' else r['messages']) if m['role']==role}
        overlaps[role] = len(texts(splits['train']) & texts(splits['validation']))
    assert not any(overlaps.values())
    near = json.loads(project_path('outputs/optimization/cmexam-strict-near-stem-audit.json').read_text(encoding='utf-8'))
    assert near['source_sha256'] == stats['source_sha256'] and not near['matches']
    run = json.loads((project_path(config['training']['output_dir'])/'run.json').read_text(encoding='utf-8'))
    assert run.get('completed_at') and run['config'] == config and run['data'] == stats
    adapter = Path(run['adapter_dir'])/'adapter_model.safetensors'
    best = Path(run['selection']['best_model_checkpoint'])/'adapter_model.safetensors'
    assert file_hash(adapter) == file_hash(best)
    evaluations = {p:check_scores(project_path(p),config,preparation) for p in args.evaluation_dir}
    summaries = {p:json.loads((project_path(p)/'summary.json').read_text(encoding='utf-8')) for p in args.evaluation_dir}
    original_config = json.loads((adapter.parent/'adapter_config.json').read_text(encoding='utf-8'))
    strengths = {}
    for folder, summary in summaries.items():
        if summary['adapter'] is None:
            strengths[folder] = 0.0
            continue
        used = Path(summary['adapter'])
        assert file_hash(used/'adapter_model.safetensors') == file_hash(adapter)
        actual_config = json.loads((used/'adapter_config.json').read_text(encoding='utf-8'))
        if used == adapter.parent:
            assert actual_config == original_config
            strengths[folder] = 1.0
        else:
            derived = json.loads((used.parent/'derived_adapter.json').read_text(encoding='utf-8'))
            assert derived['kind'] == 'lora_inference_strength_calibration'
            assert derived['source_adapter'] == str(adapter.parent)
            assert derived['source_run_sha256'] == file_hash(project_path(config['training']['output_dir'])/'run.json')
            assert derived['source_adapter_config_sha256'] == file_hash(adapter.parent/'adapter_config.json')
            assert derived['model'] == run['model'] and derived['adapter_sha256'] == file_hash(adapter)
            assert derived['scale'] in [.25,.5]
            assert actual_config['lora_alpha'] == original_config['lora_alpha']*derived['scale']
            expected_config = {**original_config,'lora_alpha':actual_config['lora_alpha']}
            assert actual_config == expected_config
            strengths[folder] = derived['scale']
    for folder, summary in summaries.items():
        if 'paired_comparison' not in summary:
            continue
        bases = [p for p,s in summaries.items() if s['adapter'] is None
                 and s['split']==summary['split'] and s['view']==summary['view']]
        assert len(bases)==1
        baseline = summaries[bases[0]]
        for key in ['model','evaluation_sha256','protocol','system_prompt','seed','view','split']:
            assert baseline[key]==summary[key]
        b = read_jsonl(project_path(bases[0])/'predictions.jsonl')
        a = read_jsonl(project_path(folder)/'predictions.jsonl')
        assert [(r['id'],r['gold']) for r in b] == [(r['id'],r['gold']) for r in a]
        gained = sum(y['correct'] and not x['correct'] for x,y in zip(b,a))
        lost = sum(x['correct'] and not y['correct'] for x,y in zip(b,a))
        paired = summary['paired_comparison']
        assert paired['gained']==gained and paired['lost']==lost
        assert paired['base_correct']==baseline['correct']
        assert math.isclose(paired['delta_percentage_points'],100*(gained-lost)/len(a),abs_tol=1e-10)
        from scripts.paired_statistics import exact_paired_p
        pval = exact_paired_p(gained,lost)
        assert paired['mcnemar_exact_two_sided_p']==pval
    write_json(project_path(args.output), {'verified_at':utc_now(), 'historical_files_preserved':len(history['protected_files']),
               'source_labels_reconstructed':reconstructed_count, 'split_counts':{k:len(v) for k,v in splits.items()},
               'exact_text_overlap':overlaps,'high_near_stem_matches':len(near['matches']),
               'selection':run['selection'], 'adapter_sha256':file_hash(adapter), 'evaluations':evaluations,
               'inference_strengths':strengths,
               'limits':'Automatic source/score audit, not physician review. Public pretraining contamination and lower semantic similarity remain possible.'})
    print('CMExam source, freeze, checkpoint and scores verified.', flush=True)


if __name__ == '__main__':
    main()
