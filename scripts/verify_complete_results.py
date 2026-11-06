"""File-only audit of the completed optimization, without changing scores."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.config import file_hash,project_path,write_json,utc_now,load_config
from medft.data import read_jsonl
from medft.evaluation import score_response,compare_reports
from medft.training import load_training_data


def check_predictions(directory, source):
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    cases=read_jsonl(source)
    rows=read_jsonl(directory/'predictions.jsonl')
    assert len(rows)==len(cases)==summary['case_count']
    assert [r['id'] for r in rows]==[c['id'] for c in cases]
    assert summary['evaluation_sha256']==file_hash(source)
    for case,row in zip(cases,rows):
        body=case['messages'][1:] if case['messages'][0]['role']=='system' else case['messages']
        assert row['messages']==body
        assert row['raw_output']==row['served_output']
        assert row['scores']==score_response(case,row['raw_output'])
    kw=[r['scores']['keyword_match'] for r in rows if r['scores']['keyword_match'] is not None]
    groups=[r['scores']['required_groups_pass'] for r in rows if 'required_groups_pass' in r['scores']]
    assert summary['keyword_match_rate']==sum(kw)/len(kw)
    assert summary['required_groups_pass_rate']==sum(groups)/len(groups)
    assert summary['forbidden_pattern_hits']==sum(r['scores']['forbidden_pattern_hit'] for r in rows)
    assert summary['empty_outputs']==sum(r['scores']['empty'] for r in rows)
    assert summary['required_groups_pass_by_task']=={
        task:{'count':len(sub),'passed':sum(r['scores']['required_groups_pass'] for r in sub)}
        for task in sorted({r['task'] for r in rows})
        if (sub:=[r for r in rows if r['task']==task])}
    return rows,{'case_count':len(rows),'keyword_passes':sum(kw),'required_group_passes':sum(groups),
                 'mean_answer_chars':round(sum(len(r['raw_output']) for r in rows)/len(rows),1),
                 'predictions_sha256':file_hash(directory/'predictions.jsonl')}


def main():
    start=json.loads(project_path('outputs/optimization/optimization-start.json').read_text(encoding='utf-8'))
    changed=[p for p,h in start['protected_files'].items() if not project_path(p).is_file() or file_hash(project_path(p))!=h]
    assert not changed,changed
    freeze=json.loads(project_path('outputs/optimization/complete-evaluation-freeze.json').read_text(encoding='utf-8'))
    assert file_hash(project_path('assets/evaluation_complete_frozen.jsonl'))==freeze['sha256']
    config=load_config('configs/v3_1_complete_grouped.yaml')
    splits,stats=load_training_data(config)
    run=json.loads(project_path('outputs/v3-1-complete-grouped/run.json').read_text(encoding='utf-8'))
    assert run.get('completed_at') and run['data']==stats and run['config']==config
    weights=project_path('outputs/v3-1-complete-grouped/adapter/adapter_model.safetensors')
    best=Path(run['selection']['best_model_checkpoint'])/'adapter_model.safetensors'
    assert file_hash(weights)==file_hash(best)
    from safetensors import safe_open
    with safe_open(str(weights),framework='pt',device='cpu') as stream:
        keys=list(stream.keys())
        parameters=sum(math.prod(stream.get_slice(k).get_shape()) for k in keys)
    assert parameters==run['trainable_parameters']
    evaluations={}
    for name,folder,source in [
        ('base_original','v2-base','evaluation_v2.jsonl'),
        ('complete_original','v3-1-complete-grouped','evaluation_v2.jsonl'),
        ('base_extension','complete-frozen-base','evaluation_complete_frozen.jsonl'),
        ('complete_extension','complete-frozen-adapter','evaluation_complete_frozen.jsonl')]:
        rows,audit=check_predictions(project_path('outputs/evaluations')/folder,project_path('assets')/source)
        evaluations[name]=audit
    comparisons={}
    for label,b,a in [('original','v2-base','v3-1-complete-grouped'),('extension','complete-frozen-base','complete-frozen-adapter')]:
        comparisons[label]=compare_reports(f'outputs/evaluations/{b}',f'outputs/evaluations/{a}',f'outputs/evaluations/v3-1-{label}-comparison.json')
    result={'verified_at':utc_now(),'protected_file_count':len(start['protected_files']),'changed_historical_files':changed,
            'training':run['selection'],'adapter_sha256':file_hash(weights),'adapter_tensors':len(keys),'adapter_parameters':parameters,
            'source_sha256':stats['source_sha256'],'split_counts':{k:len(v) for k,v in splits.items()},
            'evaluations':evaluations,'comparisons':comparisons,
            'interpretation':'Unchanged raw-output development regressions; rule coverage is not clinical accuracy. Additional cases were authored with the curriculum.'}
    write_json(project_path('outputs/optimization/final-verification.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
