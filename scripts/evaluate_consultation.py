"""Evaluate raw V5 consultation generations; never applies a safety rewrite.

Original36 uses the unchanged single-item evaluator/256-token limit. New32 uses
identical explicitly recorded batches for both models. Heuristics are review
assistance, not medical correctness labels.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.config import load_config,project_path,file_hash,utc_now,write_json
from medft.data import read_jsonl,write_jsonl,normalize
from medft.evaluation import evaluate,score_response
from medft.inference import Generator


def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/v5_consultation.yaml')
    parser.add_argument('--suite',choices=['original','holdout','retention'],required=True)
    which=parser.add_mutually_exclusive_group(required=True)
    which.add_argument('--base',action='store_true');which.add_argument('--adapter')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--batch-original',action='store_true',help='Matched fresh base/adapter batches for original36; report separately from old single-item results.')
    args=parser.parse_args()
    config=load_config(args.config)
    if args.suite=='original' and not args.batch_original:
        config['generation']['max_new_tokens']=256
        print(json.dumps(evaluate(config,adapter=args.adapter,output_dir=args.output_dir),ensure_ascii=False,indent=2))
        return
    if args.suite=='original':config['generation']['max_new_tokens']=256
    source=ROOT/({'original':'assets/evaluation_v2.jsonl','holdout':'assets/consultation_v5_holdout_frozen.jsonl','retention':'assets/evaluation_retention_probes.jsonl'}[args.suite])
    protocol=json.loads((ROOT/'outputs/consultation-v5/protocol.json').read_text(encoding='utf-8'))
    if args.suite=='holdout' and file_hash(source)!=protocol['hashes'][str(source.relative_to(ROOT))]:
        raise ValueError('Frozen holdout changed')
    dest=project_path(args.output_dir)
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    if args.batch_size<1:raise ValueError('Positive batch size required')
    cases=read_jsonl(source)
    training=read_jsonl(project_path(config['data']['source']))
    seen={normalize(m['content']) for r in training for m in r['messages'] if m['role']=='user'}
    if any(normalize(m['content']) in seen for c in cases for m in c['messages'] if m['role']=='user'):
        raise ValueError('Exact train/evaluation prompt overlap')
    generator=Generator(config,args.adapter)
    generator.tokenizer.padding_side='left'
    results=[];dest.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(cases),args.batch_size):
        batch=cases[start:start+args.batch_size]
        prompts=[generator.tokenizer.apply_chat_template([{'role':'system','content':config['system_prompt']}]+c['messages'],tokenize=False,add_generation_prompt=True) for c in batch]
        inputs=generator.tokenizer(prompts,padding=True,add_special_tokens=False,return_tensors='pt').to(generator.model.device)
        if inputs['input_ids'].shape[1]>config['generation']['max_input_tokens']:raise ValueError('Input too long')
        with torch.inference_mode():
            output=generator.model.generate(**inputs,max_new_tokens=config['generation']['max_new_tokens'],do_sample=False,
                temperature=None,top_p=None,top_k=None,pad_token_id=generator.tokenizer.pad_token_id)
        answers=generator.tokenizer.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        for c,answer in zip(batch,answers):
            results.append({**c,'raw_output':answer.strip(),'served_output':answer.strip(),
                'model_adapter':args.adapter,'manual_review_required':True})
            if args.suite=='original':results[-1]['scores']=score_response(c,answer.strip())
        write_jsonl(dest/'predictions.jsonl',results)
        print(f'{args.suite} {len(results)}/{len(cases)}',flush=True)
    summary=dict(completed_at=utc_now(),model=generator.manifest,adapter=args.adapter,
        mode='raw',suite=args.suite,case_count=len(results),evaluation_sha256=file_hash(source),
        system_prompt=config['system_prompt'],generation={**config['generation'],'batch_size':args.batch_size,'padding_side':'left'},
        seed=config['project']['seed'],empty_outputs=sum(not r['raw_output'] for r in results),
        predictions_sha256=file_hash(dest/'predictions.jsonl'),
        evaluation_code_sha256=file_hash(Path(__file__)),
        interpretation='Source-group authored holdout, not clinical validation. Evaluate current state, action urgency and unsafe additions by reading every raw response. Related condition branches are not independent patients.')
    if args.suite=='original':
        summary.update(required_groups_case_count=len(results),
            required_groups_pass_rate=sum(r['scores']['required_groups_pass'] for r in results)/len(results),
            keyword_match_rate=sum(r['scores']['keyword_match'] for r in results)/len(results),
            forbidden_pattern_hits=sum(r['scores']['forbidden_pattern_hit'] for r in results),
            required_groups_pass_by_task={task:dict(count=len(sub),passed=sum(r['scores']['required_groups_pass'] for r in sub))
                for task in sorted({r['task'] for r in results}) if (sub:=[r for r in results if r['task']==task])},
            interpretation='Original36 development regression, unchanged scoring; fresh matched batching for base and adapter. Do not mix with prior single-item metrics. Not clinical accuracy.')
    write_json(dest/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
