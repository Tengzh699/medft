"""Same raw scoring on the frozen extension; both models use identical batches.

The historical 36-case evaluation remains on the original single-item CLI.
This harness records batching explicitly and never changes a case or its checks.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.config import load_config,project_path,file_hash,write_json,utc_now
from medft.data import read_jsonl,write_jsonl,validate_messages,normalize
from medft.evaluation import score_response
from medft.inference import Generator


def main():
    import torch
    parser=argparse.ArgumentParser()
    selector=parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--base',action='store_true')
    selector.add_argument('--adapter')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--config',default='configs/v3_1_complete_grouped_eval.yaml')
    args=parser.parse_args()
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    config=load_config(args.config)
    assert config['generation']['do_sample'] is False
    source=project_path(config['data']['evaluation_source'])
    if file_hash(source)!='3eeb2114f5028b0f15501d33154aa912a5d1c2a23672629dd94e9a54c1c7bf76':
        raise ValueError('Frozen extension changed.')
    dest=project_path(args.output_dir)
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    cases=read_jsonl(source)
    assert len(cases)==29 and len({c['id'] for c in cases})==29
    training=read_jsonl(project_path(config['data']['source']))
    questions={normalize(m['content']) for r in training for m in r['messages'] if m['role']=='user'}
    for case in cases:
        validate_messages(case['messages'],needs_answer=False)
        score_response(case,'')
        assert not any(normalize(m['content']) in questions for m in case['messages'] if m['role']=='user')
    generator=Generator(config,args.adapter)
    generator.tokenizer.padding_side='left'
    results=[]
    for start in range(0,len(cases),args.batch_size):
        batch=cases[start:start+args.batch_size]
        bodies=[c['messages'][1:] if c['messages'][0]['role']=='system' else c['messages'] for c in batch]
        prompts=[generator.tokenizer.apply_chat_template([{'role':'system','content':config['system_prompt']}]+body,
                 tokenize=False,add_generation_prompt=True) for body in bodies]
        inputs=generator.tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to(generator.model.device)
        if inputs['input_ids'].shape[1]>config['generation']['max_input_tokens']:raise ValueError('Input length exceeded.')
        with torch.inference_mode():
            generated=generator.model.generate(**inputs,max_new_tokens=config['generation']['max_new_tokens'],
                       do_sample=False,temperature=None,top_p=None,top_k=None,pad_token_id=generator.tokenizer.pad_token_id)
        answers=generator.tokenizer.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        for case,body,answer in zip(batch,bodies,answers):
            answer=answer.strip()
            results.append({'id':case['id'],'task':case['task'],'risk':case['risk'],'messages':body,'notes':case.get('notes',''),
                            'raw_output':answer,'served_output':answer,'scores':score_response(case,answer),'safety':None,'output_issues':[]})
        print(f'Frozen extension {len(results)}/{len(cases)}',flush=True)
    dest.mkdir(parents=True,exist_ok=True)
    write_jsonl(dest/'predictions.jsonl',results)
    summary={'completed_at':utc_now(),'model':generator.manifest,'adapter':str(project_path(args.adapter)) if args.adapter else None,
             'mode':'raw','evaluation_sha256':file_hash(source),'system_prompt':config['system_prompt'],
             'generation':{**config['generation'],'inference_batch_size':args.batch_size,'padding_side':'left'},
             'seed':config['project']['seed'],'case_count':len(results),
             'keyword_match_rate':sum(r['scores']['keyword_match'] for r in results)/len(results),
             'forbidden_pattern_hits':sum(r['scores']['forbidden_pattern_hit'] for r in results),
             'empty_outputs':sum(r['scores']['empty'] for r in results),
             'required_groups_case_count':len(results),
             'required_groups_pass_rate':sum(r['scores']['required_groups_pass'] for r in results)/len(results),
             'required_groups_pass_by_task':{
                 task:{'count':len(sub),'passed':sum(r['scores']['required_groups_pass'] for r in sub)}
                 for task in sorted({r['task'] for r in results}) if (sub:=[r for r in results if r['task']==task])},
             'interpretation':'Pre-training frozen development extension authored with this curriculum. Same batched raw generation for base and adapter; unchanged keyword checks are not clinical accuracy.'}
    write_json(dest/'summary.json',summary)
    with (dest/'manual_review.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['id','task','risk','conversation','raw_output','medical_correctness','helpfulness','safe','reviewer_notes']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in results:
            writer.writerow({k:r[k] for k in ['id','task','risk','raw_output']}|{'conversation':'\n'.join(m['role']+': '+m['content'] for m in r['messages'])})
    print(summary)


if __name__=='__main__':main()
