"""Generate complete base-model answers for retention, not clinical gold labels."""
from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.config import load_config, project_path, file_hash, utc_now, write_json
from medft.data import read_jsonl, write_jsonl, normalize
from medft.models import load_quantized_model, load_tokenizer


def main():
    import torch
    from transformers import set_seed
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    config=load_config('configs/v3.yaml')
    questions=[r for r in read_jsonl(project_path('assets/v2_1_seeds.jsonl'))
               if r['task']=='knowledge_qa' and r['id'].startswith('v21-k')]
    blocked={normalize(m['content']) for p in ['assets/evaluation_v1.jsonl','assets/evaluation_v2.jsonl','assets/evaluation_complete_frozen.jsonl']
             for r in read_jsonl(project_path(p)) for m in r['messages'] if m['role']=='user'}
    questions=[r for r in questions if normalize(r['messages'][0]['content']) not in blocked]
    destination=project_path('data/generated/v3-1-complete/base_replay_candidates.jsonl')
    existing=read_jsonl(destination) if destination.exists() else []
    previous_sha256=file_hash(destination) if existing else None
    done={r['teacher_seed_id'] for r in existing}
    questions=[r for r in questions if r['id'] not in done]
    set_seed(42)
    tokenizer=load_tokenizer(config); tokenizer.padding_side='left'
    model, manifest=load_quantized_model(config); model.eval(); model.config.use_cache=True
    results=list(existing)
    started=utc_now(); clock=time.monotonic()
    for index in range(0,len(questions),args.batch_size):
        batch=questions[index:index+args.batch_size]
        prompts=[tokenizer.apply_chat_template([{'role':'system','content':config['system_prompt']},r['messages'][0]],
                 tokenize=False,add_generation_prompt=True) for r in batch]
        inputs=tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=384,do_sample=False,temperature=None,top_p=None,top_k=None,
                                  pad_token_id=tokenizer.pad_token_id)
        generated=output[:,inputs['input_ids'].shape[1]:]
        for row,sequence in zip(batch,generated):
            answer=tokenizer.decode(sequence,skip_special_tokens=True).strip()
            token_ids=sequence.tolist(); eos=model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else eos
            finished=any(token in token_ids for token in (eos or []))
            results.append({'id':'complete-replay-'+row['id'],'case_id':row['case_id'],'task':'knowledge_qa',
                            'source':'local_base_self_distillation','origin':'base_replay','review_status':'unreviewed',
                            'curriculum_version':'complete-answer-v1','teacher_seed_id':row['id'],
                            'teacher_model':manifest,'teacher_generation':{'max_new_tokens':384,'do_sample':False,'batch_size':args.batch_size},
                            'teacher_finished':finished,'messages':[row['messages'][0],{'role':'assistant','content':answer}]})
        write_jsonl(destination,results)
        print(f'Base replay {len(results)}/180; elapsed {time.monotonic()-clock:.1f}s',flush=True)
    write_json(project_path('outputs/optimization/base-replay-generation.json'),{'started_at':started,'completed_at':utc_now(),
               'model':manifest,'count':len(results),'source_questions_sha256':file_hash(project_path('assets/v2_1_seeds.jsonl')),
               'candidates_sha256':file_hash(destination),'clinical_review':False,'batch_size':args.batch_size,
               'resumed_count':len(existing),'resumed_candidates_sha256':previous_sha256})


if __name__=='__main__':main()
