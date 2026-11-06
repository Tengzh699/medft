"""Verify V5 lineage, split isolation, final-turn labels and selected weights.

This audits experiment integrity, not the medical correctness of every answer.
"""
from __future__ import annotations
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.config import file_hash,write_json,utc_now,load_config
from medft.data import read_jsonl,normalize
from medft.models import load_tokenizer
from scripts.train_consultation_preferences import encode_reply


def audit():
    protocol=json.loads((ROOT/'outputs/consultation-v5/protocol.json').read_text(encoding='utf-8'))
    for path,digest in {**protocol['protected'],**protocol['hashes']}.items():
        assert file_hash(ROOT/path)==digest, f'Changed protected/frozen file {path}'
    train=read_jsonl(ROOT/'data/generated/v5-consultation/train.jsonl')
    val=read_jsonl(ROOT/'data/generated/v5-consultation/validation.jsonl')
    test=read_jsonl(ROOT/'assets/consultation_v5_holdout_frozen.jsonl')
    assert len(train)==184 and len(val)==24 and len(test)==32
    sets=[{r['case_id'] for r in rows} for rows in [train,val,test]]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    prompts=[{normalize(m['content']) for r in rows for m in r.get('prompt',r.get('messages',[])) if m['role']=='user'} for rows in [train,val,test]]
    assert not prompts[0]&prompts[1] and not prompts[0]&prompts[2] and not prompts[1]&prompts[2]
    assert len({r['id'] for r in train+val+test})==len(train+val+test)
    config=load_config('configs/v5_consultation.yaml')
    tokenizer=load_tokenizer(config)
    max_length=0;supervised=0
    for r in train+val:
        prompt=[{'role':'system','content':config['system_prompt']}]+r['prompt']
        assert r['chosen']!=r['rejected']
        assert prompt[-1]['role']=='user'
        for key in ['chosen','rejected']:
            feature=encode_reply(tokenizer,prompt,r[key],config['training']['max_length'])
            prefix=tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True)
            assert feature['labels'][:len(prefix)]==[-100]*len(prefix)
            assert feature['labels'][len(prefix):]==feature['input_ids'][len(prefix):]
            max_length=max(max_length,len(feature['input_ids']))
            if key=='chosen':supervised+=sum(i!=-100 for i in feature['labels'])
    corpus=read_jsonl(ROOT/'data/generated/v5-consultation/corpus.jsonl')
    assert [r['messages'][-1]['content'] for r in corpus]==[r['chosen'] for r in train+val]
    manifest=json.loads((ROOT/'data/raw/consultation-v5-evidence/manifest.json').read_text(encoding='utf-8'))
    for s in manifest['sources']:
        path=ROOT/s['local_text']
        assert file_hash(path)==s['text_sha256'] and file_hash(path.with_suffix('.html'))==s['html_sha256']
    dest=ROOT/'outputs/v5-consultation'
    run=json.loads((dest/'run.json').read_text(encoding='utf-8'))
    assert run['code_sha256']==file_hash(ROOT/'scripts/train_consultation_preferences.py')
    ref=read_jsonl(dest/'reference_logps.jsonl')
    assert {(r['split'],r['id']) for r in ref}=={(s,r['id']) for s,rows in [('train',train),('validation',val)] for r in rows}
    assert all(math.isfinite(r[k]) for r in ref for k in ['chosen_logp','rejected_logp'])
    assert all(r['chosen_tokens']>0 and r['rejected_tokens']>0 for r in ref)
    assert run['reference_logps_sha256']==file_hash(dest/'reference_logps.jsonl')
    checkpoint=Path(run['selection']['best_model_checkpoint'])
    weight_sha=file_hash(dest/'adapter/adapter_model.safetensors')
    assert file_hash(checkpoint/'adapter_model.safetensors')==weight_sha==run['adapter_sha256']
    state=json.loads((checkpoint/'trainer_state.json').read_text(encoding='utf-8'))
    evaluations=[r for r in state['log_history'] if 'eval_loss' in r]
    assert math.isclose(min(r['eval_loss'] for r in evaluations),run['selection']['best_metric'],abs_tol=1e-7)
    result=dict(completed_at=utc_now(),verified=True,protected_files=len(protocol['protected']),
        train_count=len(train),validation_count=len(val),condition_holdout_count=len(test),
        train_groups=len(sets[0]),validation_groups=len(sets[1]),holdout_groups=len(sets[2]),
        split_exact_prompt_intersections=0,split_group_intersections=0,
        max_sequence_length=max_length,chosen_supervised_tokens=supervised,
        evidence_pages=len(manifest['sources']),reference_cache_rows=len(ref),
        selected_checkpoint=str(checkpoint),selected_weight_sha256=weight_sha,
        state_counts=dict(Counter(r['state'] for r in train)),
        limitations='Integrity checks only; synthetic preferences and source-informed holdout are not clinician-validated patient data.')
    write_json(ROOT/'outputs/consultation-v5/audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':audit()
