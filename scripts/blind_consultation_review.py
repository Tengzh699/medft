"""Shuffle A/B identities for the new holdout's human/assistant content review.

The mapping is saved separately; grades should be recorded before it is opened.
This reduces identity bias, not the lack of clinical expertise or independent
authorship. No automatic medical grading is performed here.
"""
import argparse
import json
import secrets
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.data import read_jsonl,write_jsonl
from medft.config import project_path,write_json,file_hash,utc_now


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-dir',required=True);p.add_argument('--adapter-dir',required=True);p.add_argument('--output-dir',required=True)
    args=p.parse_args();bdir=project_path(args.base_dir);adir=project_path(args.adapter_dir);dest=project_path(args.output_dir)
    if dest.exists():raise FileExistsError(dest)
    summaries=[json.loads((d/'summary.json').read_text(encoding='utf-8')) for d in [bdir,adir]]
    for key in ['model','evaluation_sha256','generation','system_prompt','seed','mode']:
        assert summaries[0][key]==summaries[1][key],key
    base,tuned=[read_jsonl(d/'predictions.jsonl') for d in [bdir,adir]]
    assert [r['id'] for r in base]==[r['id'] for r in tuned]
    rows=[];mapping={}
    for b,a in zip(base,tuned):
        assert b['messages']==a['messages']
        reverse=bool(secrets.randbits(1))
        mapping[b['id']]={'A':'adapter' if reverse else 'base','B':'base' if reverse else 'adapter'}
        rows.append(dict(id=b['id'],case_id=b['case_id'],state=b['state'],messages=b['messages'],source=b['source'],
            rubric=b['review_rubric'],A=(a if reverse else b)['raw_output'],B=(b if reverse else a)['raw_output']))
    dest.mkdir(parents=True)
    write_jsonl(dest/'blind.jsonl',rows)
    write_json(dest/'identity-key.json',mapping)
    write_json(dest/'freeze.json',dict(created_at=utc_now(),cases=len(rows),blind_sha256=file_hash(dest/'blind.jsonl'),
        key_sha256=file_hash(dest/'identity-key.json'),
        base_predictions_sha256=file_hash(bdir/'predictions.jsonl'),adapter_predictions_sha256=file_hash(adir/'predictions.jsonl'),
        instruction='Record per-answer grades and reasons before opening identity-key.json. Same source-author and engineering reviewer; not clinician validation.'))
    print('Blinded',len(rows),'pairs. Review blind.jsonl before identity-key.json.')


if __name__=='__main__':main()
