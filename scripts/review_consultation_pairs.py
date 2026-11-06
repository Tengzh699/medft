"""Export raw paired answers for transparent content review, without grading."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.config import project_path,write_json,file_hash
from medft.data import read_jsonl


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-dir',required=True);p.add_argument('--adapter-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    bdir,adir=project_path(args.base_dir),project_path(args.adapter_dir)
    base_summary=json.loads((bdir/'summary.json').read_text(encoding='utf-8'))
    tuned_summary=json.loads((adir/'summary.json').read_text(encoding='utf-8'))
    for key in ['model','system_prompt','generation','seed','mode','evaluation_sha256']:
        assert base_summary[key]==tuned_summary[key], f'Different conditions: {key}'
    base=read_jsonl(bdir/'predictions.jsonl');tuned=read_jsonl(adir/'predictions.jsonl')
    assert [r['id'] for r in base]==[r['id'] for r in tuned]
    out=project_path(args.output)
    if out.exists():raise FileExistsError(out)
    lines=['# 基模与问诊微调：原始回答对照','','未经临床审核；逐题判断事实、行动顺序和附加建议，不能只读关键词分数。','']
    changes={'gained':[],'lost':[]}
    for b,a in zip(base,tuned):
        assert b['messages']==a['messages']
        lines += ['## '+a['id'],'','\n\n'.join(m['role']+': '+m['content'] for m in a['messages']),'']
        if 'scores' in a:
            bp,ap=b['scores']['required_groups_pass'],a['scores']['required_groups_pass']
            lines += [f'原规则通过：基模 {bp} / adapter {ap}','']
            if ap and not bp:changes['gained'].append(a['id'])
            if bp and not ap:changes['lost'].append(a['id'])
        lines += ['### 基模','',b['raw_output'],'','### 微调后','',a['raw_output'],'']
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines),encoding='utf-8')
    write_json(out.with_suffix('.json'),dict(count=len(base),changes=changes,
        base_predictions_sha256=file_hash(bdir/'predictions.jsonl'),adapter_predictions_sha256=file_hash(adir/'predictions.jsonl'),
        manual_verdicts_added=False))
    print(json.dumps(changes,ensure_ascii=False))


if __name__=='__main__':main()
