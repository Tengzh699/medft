"""Publish mechanically screened retention examples, preserving every candidate.

This screens completeness and obvious dose instructions. It does not certify
medical accuracy; all accepted rows deliberately keep review_status=unreviewed.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.config import project_path, file_hash, write_json
from medft.data import read_jsonl, write_jsonl


def screen(rows, exclusions=None):
    accepted,rejected=[],[]
    for row in rows:
        text=row['messages'][-1]['content']
        reasons=[]
        if exclusions and row['id'] in exclusions:
            reasons.append('engineering_review_exclusion: '+exclusions[row['id']])
        if not row.get('teacher_finished'):reasons.append('generation_limit_without_eos')
        if len(text)<100:reasons.append('insufficient_explanation_length')
        if not text.endswith(('。','！','？','.','!','?')):reasons.append('incomplete_ending')
        if re.search(r'(?:一次|每次|每日|每天|起始剂量)[^。\n]{0,15}\d+(?:\.\d+)?\s*(?:mg|毫克|克|g)',text,re.I):
            reasons.append('specific_dose_instruction_requires_professional_review')
        if reasons:rejected.append({'id':row['id'],'reasons':reasons})
        else:accepted.append(row)
    return accepted,rejected


def main():
    source=project_path('data/generated/v3-1-complete/base_replay_candidates.jsonl')
    rows=read_jsonl(source)
    if len(rows)!=180:raise ValueError(f'Expected 180 completed candidates, found {len(rows)}')
    exclusions_path=project_path('assets/complete_replay_exclusions.json')
    exclusions=json.loads(exclusions_path.read_text(encoding='utf-8'))['excluded']
    if set(exclusions)-{r['id'] for r in rows}:
        raise ValueError('Review exclusions refer to nonexistent candidates.')
    accepted,rejected=screen(rows,exclusions)
    if len(accepted)<60:raise ValueError(f'Only {len(accepted)} eligible complete answers; inspect before training.')
    dest=source.with_name('base_replay.jsonl')
    if dest.exists() and read_jsonl(dest)!=accepted:raise FileExistsError(f'Preserve published data: {dest}')
    if not dest.exists():write_jsonl(dest,accepted)
    write_json(project_path('outputs/optimization/base-replay-screening.json'),{
        'candidate_count':len(rows),'accepted_count':len(accepted),'rejected':rejected,
        'candidate_sha256':file_hash(source),'accepted_sha256':file_hash(dest),
        'engineering_exclusions_sha256':file_hash(exclusions_path),
        'clinician_reviewed':False,'scope':'Engineering completeness/dose screening; base answers can still contain factual errors.'})
    print(f'Replay: {len(accepted)} accepted, {len(rejected)} preserved but excluded.')


if __name__=='__main__':main()
