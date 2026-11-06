"""Flag explicit unavailable-media references without changing frozen data.

This is a preflight for future datasets. An audit after final-test launch must
never silently remove examples, rescore the test or replace trained weights.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medft.config import project_path, file_hash, write_json, utc_now
from medft.data import read_jsonl

REFERENCE = re.compile(r'暂无图|缺[少失]图片|(?:如|下|附)图|见下表|如下表')


def media_hints(text):
    # "心电图示ST段…", "超声心动图示…", "镜下表现…" contain
    # textual findings and are not themselves references to missing media.
    return [m.group(0) for m in REFERENCE.finditer(text)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = project_path(args.input)
    dest = project_path(args.output)
    if dest.exists():
        raise FileExistsError(dest)
    flags = []
    rows = read_jsonl(source)
    for row in rows:
        prompt = '\n'.join(m['content'] for m in row['messages'] if m['role'] == 'user')
        if hints := media_hints(prompt):
            flags.append({'id': row['id'], 'hints': hints, 'status': 'needs_input_completeness_review',
                          'source_file': row.get('source_file'), 'source_line': row.get('source_line'),
                          'excerpt': prompt[-1000:]})
    write_json(dest, {'checked_at': utc_now(), 'input': str(source), 'source_sha256': file_hash(source),
                     'count': len(rows), 'flagged': flags, 'mutated_source': False,
                     'policy': 'For future training, resolve the referenced media or quarantine flagged rows before freezing. Historical training/evaluations remain unchanged; a wording hint alone does not prove that the media is essential.'})
    print(f'{len(flags)} explicit media references flagged; source unchanged.', flush=True)


if __name__ == '__main__':
    main()
