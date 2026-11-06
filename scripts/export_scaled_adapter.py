"""Calibrate LoRA update strength without changing learned matrices or base.

Checkpoint selection stays on fixed validation loss. This separate inference
calibration must use development data, never final-test accuracy.
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medft.config import project_path, file_hash, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--scale', type=float, choices=[0.25,0.5], required=True)
    args = parser.parse_args()
    source = project_path(args.source)
    dest = project_path(args.output_dir)
    run = json.loads((source.parent/'run.json').read_text(encoding='utf-8'))
    assert run.get('completed_at') and run.get('adapter_dir') == str(source)
    original = json.loads((source/'adapter_config.json').read_text(encoding='utf-8'))
    assert not any(original.get(k) for k in ['alpha_pattern','rank_pattern','modules_to_save','use_dora','use_rslora'])
    if dest.exists():
        raise FileExistsError(dest)
    # Config scaling implements W + scale * (alpha/r) B A. Keep every B/A byte.
    shutil.copytree(source, dest)
    scaled = dict(original)
    scaled['lora_alpha'] = int(original['lora_alpha']*args.scale)
    assert scaled['lora_alpha']/original['lora_alpha'] == args.scale
    write_json(dest/'adapter_config.json',scaled)
    assert file_hash(dest/'adapter_model.safetensors') == file_hash(source/'adapter_model.safetensors')
    metadata = {'created_at':utc_now(),'kind':'lora_inference_strength_calibration',
                'model':run['model'], 'source_adapter':str(source),
                'source_run_sha256':file_hash(source.parent/'run.json'),
                'source_adapter_config_sha256':file_hash(source/'adapter_config.json'),
                'adapter_sha256':file_hash(source/'adapter_model.safetensors'),
                'scale':args.scale,'original_alpha':original['lora_alpha'],'scaled_alpha':scaled['lora_alpha'],
                'selection_policy':'Development checks only; no new training and no final-test-based selection.'}
    write_json(dest.parent/'derived_adapter.json',metadata)
    # Generator uses only the model identity from this file. Do not copy a
    # training run and pretend that the derived artifact was newly trained.
    write_json(dest.parent/'run.json',metadata)
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
