"""Restrict the adaptation capacity while keeping the curated data fixed."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from medft.config import load_config, project_path, write_json, utc_now, file_hash
from medft.training import plan_training

config = copy.deepcopy(load_config('configs/v3_1_complete_grouped.yaml'))
config['project']['name'] = 'medft-v3-3-attention-control'
config['training'].update(output_dir='outputs/v3-3-attention-control', learning_rate=5e-5,
                          epochs=2, lora_r=8, lora_alpha=16,
                          target_modules=['q_proj', 'v_proj'])
path = project_path('configs/v3_3_attention_control.yaml')
if path.exists():
    assert load_config(path) == config
else:
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
protocol = {
    'created_at': utc_now(), 'experiment': config['project']['name'],
    'reason': 'All-projection SFT with stronger learning learned stock templates but introduced false numerical trends and conversions. Restrict updates to attention q/v, reduce rank, and preserve the base feed-forward matrices. This is exploratory recipe selection, not a single-factor causal study.',
    'source_sha256': file_hash(project_path(config['data']['source'])),
    'starting_weights': 'Original base, fresh LoRA; no previous adapter continuation.',
    'checkpoint_selection': 'Minimum grouped validation loss, patience=3; no generation-based checkpoint selection.',
    'evaluation_role': 'Original 36 and frozen 29 are inspected development regressions. Retain all failures and all original scoring; no blind-test or clinical-superiority claim.',
    'plan': plan_training(config),
}
dest = project_path('outputs/optimization/v3-3-attention-protocol.json')
if not dest.exists(): write_json(dest, protocol)
print(protocol)
