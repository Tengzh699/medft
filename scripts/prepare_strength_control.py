"""Record a stronger SFT control before running it; keep the data unchanged."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from medft.config import load_config, project_path, write_json, utc_now, file_hash
from medft.training import plan_training

config = copy.deepcopy(load_config('configs/v3_1_complete_grouped.yaml'))
config['project']['name'] = 'medft-v3-2-strength-control'
config['training'].update(output_dir='outputs/v3-2-strength-control',
                          learning_rate=1e-4, epochs=3)
path = project_path('configs/v3_2_strength_control.yaml')
if path.exists():
    assert load_config(path) == config, 'Preserve a different existing config.'
else:
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
protocol = {
    'created_at': utc_now(), 'experiment': config['project']['name'],
    'reason': 'V3.1 1e-5 SFT retained verbose base completions and failed to learn several authored constraints. Test a stronger update on exactly the same grouped data, including 102 accepted base replay answers.',
    'source_sha256': file_hash(project_path(config['data']['source'])),
    'changes': {'learning_rate': [1e-5, 1e-4], 'maximum_epochs': [2, 3]},
    'checkpoint_selection': 'Minimum grouped validation loss among checkpoints, patience=3; no generation-score checkpoint selection.',
    'evaluation': 'Original 36 and frozen 29 development checks, unchanged. Both have now been inspected and are not blind tests. Keep failures and compare raw outputs against the same base.',
    'starting_weights': 'Original base with fresh LoRA; no continuation of a degraded adapter.',
    'plan': plan_training(config),
}
dest = project_path('outputs/optimization/v3-2-strength-protocol.json')
if not dest.exists():
    write_json(dest, protocol)
print(protocol)
