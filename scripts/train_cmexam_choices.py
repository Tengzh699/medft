"""Choice-specific QLoRA experiment; never train on development/test answers.

Unlike explanation SFT, this objective learns one A-E decision per official
training question. It does not train explanations or claim consultation gains.
The original data, CLI, evaluation and historical adapters remain untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medft.config import load_config, project_path, file_hash, utc_now, write_json
from medft.models import load_tokenizer, load_quantized_model
from medft.training import load_training_data


def decision_feature(tokenizer, row, max_length):
    """Tokenize only the question; keep gold in a separate scalar target."""
    if row['task'] != 'medical_exam' or row['gold_answer'] not in 'ABCDE':
        raise ValueError('A single official exam answer is required')
    assert [m['role'] for m in row['messages']] == ['system', 'user', 'assistant']
    prompt = tokenizer.apply_chat_template(row['messages'][:2], tokenize=False,
                                           add_generation_prompt=True) + '答案：'
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_length:
        raise ValueError('Choice prompt exceeds length limit')
    for label in 'ABCDE':
        target = tokenizer.encode(label, add_special_tokens=False)
        assert len(target) == 1
        assert tokenizer.encode(prompt + label, add_special_tokens=False) == ids + target
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': 'ABCDE'.index(row['gold_answer'])}


class ChoiceCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        length = ((max(len(r['input_ids']) for r in features) + 7) // 8) * 8
        inputs = torch.full((len(features), length), self.pad_token_id, dtype=torch.long)
        masks = torch.zeros_like(inputs)
        for i, row in enumerate(features):
            n = len(row['input_ids'])
            inputs[i, -n:] = torch.tensor(row['input_ids'], dtype=torch.long)
            masks[i, -n:] = 1
        return {'input_ids': inputs, 'attention_mask': masks,
                'labels': torch.tensor([r['labels'] for r in features], dtype=torch.long)}


def choice_loss(logits, labels):
    import torch.nn.functional as F
    if logits.ndim != 2 or logits.shape[1] != 5 or labels.shape != logits.shape[:1]:
        raise ValueError('Expected five next-option logits and one target per question')
    return F.cross_entropy(logits.float(), labels)


def main():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/v4_2_choices.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    spec = config['training']
    assert spec['objective'] == 'next_option_five_class_cross_entropy'
    dest = project_path(spec['output_dir'])
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(dest)
    splits, stats = load_training_data(config)
    selected = {k: [r for r in splits[k] if r['task'] == 'medical_exam']
                for k in ('train', 'validation')}
    assert len(selected['train']) == 2000 and len(selected['validation']) == 100
    assert {r['source_file'] for r in selected['train']} == {'train.csv'}
    assert {r['source_file'] for r in selected['validation']} == {'val.csv'}
    assert not {r['case_id'] for r in selected['train']} & {r['case_id'] for r in selected['validation']}
    set_seed(config['project']['seed'])
    tokenizer = load_tokenizer(config)
    tokenizer.padding_side = 'left'
    features = {k: [decision_feature(tokenizer, r, spec['max_length']) for r in rows]
                for k, rows in selected.items()}
    ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in 'ABCDE']
    datasets = {k: Dataset.from_list(rows) for k, rows in features.items()}

    class ChoiceTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs['labels']
            output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                           use_cache=False, logits_to_keep=1)
            logits = output.logits[:, -1, ids]
            loss = choice_loss(logits, labels)
            return (loss, {'logits': logits}) if return_outputs else loss

    model, manifest = load_quantized_model(config)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=spec['lora_r'], lora_alpha=spec['lora_alpha'],
        lora_dropout=spec['lora_dropout'], bias='none', task_type='CAUSAL_LM',
        target_modules=spec['target_modules']))
    model.print_trainable_parameters()
    dest.mkdir(parents=True, exist_ok=True)
    metadata = {'started_at': utc_now(), 'config': config, 'model': manifest, 'data': stats,
        'objective': spec['objective'], 'smoke_run': False,
        'selection_inputs': {k: [r['id'] for r in rows] for k, rows in selected.items()},
        'excluded_non_exam': {k: len(splits[k]) - len(selected[k]) for k in selected},
        'source_sha256': stats['source_sha256'],
        'code_sha256': file_hash(Path(__file__)),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'limitation': 'Choice-specific loss; no explanation or consultation training. Fixed training-validation loss selects checkpoint, never final-test accuracy.'}
    write_json(dest / 'run.json', metadata)
    training_args = TrainingArguments(output_dir=str(dest / 'checkpoints'),
        per_device_train_batch_size=spec['batch_size'], per_device_eval_batch_size=8,
        gradient_accumulation_steps=spec['gradient_accumulation_steps'],
        learning_rate=spec['learning_rate'], num_train_epochs=spec['epochs'],
        eval_strategy='steps', save_strategy='steps', eval_steps=spec['eval_steps'],
        save_steps=spec['eval_steps'], load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, save_total_limit=2, logging_steps=spec['logging_steps'],
        lr_scheduler_type='cosine', warmup_ratio=spec['warmup_ratio'], max_grad_norm=0.3,
        optim='paged_adamw_8bit', gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=spec['compute_dtype'] == 'float16', bf16=spec['compute_dtype'] == 'bfloat16',
        report_to=['tensorboard'], dataloader_num_workers=0, remove_unused_columns=False,
        label_names=['labels'], seed=config['project']['seed'], data_seed=config['project']['seed'])
    trainer = ChoiceTrainer(model=model, args=training_args, processing_class=tokenizer,
        train_dataset=datasets['train'], eval_dataset=datasets['validation'],
        data_collator=ChoiceCollator(tokenizer.pad_token_id))
    # Equivalent to a custom compute_loss without num_items_in_batch scaling.
    # The model's **kwargs would otherwise incorrectly enable Trainer's token-sum path.
    trainer.model_accepts_loss_kwargs = False
    result = trainer.train()
    adapter = dest / 'adapter'
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    metadata.update({'completed_at': utc_now(), 'metrics': result.metrics, 'adapter_dir': str(adapter),
        'selection': {'global_step': trainer.state.global_step, 'max_steps': trainer.state.max_steps,
        'best_metric': trainer.state.best_metric, 'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'stopped_early': False}})
    write_json(dest / 'run.json', metadata)
    print(json.dumps(metadata['selection'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
