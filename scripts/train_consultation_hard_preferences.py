"""V5.1 minimum-edit reference-cached DPO with a small chosen-response SFT anchor.

Reference is the same original NF4 base, not the exam adapter. No external TRL
dependency. Final assistant turn only; no user/history-answer supervision.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.config import load_config, project_path, write_json, file_hash, utc_now
from medft.data import read_jsonl, write_jsonl
from medft.models import load_quantized_model, load_tokenizer
from medft.formatting import AssistantOnlyCollator, SequenceTooLong


def encode_reply(tokenizer, prompt, answer, max_length):
    prefix=tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(prompt+[{'role':'assistant','content':answer}],
                                      tokenize=True,add_generation_prompt=False)
    if full[:len(prefix)]!=prefix: raise ValueError('Non-prefix chat template')
    if len(full)>max_length: raise SequenceTooLong(f'{len(full)} > {max_length}')
    if len(full)<=len(prefix): raise ValueError('No supervised answer')
    return dict(input_ids=full,attention_mask=[1]*len(full),
                labels=[-100]*len(prefix)+full[len(prefix):])


def sequence_logps(logits, labels):
    """Sum log p(y_t|x,y_<t) at exactly the final assistant's labels."""
    import torch.nn.functional as F
    shifted=labels[:,1:].contiguous()
    mask=shifted.ne(-100)
    # CE avoids retaining a second batch x length x vocabulary logsoftmax.
    losses=F.cross_entropy(logits[:,:-1,:].transpose(1,2).float(),shifted,
                           ignore_index=-100,reduction='none')
    counts=mask.sum(-1)
    if not bool(counts.gt(0).all()): raise ValueError('Empty supervised reply')
    return -losses.sum(-1),counts


def preference_loss(policy_logps, reference_logps, counts, beta=.1, sft_weight=.1):
    import torch.nn.functional as F
    half=policy_logps.shape[0]//2
    chosen,rejected=policy_logps[:half],policy_logps[half:]
    advantage=(chosen-rejected)-(reference_logps[:half]-reference_logps[half:])
    dpo=-F.logsigmoid(beta*advantage).mean()
    sft=(-chosen/counts[:half]).mean()
    return dpo+sft_weight*sft,dict(dpo=dpo.detach(),sft=sft.detach(),
                                  preference_accuracy=advantage.gt(0).float().mean().detach())


@dataclass
class PreferenceCollator:
    pad_token_id:int
    def __call__(self,features):
        import torch
        batch=AssistantOnlyCollator(self.pad_token_id)([r['chosen'] for r in features]+[r['rejected'] for r in features])
        batch['reference_logps']=torch.tensor([r['reference_chosen'] for r in features]+[r['reference_rejected'] for r in features],dtype=torch.float32)
        return batch


def main():
    import torch
    from datasets import Dataset
    from transformers import Trainer,TrainingArguments,set_seed
    from peft import get_peft_model,LoraConfig,prepare_model_for_kbit_training
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/v5_1_consultation_hard.yaml')
    args=parser.parse_args()
    config=load_config(args.config);spec=config['training']
    dest=project_path(spec['output_dir'])
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    protocol=json.loads((ROOT/'outputs/consultation-v5-1/protocol.json').read_text(encoding='utf-8'))
    for path,digest in protocol['hashes'].items():
        if file_hash(ROOT/path)!=digest: raise ValueError(f'Frozen input changed: {path}')
    rows={s:read_jsonl(ROOT/f'data/generated/v5-1-consultation-hard/{s}.jsonl') for s in ['train','validation']}
    tokenizer=load_tokenizer(config)
    features={}
    for split,records in rows.items():
        features[split]=[]
        for r in records:
            prompt=[{'role':'system','content':config['system_prompt']}]+r['prompt']
            features[split].append(dict(id=r['id'],chosen=encode_reply(tokenizer,prompt,r['chosen'],spec['max_length']),
                rejected=encode_reply(tokenizer,prompt,r['rejected'],spec['max_length'])))
    set_seed(config['project']['seed'])
    model,manifest=load_quantized_model(config)
    # Cache the reference AFTER the same k-bit preparation used by the policy.
    # It casts nonquantized modules to fp32; caching before it changes numerics.
    model=prepare_model_for_kbit_training(model)
    model.eval();model.config.use_cache=False
    dest.mkdir(parents=True,exist_ok=True)
    cache=[]
    for split,pairs in features.items():
        for i,pair in enumerate(pairs):
            tensors=AssistantOnlyCollator(tokenizer.pad_token_id)([pair['chosen'],pair['rejected']])
            tensors={k:v.to(model.device) for k,v in tensors.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=getattr(torch, spec["compute_dtype"])):
                output=model(input_ids=tensors['input_ids'],attention_mask=tensors['attention_mask'],use_cache=False)
                logps,counts=sequence_logps(output.logits,tensors['labels'])
            pair.update(reference_chosen=logps[0].item(),reference_rejected=logps[1].item())
            cache.append(dict(split=split,id=pair['id'],chosen_logp=logps[0].item(),rejected_logp=logps[1].item(),
                              chosen_tokens=counts[0].item(),rejected_tokens=counts[1].item()))
            del output,tensors,logps,counts
            if (i+1)%24==0:print(f'Reference {split} {i+1}/{len(pairs)}',flush=True)
    write_jsonl(dest/'reference_logps.jsonl',cache)
    model=get_peft_model(model,LoraConfig(r=spec['lora_r'],lora_alpha=spec['lora_alpha'],lora_dropout=spec['lora_dropout'],
        target_modules=spec['target_modules'],bias='none',task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    metadata=dict(started_at=utc_now(),config=config,model=manifest,objective=spec['objective'],
        source_hashes=protocol['hashes'],reference_logps_sha256=file_hash(dest/'reference_logps.jsonl'),
        code_sha256=file_hash(Path(__file__)),data={s:dict(rows=len(v),ids=[r['id'] for r in v]) for s,v in rows.items()},
        starts_from='original_base_with_new_lora',trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        limits='Synthetic source-backed preference pairs, engineering review only. Preference loss not consultation accuracy.')
    write_json(dest/'run.json',metadata)

    class PreferenceTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            ref=inputs.pop('reference_logps');labels=inputs.pop('labels')
            output=model(**inputs,use_cache=False)
            logps,counts=sequence_logps(output.logits,labels)
            loss,metrics=preference_loss(logps,ref,counts,spec['beta'],spec['chosen_sft_weight'])
            return (loss,{'logits':logps.unsqueeze(-1)}) if return_outputs else loss

    training_args=TrainingArguments(output_dir=str(dest/'checkpoints'),per_device_train_batch_size=spec['batch_size'],
        per_device_eval_batch_size=1,gradient_accumulation_steps=spec['gradient_accumulation_steps'],
        learning_rate=spec['learning_rate'],num_train_epochs=spec['epochs'],
        eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='eval_loss',greater_is_better=False,save_total_limit=2,
        logging_steps=4,lr_scheduler_type='cosine',warmup_ratio=.1,max_grad_norm=.3,
        optim='paged_adamw_8bit',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        fp16=spec['compute_dtype']=='float16',bf16=spec['compute_dtype']=='bfloat16',
        report_to=['tensorboard'],dataloader_num_workers=0,remove_unused_columns=False,
        label_names=['labels'],seed=config['project']['seed'],data_seed=config['project']['seed'])
    datasets={s:Dataset.from_list(v) for s,v in features.items()}
    trainer=PreferenceTrainer(model=model,args=training_args,processing_class=tokenizer,
        train_dataset=datasets['train'],eval_dataset=datasets['validation'],data_collator=PreferenceCollator(tokenizer.pad_token_id))
    trainer.model_accepts_loss_kwargs=False
    result=trainer.train()
    adapter=dest/'adapter';trainer.save_model(str(adapter));tokenizer.save_pretrained(adapter)
    metadata.update(completed_at=utc_now(),metrics=result.metrics,adapter_dir=str(adapter),
        selection=dict(global_step=trainer.state.global_step,max_steps=trainer.state.max_steps,
                       best_metric=trainer.state.best_metric,best_model_checkpoint=trainer.state.best_model_checkpoint),
        adapter_sha256=file_hash(adapter/'adapter_model.safetensors'))
    write_json(dest/'run.json',metadata)
    print(json.dumps(metadata['selection'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
