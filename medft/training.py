from __future__ import annotations

import json
import math
import os
from collections import Counter

from .config import evaluation_paths, file_hash, project_path, split_names, utc_now, write_json
from .data import assert_no_leakage, read_jsonl
from .environment import check_data_backend
from .formatting import AssistantOnlyCollator, SequenceTooLong, encode_conversation
from .models import load_quantized_model, load_tokenizer
from .public_filter import assert_current_public_filter
from .quality import validate_grounding


def encode_splits(rows: dict, tokenizer, spec: dict) -> tuple[dict, dict]:
    encoded, summary = {}, {}
    for name in ("train", "validation"):
        features, skipped, tasks = [], [], Counter()
        for row in rows[name]:
            try:
                features.append(encode_conversation(tokenizer, row["messages"], spec["max_length"]))
                tasks[row["task"]] += 1
            except SequenceTooLong:
                skipped.append(row["id"])
        if not features:
            raise ValueError(f"{name} 没有符合长度限制的样本，请调整 max_length 或数据。")
        if skipped and spec.get("require_all_samples", False):
            raise ValueError(f"{name} 有{len(skipped)}条超长，实验不允许静默丢样本；请调整max_length或数据：{skipped[:5]}")
        lengths = sorted(len(feature["input_ids"]) for feature in features)
        encoded[name] = features
        summary[name] = {
            "used": len(features), "by_task": dict(tasks), "too_long_ids": skipped,
            "supervised_tokens": sum(sum(label != -100 for label in feature["labels"]) for feature in features),
            "length_tokens": {"min": lengths[0], "median": lengths[len(lengths) // 2],
                              "p95": lengths[math.ceil(len(lengths) * 0.95) - 1], "max": lengths[-1]},
        }
    return encoded, summary


def plan_training(config: dict) -> dict:
    """Inspect local data and tokenizer; never load 7B weights or start training."""
    rows, stats = load_training_data(config)
    spec = config["training"]
    schedule = checkpoint_schedule(spec)
    _, lengths = encode_splits(rows, load_tokenizer(config), spec)
    batch, accumulation, epochs = spec["batch_size"], spec["gradient_accumulation_steps"], spec["epochs"]
    if any(type(value) is not int or value < 1 for value in (batch, accumulation)) or epochs <= 0:
        raise ValueError("batch_size、gradient_accumulation_steps须为正整数，epochs须为正数。")
    batches = math.ceil(lengths["train"]["used"] / batch)
    # Match set_initial_training_values in the pinned Transformers 4.51.3.
    steps_per_epoch = max(batches // accumulation, 1)
    maximum = math.ceil(epochs * steps_per_epoch)
    interval = schedule.get("eval_steps", steps_per_epoch)
    return {
        "project": config["project"]["name"], "loads_model_weights": False,
        "training_starts_from": "original_base_with_new_lora",
        "learning_rate": spec["learning_rate"], "effective_batch_size": batch * accumulation,
        "planned_max_steps": maximum, "warmup_steps": math.ceil(maximum * spec["warmup_ratio"]),
        "evaluation_steps": list(range(interval, maximum + 1, interval)),
        "early_stopping_patience": spec.get("early_stopping_patience"),
        "tokenization": lengths, "validation_sha256": stats["splits"]["validation"]["sha256"],
        "source_sha256": stats["source_sha256"],
        "interpretation": "Step estimate follows pinned Transformers 4.51.3, single GPU; early stopping may shorten the run. Validation loss is not task quality.",
    }


def checkpoint_schedule(spec: dict, *, smoke: bool = False) -> dict:
    if smoke:
        return {"eval_strategy": "no", "save_strategy": "no"}
    steps = spec.get("eval_steps")
    if steps is None:
        return {"eval_strategy": "epoch", "save_strategy": "epoch"}
    if type(steps) is not int or steps < 1:
        raise ValueError("eval_steps须为正整数。")
    return {"eval_strategy": "steps", "save_strategy": "steps", "eval_steps": steps, "save_steps": steps}


def training_callbacks(spec: dict, *, smoke: bool = False) -> list:
    patience = spec.get("early_stopping_patience")
    threshold = spec.get("early_stopping_threshold", 0.0)
    if patience is None:
        if "early_stopping_threshold" in spec:
            raise ValueError("early_stopping_threshold须与early_stopping_patience一起使用。")
        return []
    if type(patience) is not int or patience < 1:
        raise ValueError("early_stopping_patience须为正整数。")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("early_stopping_threshold须为非负有限数字。")
    if smoke:
        return []
    from transformers import EarlyStoppingCallback

    return [EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=threshold)]


def load_training_data(config: dict) -> tuple[dict, dict]:
    """Validate the enabled splits before loading a tokenizer or a model."""
    data_dir = project_path(config["data"]["output_dir"])
    paths = {name: data_dir / f"{name}.jsonl" for name in split_names(config["data"])}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("尚未构建数据，请先运行 python -m medft build-data。")
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    assert_current_public_filter((row for split in rows.values() for row in split), config)
    assert_no_leakage(rows)
    for split in rows.values():
        for record in split:
            validate_grounding(record, required=config.get("data_generation", {}).get("template_profile") == "v2_1")
    stats_path = data_dir / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError("缺少数据清洗记录 stats.json，请重新构建数据。")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    source = project_path(config["data"]["source"])
    if not source.is_file() or file_hash(source) != stats["source_sha256"]:
        raise ValueError("源数据已缺失或修改，请重新构建数据以更新数据记录。")
    fixed_value = config["data"].get("fixed_validation_source")
    if fixed_value:
        fixed = project_path(fixed_value)
        if (not fixed.is_file() or str(fixed.resolve()) != stats.get("fixed_validation_source")
                or file_hash(fixed) != stats.get("fixed_validation_sha256")):
            raise ValueError("固定验证源已缺失、修改或更换，请重新构建数据。")
    elif "fixed_validation_sha256" in stats:
        raise ValueError("固定验证配置已移除，请重新构建数据。")
    if set(stats["splits"]) != set(paths):
        raise ValueError("数据划分与当前配置不同，请重新构建数据。")
    for name, path in paths.items():
        if file_hash(path) != stats["splits"][name]["sha256"]:
            raise ValueError(f"{name} 数据已被修改，请重新运行 build-data 以更新数据记录。")
    counts = config["data"].get("split_counts")
    if counts is not None and any(len(rows.get(name, [])) != count for name, count in counts.items()):
        raise ValueError("数据条数与当前split_counts不同，请重新构建数据。")
    if file_hash(project_path(config["data"]["evaluation_source"])) != stats["evaluation_sha256"]:
        raise ValueError("独立评测题已变更，请重新构建数据并检查污染。")
    if config["data"].get("evaluation_exclusions"):
        expected = {str(path.resolve()): file_hash(path) for path in evaluation_paths(config)}
        if stats.get("evaluation_sources_sha256") != expected:
            raise ValueError("保留评测文件或排除列表已变更，请重新构建数据并检查污染。")
    if any(row["messages"][0] != {"role": "system", "content": config["system_prompt"]} for split in rows.values() for row in split):
        raise ValueError("数据中的 system prompt 与当前配置不同，请重新构建数据。")
    return rows, stats


def train(config: dict, *, max_steps: int | None = None, output_dir: str | None = None, resume: str | None = None) -> dict:
    backend = check_data_backend()
    if not backend["ok"]:
        raise RuntimeError(
            "数据处理库检查失败：" + backend["error"]
            + "\n请在 medft 环境执行 python -m pip install -r requirements.txt，"
            "然后运行 python -m medft check。"
        )
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("当前配置面向单 GPU，请使用 python -m medft train 启动。")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps 必须为正整数。")
    spec = config["training"]
    schedule = checkpoint_schedule(spec, smoke=max_steps is not None)
    callbacks = training_callbacks(spec, smoke=max_steps is not None)
    destination = project_path(output_dir or (spec["output_dir"] + "-smoke" if max_steps else spec["output_dir"]))
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(f"训练目录已有内容：{destination}；请指定新的 --output-dir，或使用 --resume。")
    if resume and not project_path(resume).is_dir():
        raise FileNotFoundError(f"断点不存在：{resume}")
    rows, stats = load_training_data(config)

    set_seed(config["project"]["seed"])
    tokenizer = load_tokenizer(config)
    encoded, lengths = encode_splits(rows, tokenizer, spec)
    tokenized = {name: Dataset.from_list(features) for name, features in encoded.items()}
    for name, summary in lengths.items():
        print(f"{name}: 可用 {summary['used']} 条，超长跳过 {len(summary['too_long_ids'])} 条。", flush=True)

    model, manifest = load_quantized_model(config)
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = set(spec["target_modules"]) - available
    if missing:
        raise ValueError(f"模型中不存在这些 LoRA 模块：{sorted(missing)}")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=spec["lora_r"], lora_alpha=spec["lora_alpha"], lora_dropout=spec["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=spec["target_modules"],
    ))
    model.print_trainable_parameters()
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"started_at": utc_now(), "config": config, "model": manifest, "data": stats,
                "tokenization": lengths, "smoke_run": max_steps is not None,
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
    write_json(destination / "run.json", metadata)
    arguments = TrainingArguments(
        output_dir=str(destination / "checkpoints"),
        per_device_train_batch_size=spec["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=spec["gradient_accumulation_steps"],
        learning_rate=spec["learning_rate"], num_train_epochs=spec["epochs"],
        max_steps=max_steps if max_steps is not None else -1,
        **schedule,
        load_best_model_at_end=max_steps is None, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, logging_steps=spec["logging_steps"], logging_dir=str(destination / "logs"),
        lr_scheduler_type="cosine", warmup_ratio=spec["warmup_ratio"], max_grad_norm=0.3,
        optim="paged_adamw_8bit", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=spec["compute_dtype"] == "float16", bf16=spec["compute_dtype"] == "bfloat16",
        report_to=["tensorboard"], dataloader_num_workers=0, remove_unused_columns=False,
        label_names=["labels"],  # PEFT hides the base signature; evaluation must recognize labels.
        restore_callback_states_from_checkpoint=bool(callbacks),
        seed=config["project"]["seed"], data_seed=config["project"]["seed"],
    )
    trainer = Trainer(
        model=model, args=arguments, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        processing_class=tokenizer, data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    result = trainer.train(resume_from_checkpoint=str(project_path(resume)) if resume else None)
    adapter_dir = destination / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    metadata.update({"completed_at": utc_now(), "metrics": result.metrics, "adapter_dir": str(adapter_dir),
                     "selection": {"global_step": trainer.state.global_step, "max_steps": trainer.state.max_steps,
                                   "best_metric": trainer.state.best_metric, "best_model_checkpoint": trainer.state.best_model_checkpoint,
                                   "stopped_early": trainer.state.global_step < trainer.state.max_steps}})
    write_json(destination / "run.json", metadata)
    return metadata
