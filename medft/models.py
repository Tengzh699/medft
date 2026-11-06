from __future__ import annotations

import json
from pathlib import Path

from .config import project_path, utc_now, write_json


def download_model(config: dict) -> dict:
    from huggingface_hub import snapshot_download

    spec = config["model"]
    location = snapshot_download(
        repo_id=spec["name"], revision=spec["revision"], cache_dir=str(project_path(spec["cache_dir"])),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "LICENSE*", "README*"],
    )
    manifest = {
        "repo_id": spec["name"], "requested_revision": spec["revision"],
        "resolved_revision": Path(location).name, "path": str(Path(location).resolve()), "downloaded_at": utc_now(),
    }
    write_json(project_path(spec["manifest"]), manifest)
    return manifest


def model_location(config: dict) -> tuple[Path, dict]:
    spec = config["model"]
    if spec.get("local_path"):
        location = project_path(spec["local_path"])
        if not (location / "config.json").is_file():
            raise FileNotFoundError(f"本地模型目录缺少 config.json：{location}")
        return location, {"repo_id": spec["name"], "path": str(location), "resolved_revision": "local-unversioned"}
    manifest_path = project_path(spec["manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("模型尚未准备好。请先运行 python -m medft download，或配置 model.local_path。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["repo_id"] != spec["name"] or manifest["requested_revision"] != spec["revision"]:
        raise ValueError("模型下载记录与当前配置不一致，请重新运行 download。")
    location = Path(manifest["path"])
    if not location.is_dir():
        raise FileNotFoundError(f"缓存目录已移动或删除：{location}")
    return location, manifest


def load_tokenizer(config: dict):
    from transformers import AutoTokenizer

    location, _ = model_location(config)
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True, trust_remote_code=config["model"]["trust_remote_code"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("分词器没有可用的 pad/eos token。")
    tokenizer.padding_side = "right"
    return tokenizer


def load_quantized_model(config: dict):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA 需要 CUDA GPU，请先运行 python -m medft check。")
    location, manifest = model_location(config)
    dtype_name = config["training"]["compute_dtype"]
    if dtype_name not in ("float16", "bfloat16"):
        raise ValueError("compute_dtype 只支持 float16 或 bfloat16。")
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持所配置的 bfloat16。")
    dtype = getattr(torch, dtype_name)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        location, local_files_only=True, trust_remote_code=config["model"]["trust_remote_code"],
        quantization_config=quantization, torch_dtype=dtype, device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa",
    )
    return model, manifest
