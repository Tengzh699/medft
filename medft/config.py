from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = "configs/v1.yaml") -> dict:
    import yaml

    with project_path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("project", "model", "system_prompt", "data", "training", "generation"):
        if key not in config:
            raise ValueError(f"配置缺少字段：{key}")
    ratios = config["data"]
    if not (0 < ratios["validation_ratio"] < 1 and 0 <= ratios["test_ratio"] < 1):
        raise ValueError("验证集比例须大于0，测试集比例可为0；两者均须小于1。")
    if ratios["validation_ratio"] + ratios["test_ratio"] >= 1:
        raise ValueError("需要为训练集保留样本。")
    counts = ratios.get("split_counts")
    if counts is not None:
        if (not isinstance(counts, dict) or set(counts) != {"train", "validation", "test"}
                or any(type(n) is not int or n < 0 for n in counts.values())
                or counts["train"] == 0 or counts["validation"] == 0):
            raise ValueError("split_counts须指定正整数train、validation和非负整数test。")
        if counts["test"] != 0 or ratios["test_ratio"] != 0:
            raise ValueError("精确条数划分目前支持train/validation两份，请将test和test_ratio设为0。")
    if config["training"]["max_length"] <= 0:
        raise ValueError("max_length 必须为正整数。")
    return config


def split_names(data_spec: dict) -> tuple[str, ...]:
    """Only enabled splits participate in previews and training checks."""
    return ("train", "validation", "test") if data_spec["test_ratio"] > 0 else ("train", "validation")


def evaluation_paths(config: dict) -> list[Path]:
    extra = config["data"].get("evaluation_exclusions", [])
    if not isinstance(extra, list) or any(not isinstance(value, str) for value in extra):
        raise ValueError("evaluation_exclusions须为保留评测文件路径列表。")
    return list(dict.fromkeys(project_path(value) for value in [config["data"]["evaluation_source"], *extra]))


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
