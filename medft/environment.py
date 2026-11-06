from __future__ import annotations

import platform
import shutil
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from .config import utc_now

PACKAGES = ("torch", "numpy", "pyarrow", "transformers", "peft", "accelerate", "datasets", "bitsandbytes", "sentencepiece", "tensorboard", "PyYAML")


def check_data_backend() -> dict:
    """Exercise NumPy/Arrow conversion; package metadata alone cannot validate it."""
    try:
        numpy = import_module("numpy")
        arrow = import_module("pyarrow")
        datasets = import_module("datasets")
        sample = datasets.Dataset.from_dict({"value": numpy.array([1, 2], dtype=numpy.int64)})
        if len(sample) != 2 or sample[0]["value"] != 1:
            raise ValueError("Dataset 读取结果与输入不一致。")
        return {"ok": True, "rows": len(sample), "numpy": numpy.__version__,
                "pyarrow": arrow.__version__, "datasets": datasets.__version__}
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def check_environment(*, test_4bit: bool = True) -> dict:
    report = {
        "checked_at": utc_now(), "python": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "free_disk_gib": round(shutil.disk_usage(__file__).free / 1024**3, 1),
        "packages": {}, "errors": [],
    }
    for name in PACKAGES:
        try:
            report["packages"][name] = version(name)
        except PackageNotFoundError:
            report["packages"][name] = None
            report["errors"].append(f"缺少依赖 {name}")
    report["data_backend"] = check_data_backend()
    if not report["data_backend"]["ok"]:
        report["errors"].append("数据处理库检查失败：" + report["data_backend"]["error"])
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if not torch.cuda.is_available():
            report["errors"].append("PyTorch 未检测到 CUDA GPU。")
        else:
            device = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(device)
            report["gpu"] = {"name": torch.cuda.get_device_name(device), "total_gib": round(total / 1024**3, 2), "free_gib": round(free / 1024**3, 2)}
            if test_4bit:
                import bitsandbytes as bnb

                device_name = f"cuda:{device}"
                layer = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.float16, quant_type="nf4").to(device_name)
                with torch.inference_mode():
                    result = layer(torch.randn(2, 64, device=device_name, dtype=torch.float16))
                torch.cuda.synchronize()
                report["4bit_test"] = {"shape": list(result.shape), "finite": bool(torch.isfinite(result).all().item())}
                if not report["4bit_test"]["finite"]:
                    report["errors"].append("4bit 运算出现非有限值。")
    except Exception as error:
        report["errors"].append(f"GPU / 4bit 检查失败：{type(error).__name__}: {error}")
    report["ok"] = not report["errors"]
    return report
