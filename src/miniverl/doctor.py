"""Environment diagnostics.

``miniverl doctor`` must work from a bare ``pip install miniverl``, so nothing
here imports torch: optional dependencies are probed with
:func:`importlib.util.find_spec` and only imported once they are known to exist.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.environments.registry import available_environments
from miniverl.utils.lazy import have_module

__all__ = ["Check", "DoctorReport", "run_doctor"]

_REQUIRED = ("typer", "rich", "pydantic", "yaml", "jinja2", "platformdirs", "safetensors")
_DISTRIBUTIONS = {
    "yaml": "pyyaml",
    "jinja2": "jinja2",
    "platformdirs": "platformdirs",
    "safetensors": "safetensors",
    "typer": "typer",
    "rich": "rich",
    "pydantic": "pydantic",
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "numpy": "numpy",
}


def _version_of(module: str) -> str | None:
    try:
        return importlib.metadata.version(_DISTRIBUTIONS.get(module, module))
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass
class Check:
    """One diagnostic line."""

    name: str
    status: str  # ok | warn | missing | fail
    detail: str
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {"name": self.name, "status": self.status, "detail": self.detail, "hint": self.hint}


@dataclass
class DoctorReport:
    """Aggregated diagnostics."""

    miniverl_version: str
    checks: list[Check] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)

    @property
    def can_run_core(self) -> bool:
        """``True`` when the lightweight commands will work."""
        return all(c.status != "fail" for c in self.checks if c.name.startswith("dependency:"))

    @property
    def can_train_cpu(self) -> bool:
        """``True`` when toy/CPU training is possible."""
        return bool(self.capabilities.get("torch"))

    @property
    def can_train_gpu(self) -> bool:
        """``True`` when a CUDA device is usable."""
        return bool(self.capabilities.get("cuda_available"))

    @property
    def can_qlora(self) -> bool:
        """``True`` when 4-bit QLoRA training is possible."""
        return bool(
            self.capabilities.get("cuda_available")
            and self.capabilities.get("bitsandbytes")
            and self.capabilities.get("peft")
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view for ``--json``."""
        return {
            "miniverl_version": self.miniverl_version,
            "checks": [c.to_dict() for c in self.checks],
            "capabilities": self.capabilities,
            "verdict": {
                "core_commands": self.can_run_core,
                "cpu_training": self.can_train_cpu,
                "gpu_training": self.can_train_gpu,
                "qlora_4bit": self.can_qlora,
            },
        }


def run_doctor(output_dir: str | Path = "runs") -> DoctorReport:
    """Collect diagnostics about the current environment."""
    report = DoctorReport(miniverl_version=__version__)
    add = report.checks.append

    add(
        Check(
            "miniverl",
            "ok",
            f"{__version__} at {Path(__file__).parent}",
        )
    )
    version = sys.version_info
    supported = (3, 10) <= (version.major, version.minor) <= (3, 13)
    add(
        Check(
            "python",
            "ok" if supported else "warn",
            f"{platform.python_implementation()} {sys.version.split()[0]}",
            None if supported else "miniVERL is tested on CPython 3.10-3.13",
        )
    )
    add(Check("platform", "ok", f"{platform.system()} {platform.release()} ({platform.machine()})"))

    for module in _REQUIRED:
        present = have_module(module)
        add(
            Check(
                f"dependency:{module}",
                "ok" if present else "fail",
                _version_of(module) or ("present" if present else "not installed"),
                None if present else "pip install miniverl",
            )
        )

    report.capabilities["environments"] = available_environments()
    add(Check("environments", "ok", ", ".join(available_environments())))

    for module, extra in (
        ("torch", "train"),
        ("transformers", "train"),
        ("peft", "train"),
        ("accelerate", "train"),
        ("numpy", "train"),
        ("bitsandbytes", "cuda"),
    ):
        present = have_module(module)
        report.capabilities[module] = _version_of(module) if present else None
        add(
            Check(
                f"optional:{module}",
                "ok" if present else "missing",
                _version_of(module) or "not installed",
                None if present else f'pip install "miniverl[{extra}]"',
            )
        )

    if have_module("torch"):
        import torch

        cuda_available = bool(torch.cuda.is_available())
        report.capabilities["cuda_available"] = cuda_available
        report.capabilities["torch_cuda_version"] = torch.version.cuda
        if cuda_available:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            total_gib = props.total_memory / (1024**3)
            report.capabilities.update(
                {
                    "gpu_name": props.name,
                    "gpu_total_memory_gib": round(total_gib, 3),
                    "gpu_capability": f"{props.major}.{props.minor}",
                    "gpu_count": torch.cuda.device_count(),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
            add(
                Check(
                    "cuda",
                    "ok",
                    f"{props.name} | {total_gib:.1f} GiB | capability "
                    f"{props.major}.{props.minor} | torch cuda {torch.version.cuda}",
                )
            )
            add(
                Check(
                    "bf16",
                    "ok" if torch.cuda.is_bf16_supported() else "warn",
                    "supported"
                    if torch.cuda.is_bf16_supported()
                    else "not supported; fp16 will be used",
                )
            )
            fits_16gb = total_gib >= 15.0
            add(
                Check(
                    "consumer-gpu recipe",
                    "ok" if fits_16gb else "warn",
                    (
                        "the 16 GB recipes target a card of this size"
                        if fits_16gb
                        else f"only {total_gib:.1f} GiB visible; the published 16 GB recipe "
                        "may need smaller rollout.max_total_tokens"
                    ),
                    None
                    if fits_16gb
                    else "start from recipes/qwen_consumer_gpu_calc.yaml and lower "
                    "rollout.max_total_tokens / loss.chunk_size",
                )
            )
        else:
            add(
                Check(
                    "cuda",
                    "missing",
                    "torch.cuda.is_available() is False; CPU-only paths still work",
                    "install a CUDA build of torch, e.g. "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu130",
                )
            )
    else:
        report.capabilities["cuda_available"] = False

    target = Path(output_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".miniverl-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add(Check("output directory", "ok", f"{target.resolve()} is writable"))
    except OSError as exc:
        add(
            Check(
                "output directory",
                "fail",
                f"{target} is not writable: {exc}",
                "pass --output to a writable location",
            )
        )
    return report
