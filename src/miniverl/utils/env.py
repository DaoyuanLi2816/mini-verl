"""Environment capture for run manifests.

Deliberately excluded, because a run directory is meant to be shareable:
hostname, username, home directory, absolute paths outside the run, and every
environment variable except a short allowlist of ones that change numerics.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from miniverl.utils.lazy import have_module

__all__ = ["collect_environment", "gpu_info", "package_versions", "git_commit", "TRACKED_ENV_VARS"]

#: Environment variables that can change numerical results.  Values are
#: recorded; nothing else from the environment is.
TRACKED_ENV_VARS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)

_PACKAGES = (
    "miniverl",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "safetensors",
    "pydantic",
    "typer",
    "rich",
    "numpy",
)


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages that affect results."""
    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> dict[str, Any]:
    """CUDA device description, or a clear 'not available' record."""
    if not have_module("torch"):
        return {"available": False, "reason": "torch is not installed"}
    import torch

    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "torch.cuda.is_available() is False",
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    info: dict[str, Any] = {
        "available": True,
        "device_index": index,
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "total_memory_gib": round(props.total_memory / (1024**3), 3),
        "capability": f"{props.major}.{props.minor}",
        "multi_processor_count": int(props.multi_processor_count),
        "device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    try:
        info["driver_version"] = _nvidia_driver_version()
    except Exception:  # pragma: no cover - purely informational
        info["driver_version"] = None
    return info


def _nvidia_driver_version() -> str | None:
    """Read the driver version through torch's own NVML binding, if present."""
    import torch

    handle = getattr(torch.cuda, "_get_nvml_device_index", None)
    if handle is None:  # pragma: no cover - depends on torch build
        return None
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        return version.decode() if isinstance(version, bytes) else str(version)
    except Exception:  # pragma: no cover - pynvml is optional
        return None


def git_commit(start: Path | None = None) -> str | None:
    """Resolve the current git commit by reading ``.git`` -- no subprocess.

    Returns ``None`` outside a git checkout (for example when installed from a
    wheel), which the manifest records honestly rather than faking.
    """
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in [here, *here.parents]:
        git_dir = candidate / ".git"
        if git_dir.is_file():
            content = git_dir.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                git_dir = (candidate / content[len("gitdir:") :].strip()).resolve()
            else:  # pragma: no cover - malformed .git file
                continue
        if not git_dir.is_dir():
            continue
        head_path = git_dir / "HEAD"
        if not head_path.is_file():
            continue
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head[4:].strip()
            ref_path = git_dir / ref
            if ref_path.is_file():
                return ref_path.read_text(encoding="utf-8").strip()
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or " " not in line:
                        continue
                    sha, name = line.split(" ", 1)
                    if name.strip() == ref:
                        return sha.strip()
            return None
        return head or None
    return None


def collect_environment() -> dict[str, Any]:
    """Reproducibility-relevant machine description, free of personal data."""
    import os

    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "processor_family": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "gpu": gpu_info(),
        "tracked_env_vars": {
            name: os.environ.get(name) for name in TRACKED_ENV_VARS if os.environ.get(name)
        },
        "git_commit": git_commit(),
    }
