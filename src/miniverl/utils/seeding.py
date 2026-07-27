"""Deterministic seeding and RNG-state capture.

Resume correctness depends on restoring *all* the randomness, not just the
optimizer: Python's `random`, torch's CPU generator and every CUDA generator
are captured together and restored together.
"""

from __future__ import annotations

import base64
import os
import random
from dataclasses import dataclass, field
from typing import Any

from miniverl.utils.lazy import have_module

__all__ = ["seed_everything", "RngSnapshot", "capture_rng", "restore_rng", "derive_seed"]


def derive_seed(*parts: object) -> int:
    """Stable 63-bit seed derived from arbitrary parts (no salted ``hash``)."""
    import hashlib

    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") >> 1


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch, and optionally request determinism.

    ``deterministic`` sets the cuBLAS workspace variable and asks torch for
    deterministic kernels in warn-only mode: a kernel without a deterministic
    implementation degrades to a warning rather than crashing a long run.
    """
    random.seed(seed)
    if have_module("numpy"):
        import numpy as np

        np.random.seed(seed % (2**32))
    if not have_module("torch"):
        return
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError):  # pragma: no cover - old torch
            pass


@dataclass
class RngSnapshot:
    """Serializable snapshot of every RNG miniVERL touches."""

    python_state: str
    torch_state: str | None = None
    cuda_states: list[str] = field(default_factory=list)
    numpy_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "python_state": self.python_state,
            "torch_state": self.torch_state,
            "cuda_states": list(self.cuda_states),
            "numpy_state": self.numpy_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RngSnapshot:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            python_state=payload["python_state"],
            torch_state=payload.get("torch_state"),
            cuda_states=list(payload.get("cuda_states") or []),
            numpy_state=payload.get("numpy_state"),
        )


def _encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def capture_rng() -> RngSnapshot:
    """Capture the current RNG states."""
    # random.getstate() is a tuple of ints; JSON-encoding it directly keeps the
    # snapshot pickle-free, which matters because checkpoints are data files.
    import json

    snapshot = RngSnapshot(python_state=json.dumps(random.getstate(), default=list))
    if have_module("torch"):
        import torch

        snapshot.torch_state = _encode(torch.get_rng_state().numpy().tobytes())
        if torch.cuda.is_available():
            snapshot.cuda_states = [
                _encode(state.numpy().tobytes()) for state in torch.cuda.get_rng_state_all()
            ]
    if have_module("numpy"):
        import numpy as np

        state = np.random.get_state()
        snapshot.numpy_state = json.dumps(
            [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]
        )
    return snapshot


def restore_rng(snapshot: RngSnapshot) -> None:
    """Restore RNG states captured by :func:`capture_rng`."""
    import json

    state = json.loads(snapshot.python_state)
    random.setstate((state[0], tuple(state[1]), state[2]))
    if snapshot.torch_state and have_module("torch"):
        import numpy as np
        import torch

        buffer = np.frombuffer(_decode(snapshot.torch_state), dtype=np.uint8).copy()
        torch.set_rng_state(torch.from_numpy(buffer))
        if snapshot.cuda_states and torch.cuda.is_available():
            tensors = [
                torch.from_numpy(np.frombuffer(_decode(s), dtype=np.uint8).copy())
                for s in snapshot.cuda_states
            ]
            if len(tensors) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(tensors)
    if snapshot.numpy_state and have_module("numpy"):
        import numpy as np

        payload = json.loads(snapshot.numpy_state)
        np.random.set_state(
            (payload[0], np.array(payload[1], dtype=np.uint32), payload[2], payload[3], payload[4])
        )
