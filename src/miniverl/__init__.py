"""miniVERL -- auditable online post-training on one GPU.

The top-level package is intentionally lightweight: importing ``miniverl`` must
not import :mod:`torch`, :mod:`transformers` or any CUDA runtime.  Heavy
dependencies are imported lazily at the point of use so that the CLI's
inspection commands (``doctor``, ``validate``, ``inspect``, ``report``,
``cache``) work from a bare ``pip install miniverl``.

Public API
----------
>>> from miniverl.config import RunConfig
>>> from miniverl.trainer import OPDTrainer
"""

from __future__ import annotations

__version__ = "0.10.0.dev0"

__all__ = ["__version__"]
