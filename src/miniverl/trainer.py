"""Public trainer entry point.

``from miniverl.trainer import OPDTrainer`` is the documented import path; the
implementation lives in :mod:`miniverl.training.trainer`.  Importing this module
pulls in torch, so lightweight CLI paths do not touch it.
"""

from __future__ import annotations

from miniverl.training.trainer import OPDTrainer, TrainerState, TrainResult, TrainSample

__all__ = ["OPDTrainer", "TrainerState", "TrainResult", "TrainSample"]
