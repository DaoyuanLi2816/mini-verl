"""Standalone evaluation of a finished run.

``miniverl eval --run runs/<id>`` rebuilds the trainer from the run's own
``config.resolved.yaml``, restores the latest checkpoint and re-evaluates
deterministically.  Rebuilding from the *resolved* config -- not the original
recipe -- is what makes the evaluation reproduce the run it is evaluating: any
``auto`` decision was already frozen when the run started.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniverl.config.models import RunConfig
from miniverl.errors import RunNotFoundError
from miniverl.utils.logging import get_logger
from miniverl.utils.runs import RunPaths, write_json

__all__ = ["evaluate_run"]

logger = get_logger("eval")


def evaluate_run(
    run_dir: str | Path,
    *,
    split: str | None = None,
    tasks: int | None = None,
    checkpoint: str | Path | None = None,
    out: str | Path | None = None,
    tag: str = "standalone",
) -> dict[str, Any]:
    """Re-evaluate a finished run and write the result next to it."""
    from miniverl.trainer import OPDTrainer
    from miniverl.training.checkpoint import latest_checkpoint, load_checkpoint

    paths = RunPaths.open(run_dir)
    if not paths.config_resolved.is_file():
        raise RunNotFoundError(
            f"{paths.root} has no config.resolved.yaml",
            hint="only runs created by `miniverl train` can be re-evaluated",
        )
    config = RunConfig.from_yaml(paths.config_resolved)
    if tasks is not None:
        config = config.model_copy(update={"eval": config.eval.model_copy(update={"tasks": tasks})})

    target_checkpoint = Path(checkpoint) if checkpoint else latest_checkpoint(paths.checkpoints)
    trainer = OPDTrainer.from_config(
        config,
        output_dir=paths.root.parent,
        run_id=paths.root.name,
        write_artifacts=False,
    )
    try:
        if target_checkpoint is not None and Path(target_checkpoint).is_dir():
            load_checkpoint(
                target_checkpoint,
                backend=trainer.student,
                optimizer=trainer.optimizer,
                device=trainer.student.device,
                include_rng=False,
            )
            logger.info("restored checkpoint %s", target_checkpoint)
        else:
            logger.warning(
                "no checkpoint found under %s; evaluating the freshly initialized student",
                paths.checkpoints,
            )
        payload = trainer.evaluate(split=split, tag=tag, write=True)
        payload["checkpoint"] = str(target_checkpoint) if target_checkpoint else None
        payload["run_dir"] = str(paths.root)
        destination = Path(out) if out else paths.root / f"eval.{tag}.json"
        write_json(destination, payload)
        payload["written_to"] = str(destination)
        return payload
    finally:
        trainer.close()
