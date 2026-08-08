"""miniVERL command line interface.

A thin layer: every command parses arguments, calls one library function, and
renders the result.  No training logic lives here.

Only the heavy commands import torch, and they do it inside the command body,
so ``miniverl --help``, ``doctor``, ``validate``, ``inspect``, ``report`` and
``cache`` all work from a bare ``pip install miniverl``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from miniverl import __version__
from miniverl.errors import ConfigError, MiniVerlError
from miniverl.utils.runs import make_run_id

app = typer.Typer(
    name="miniverl",
    help=(
        "On-policy distillation for tool-using agents on one GPU.\n\n"
        "A readable single-CUDA-GPU post-training lab for exact and budgeted "
        "on-policy distillation."
    ),
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
)
cache_app = typer.Typer(help="Inspect and validate a teacher-target cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")
bridge_app = typer.Typer(
    help="Inspect a pinned, exported verl scale-out bundle.", no_args_is_help=True
)
app.add_typer(bridge_app, name="bridge")

console = Console()
err_console = Console(stderr=True)

_STATUS_STYLE = {"ok": "green", "warn": "yellow", "missing": "yellow", "fail": "red"}


def _emit_json(payload: Any) -> None:
    try:
        serialized = json.dumps(payload, default=str, allow_nan=False)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        from miniverl.errors import SerializationError

        _fail(SerializationError(f"command result is not finite JSON: {exc}"))
        return
    console.print_json(serialized)


def _esc(value: object) -> str:
    """Escape dynamic text before it reaches Rich.

    Rich treats square brackets as markup, so unescaped values silently lose
    content: the hint ``pip install "miniverl[train]"`` would print as
    ``pip install "miniverl"`` -- the wrong command. Every dynamic string this
    module prints goes through here.
    """
    return escape(str(value))


def _require_training_stack(purpose: str) -> None:
    """Fail with an install command if the training extra is not present.

    Called before any heavy import so a bare ``pip install miniverl`` produces
    the exact command to run rather than a bare ``ModuleNotFoundError``.
    """
    from miniverl.errors import MissingDependencyError
    from miniverl.utils.lazy import have_module

    for module in ("torch", "transformers", "peft"):
        if not have_module(module):
            raise MissingDependencyError(module, "train", purpose)


def _fail(exc: Exception, *, code: int = 1) -> None:
    """Print an actionable error and exit non-zero."""
    if isinstance(exc, ModuleNotFoundError):
        from miniverl.errors import MissingDependencyError

        exc = MissingDependencyError(exc.name or "a dependency", "train", "This command")
    if isinstance(exc, MiniVerlError):
        err_console.print(f"[red]error[/red] {_esc(exc.message)}")
        if exc.hint:
            err_console.print(f"[yellow]hint[/yellow]  {_esc(exc.hint)}")
    else:
        err_console.print(f"[red]error[/red] {_esc(exc)}")
    raise typer.Exit(code)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"miniverl {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print the miniVERL version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
        envvar="MINIVERL_LOG_LEVEL",
    ),
) -> None:
    """miniVERL: on-policy distillation for tool-using agents on one GPU."""
    from miniverl.utils.logging import configure_logging

    configure_logging(log_level)


# ---------------------------------------------------------------- doctor


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    output_dir: Path = typer.Option(
        Path("runs"), "--output", help="Directory to test for writability."
    ),
) -> None:
    """Report what this machine can run, and what to install for the rest."""
    from miniverl.doctor import run_doctor

    try:
        report = run_doctor(output_dir)
    except Exception as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(report.to_dict())
        return

    table = Table(title=f"miniVERL {report.miniverl_version} environment", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail")
    for check in report.checks:
        table.add_row(
            _esc(check.name),
            f"[{_STATUS_STYLE.get(check.status, 'white')}]{check.status}[/]",
            _esc(check.detail),
        )
    console.print(table)

    verdict = report.to_dict()["verdict"]
    console.print()
    for label, key, command in (
        ("core commands (doctor/validate/inspect/report/cache)", "core_commands", None),
        (
            "CPU + toy training (demo, recipes/toy_cpu.yaml)",
            "cpu_training",
            'pip install "miniverl[train]"',
        ),
        ("GPU training", "gpu_training", "install a CUDA build of torch"),
        ("4-bit QLoRA", "qlora_4bit", 'pip install "miniverl[train,cuda]"'),
    ):
        ready = verdict[key]
        # Pad to the width of the longer word so the labels line up in a column.
        mark = "[green]yes[/green]" if ready else "[yellow]no [/yellow]"
        suffix = "" if ready or not command else f"  ->  {command}"
        console.print(f"  {mark}  {_esc(label)}{_esc(suffix)}")

    hints = [c for c in report.checks if c.hint and c.status in {"fail", "missing", "warn"}]
    if hints:
        console.print()
        console.print("[bold]suggestions[/bold]")
        for check in hints:
            console.print(f"  - {_esc(check.name)}: {_esc(check.hint)}")


# -------------------------------------------------------------- validate


@app.command()
def validate(
    recipe: Path = typer.Argument(..., help="Path to a recipe YAML file."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate a recipe without downloading models or allocating memory."""
    from miniverl.config import RunConfig
    from miniverl.environments.registry import make_environment

    try:
        config = RunConfig.from_yaml(recipe)
    except ValidationError as exc:
        if as_json:
            _emit_json({"valid": False, "path": str(recipe), "errors": exc.errors()})
            raise typer.Exit(1) from None
        err_console.print(f"[red]invalid recipe[/red] {_esc(recipe)}")
        for error in exc.errors():
            location = ".".join(str(p) for p in error["loc"])
            err_console.print(f"  [red]{_esc(location or '<root>')}[/red]: {_esc(error['msg'])}")
        err_console.print(
            "\n[yellow]hint[/yellow]  compare against recipes/toy_cpu.yaml, which is "
            "validated in CI"
        )
        raise typer.Exit(1) from None
    except MiniVerlError as exc:
        if as_json:
            _emit_json({"valid": False, "path": str(recipe), "errors": [exc.message]})
            raise typer.Exit(1) from None
        _fail(exc)
        return

    warnings: list[str] = []
    try:
        environment = make_environment(config.environment.name, **config.environment.params)
        if config.models.teacher.mode.value == "privileged_context" and not hasattr(
            environment, "privileged_context"
        ):
            warnings.append("environment provides no privileged context")
    except MiniVerlError as exc:
        if as_json:
            _emit_json({"valid": False, "path": str(recipe), "errors": [exc.message]})
            raise typer.Exit(1) from None
        _fail(exc)
        return

    steps_per_cycle = max(
        1,
        (config.train.rollouts_per_cycle + config.train.gradient_accumulation_steps - 1)
        // config.train.gradient_accumulation_steps,
    )
    if config.run.mode.value == "opd" and config.train.opd_freshness.value == "replay":
        warnings.append(
            f"opd_freshness=replay permits {steps_per_cycle} optimizer step(s) per "
            "rollout batch; this is online distillation with replay, not genuine OPD"
        )
    if config.models.backend.value == "hf" and not config.models.student.revision:
        warnings.append("models.student.revision is unpinned; the manifest will record 'unpinned'")
    if config.models.backend.value == "hf" and not config.models.teacher.revision:
        warnings.append("models.teacher.revision is unpinned")

    payload = {
        "valid": True,
        "path": str(recipe),
        "run_name": config.run.name,
        "mode": config.run.mode.value,
        "is_on_policy": config.is_on_policy,
        "opd_freshness": (
            config.train.opd_freshness.value if config.run.mode.value == "opd" else None
        ),
        "backend": config.models.backend.value,
        "student": config.models.student.model_id,
        "teacher": config.models.teacher.model_id,
        "environment": config.environment.name,
        "difficulty": config.environment.difficulty,
        "objective": (
            "sft_cross_entropy"
            if config.run.mode.value == "sft"
            else (
                "online_distillation_with_replay"
                if config.run.mode.value == "opd" and not config.is_on_policy
                else config.run.mode.value
            )
        ),
        "loss_mode": config.loss.mode.value if config.run.mode.value != "sft" else None,
        "divergence": config.loss.divergence.value if config.run.mode.value != "sft" else None,
        "top_k": config.loss.top_k if config.run.mode.value != "sft" else None,
        "selector": config.selection.selector.value,
        "memory_strategy": config.memory.strategy.value,
        "cycles": config.train.cycles,
        "sft_warmup_cycles": config.train.sft_warmup_cycles,
        "optimizer_steps_per_cycle": steps_per_cycle,
        "planned_optimizer_steps": steps_per_cycle
        * (config.train.cycles + config.train.sft_warmup_cycles),
        "eval_tasks": config.effective_eval_tasks,
        "seed": config.run.seed,
        "warnings": warnings,
    }
    if as_json:
        _emit_json(payload)
        return
    console.print(f"[green]valid[/green] {_esc(recipe)}")
    table = Table(show_header=False, box=None, pad_edge=False)
    for key, value in payload.items():
        if key in {"valid", "path", "warnings"}:
            continue
        table.add_row(f"[dim]{key}[/dim]", _esc(value))
    console.print(table)
    for warning in warnings:
        console.print(f"[yellow]warning[/yellow] {_esc(warning)}")


# ------------------------------------------------------------------ demo


@app.command()
def demo(
    output: Path = typer.Option(Path("runs/demo"), "--output", help="Run directory to create."),
    fast: bool = typer.Option(False, "--fast", help="Shrink every budget (CI smoke test)."),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Explicitly replace the whole existing demo run directory.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    report_html: bool = typer.Option(True, "--report/--no-report", help="Also render report.html."),
) -> None:
    """Run the embedded no-network toy pipeline end to end."""
    try:
        _require_training_stack("miniverl demo")
        from miniverl.demo import demo_config
        from miniverl.trainer import OPDTrainer
    except (MiniVerlError, ModuleNotFoundError) as exc:
        _fail(exc)
        return

    target = Path(output)
    try:
        config = demo_config(fast=fast, output_dir=target.parent)
        if overwrite and target.exists() and not as_json:
            console.print(
                f"[yellow]overwrite[/yellow] replacing whole run directory {_esc(target)}"
            )
        with OPDTrainer.from_config(
            config,
            output_dir=target.parent,
            run_id=target.name,
            overwrite=overwrite,
        ) as trainer:
            result = trainer.train()
            paths = trainer.paths
            report_path: Path | None = None
            if report_html:
                from miniverl.reporting import ReportData, write_markdown, write_report

                report_path = write_report(paths.root, paths.report_html)
                write_markdown(ReportData.from_run(paths.root), paths.summary_md)
            artifacts = _artifact_listing(paths.root)
    except MiniVerlError as exc:
        _fail(exc)
        return

    payload = {
        **result.to_dict(),
        "artifacts": artifacts,
        "report": str(report_path) if report_path else None,
    }
    if as_json:
        _emit_json(payload)
        return
    console.print()
    console.print(f"[bold green]demo complete[/bold green]  {_esc(paths.root)}")
    table = Table(show_header=False, box=None)
    baseline = (result.baseline_eval or {}).get("success_rate")
    final = (result.eval or {}).get("success_rate")
    table.add_row("mode", f"{result.mode} (genuine on-policy distillation)")
    table.add_row("optimizer steps", str(result.global_step))
    table.add_row("parameter version", str(result.parameter_version))
    table.add_row("rollout iterations", str(result.cycles_completed))
    table.add_row("wall clock", f"{result.duration_seconds:.1f} s")
    provenance = _provenance_summary(paths.trajectories)
    if provenance:
        table.add_row("token provenance", provenance)
    compression = _cache_summary(paths.teacher_cache)
    if compression:
        table.add_row("teacher cache", compression)
    table.add_row(
        "task success",
        f"{_fmt_pct(baseline)} -> {_fmt_pct(final)} (greedy, held-out eval split)",
    )
    console.print(table)
    console.print()
    for line in (
        "This demo proves the [bold]machinery[/bold], not capability.",
        "At this size the toy student learns the tool-call format and not the",
        "arithmetic, so 0% here is the expected outcome, not a failure.",
        "For a CPU run that does learn (measured 0.0% -> 91.7% in 192 s):",
        "  [bold]miniverl train recipes/toy_cpu.yaml[/bold]",
    ):
        console.print(line)
    console.print()
    console.print("[bold]artifacts[/bold]")
    for name, size in artifacts.items():
        console.print(f"  {_esc(name)}  [dim]{_esc(size)}[/dim]")
    console.print()
    console.print("[bold]next[/bold]")
    console.print(f"  miniverl inspect {_esc(paths.trajectories)}")
    console.print(f"  miniverl cache stats {_esc(paths.teacher_cache)}")
    console.print(f"  miniverl report {_esc(paths.root)} --out {_esc(paths.report_html)}")


def _provenance_summary(path: Path) -> str:
    """One line summarizing which tokens could enter the loss."""
    try:
        from miniverl.inspection import summarize_file

        summary = summarize_file(path, limit=0)
    except (MiniVerlError, OSError):
        # A cosmetic summary line must never turn a completed run into a failure.
        return ""
    if not summary.tokens:
        return ""
    return (
        f"{summary.model_tokens} of {summary.tokens} tokens trainable "
        f"({summary.model_token_fraction * 100:.0f}%); "
        f"{summary.context_tokens} are context and can never be a target"
    )


def _cache_summary(path: Path) -> str:
    """One line summarizing the teacher-target cache."""
    try:
        from miniverl.cache.stats import compute_stats

        stats = compute_stats(path, verify_checksums=False)
    except (MiniVerlError, OSError):
        return ""
    if not stats.get("selected_positions"):
        return ""
    return (
        f"{stats['selected_positions']} scored positions, "
        f"{stats['actual_bytes'] / 1024:.1f} KiB on disk, "
        f"{stats['compression_ratio']:.1f}x smaller than a dense fp16 dump"
    )


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _artifact_listing(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            out[rel] = f"{path.stat().st_size} B"
    return out


# ----------------------------------------------------- prepare-offline-kd


@app.command("prepare-offline-kd")
def prepare_offline_kd_command(
    recipe: Path = typer.Option(..., "--recipe", help="Frozen-student offline-KD recipe."),
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", help="Exact shared cold-start checkpoint directory."
    ),
    out: Path = typer.Option(..., "--out", help="New immutable dataset bundle directory."),
    offline: bool = typer.Option(
        False, "--offline", help="Refuse network access; use only cached model files."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and print the collection plan without loading models or writing files.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Collect one frozen-student trajectory/teacher-target dataset."""
    from miniverl.config import OfflineKDTrajectorySource, RunConfig, TrainingMode

    try:
        config = RunConfig.from_yaml(recipe)
        if config.run.mode is not TrainingMode.OFFLINE_KD:
            raise ConfigError("prepare-offline-kd requires run.mode=offline_kd")
        if config.offline_kd.trajectory_source is not OfflineKDTrajectorySource.FROZEN_STUDENT:
            raise ConfigError(
                "prepare-offline-kd requires offline_kd.trajectory_source=frozen_student"
            )
        if not checkpoint.is_dir():
            raise ConfigError(f"checkpoint directory not found: {checkpoint}")
        if out.exists():
            raise ConfigError(
                f"output already exists: {out}",
                hint="choose a new directory; frozen datasets are never overwritten",
            )
    except (ValidationError, MiniVerlError) as exc:
        if isinstance(exc, MiniVerlError):
            _fail(exc)
        err_console.print(f"[red]invalid recipe[/red] {_esc(recipe)}\n{_esc(exc)}")
        raise typer.Exit(1) from None

    plan = {
        "dry_run": dry_run,
        "recipe": str(recipe),
        "checkpoint": str(checkpoint),
        "out": str(out),
        "offline": offline,
        "trajectory_source": config.offline_kd.trajectory_source.value,
        "collection_seed": config.offline_kd.collection_seed,
        "collection_tasks": (config.offline_kd.collection_tasks or config.train.rollouts_per_cycle),
        "student": config.models.student.model_id,
        "student_revision": config.models.student.revision,
        "teacher": config.models.teacher.model_id,
        "teacher_revision": config.models.teacher.revision,
    }
    if dry_run:
        if as_json:
            _emit_json(plan)
        else:
            console.print(f"[green]dry run ok[/green] {_esc(recipe)}")
            for key, value in plan.items():
                console.print(f"  [dim]{key}[/dim] {_esc(value)}")
            console.print("\nNo models were loaded, files written, or downloads attempted.")
        return

    try:
        _require_training_stack("miniverl prepare-offline-kd")
        from miniverl.trainer import OPDTrainer
        from miniverl.training.checkpoint import load_checkpoint, validate_checkpoint

        validated = validate_checkpoint(checkpoint)
        with OPDTrainer.from_config(
            config,
            output_dir=out.parent,
            run_id=out.name,
            local_files_only=offline,
        ) as trainer:
            load_checkpoint(
                checkpoint,
                backend=trainer.student,
                optimizer=None,
                device=trainer.student.device,
                include_optimizer=False,
                include_rng=False,
                expected_identity=(trainer._checkpoint_identity() if validated.identity else None),
            )
            trainer.set_offline_collection_checkpoint_digest(validated.content_digest)
            summary = trainer.prepare_offline_dataset()
            destination = trainer.paths.root
    except (MiniVerlError, ModuleNotFoundError) as exc:
        _fail(exc)
        return

    payload = {**plan, "written": str(destination), **summary}
    if as_json:
        _emit_json(payload)
        return
    console.print(f"[green]offline dataset prepared[/green] {_esc(destination)}")
    console.print(f"  digest {_esc(summary['dataset_digest'])}")
    console.print(f"  trajectories {_esc(summary['trajectories'])}")


# --------------------------------------------------------- qualify-teacher


@app.command("qualify-teacher")
def qualify_teacher_command(
    recipe: Path = typer.Option(..., "--recipe", help="Recipe defining the frozen teacher."),
    candidate: str = typer.Option(..., "--candidate", help="Preregistered candidate id."),
    out: Path = typer.Option(..., "--out", help="New immutable gate-result directory."),
    tasks: Optional[int] = typer.Option(None, "--tasks", help="Eval-task count override."),
    offline: bool = typer.Option(
        False, "--offline", help="Refuse network access; use only cached model files."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Evaluate one RecoveryBench teacher candidate on eval only."""
    from miniverl.config import RunConfig

    try:
        config = RunConfig.from_yaml(recipe)
        if config.environment.name != "sqlite_recovery":
            raise ConfigError("qualify-teacher requires environment.name=sqlite_recovery")
        _require_training_stack("miniverl qualify-teacher")
        from miniverl.evaluation.teacher_gate import evaluate_teacher_candidate

        result = evaluate_teacher_candidate(
            config,
            candidate_id=candidate,
            out=out,
            tasks=tasks,
            split="eval",
            local_files_only=offline,
        )
    except (ValidationError, MiniVerlError, ModuleNotFoundError) as exc:
        if isinstance(exc, ValidationError):
            err_console.print(f"[red]invalid recipe[/red] {_esc(recipe)}\n{_esc(exc)}")
            raise typer.Exit(1) from None
        _fail(exc)
        return
    if as_json:
        _emit_json(result)
        return
    status = "passed" if result["gate"]["passed"] else "failed"
    style = "green" if result["gate"]["passed"] else "yellow"
    console.print(f"[{style}]teacher gate {status}[/{style}] {_esc(candidate)}")
    console.print(f"  result {_esc(out / 'result.json')}")


# ----------------------------------------------------------------- train


@app.command()
def pilot(
    recipe: Path = typer.Argument(..., help="Alignment recipe containing bounded pilot evidence."),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional JSON output path."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Recommend a method from explicit pilot evidence without loading a model."""
    from miniverl.alignment import PilotEvidence, recommend_alignment_method
    from miniverl.config import RunConfig
    from miniverl.utils.runs import write_json_atomic

    try:
        config = RunConfig.from_yaml(recipe)
        if config.alignment is None:
            raise ConfigError("miniverl pilot requires a recipe with an alignment section")
        evidence = config.alignment.pilot or PilotEvidence()
        result = recommend_alignment_method(evidence)
        payload = result.model_dump(mode="json")
        if out is not None:
            write_json_atomic(out, payload)
    except (ValidationError, MiniVerlError) as exc:
        if isinstance(exc, MiniVerlError):
            _fail(exc)
        err_console.print(f"[red]invalid recipe[/red] {_esc(recipe)}\n{_esc(exc)}")
        raise typer.Exit(1) from None
    if as_json:
        _emit_json(payload)
        return
    console.print(f"[bold]recommendation[/bold] {_esc(payload['recommendation'])}")
    for reason in payload["reasons"]:
        console.print(f"  - {_esc(reason)}")
    if out is not None:
        console.print(f"  evidence {_esc(out)}")


@app.command()
def align(
    recipe: Path = typer.Argument(..., help="Path to a post-SFT alignment recipe YAML file."),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Parent directory for the run (default: run.output_dir)."
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run id."),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Explicitly replace the whole target run directory.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and print all alignment stages without loading models.",
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Refuse network access; use only cached model files."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run base -> SFT checkpoint -> teacher -> alignment -> eval -> card."""
    from miniverl.alignment import build_alignment_stage_plan
    from miniverl.config import RunConfig

    try:
        config = RunConfig.from_yaml(recipe)
        if config.alignment is None:
            raise ConfigError("miniverl align requires a recipe with an alignment section")
    except (ValidationError, MiniVerlError) as exc:
        if isinstance(exc, MiniVerlError):
            _fail(exc)
        err_console.print(f"[red]invalid recipe[/red] {_esc(recipe)}\n{_esc(exc)}")
        raise typer.Exit(1) from None

    workflow = build_alignment_stage_plan(
        config.alignment,
        sft_warmup_cycles=config.train.sft_warmup_cycles,
    )
    if dry_run:
        payload = {
            "dry_run": True,
            "recipe": str(recipe),
            "method": config.alignment.method.value,
            "workflow": workflow,
            "backend": config.models.backend.value,
            "downloads_required": config.models.backend.value == "hf",
            "output_dir": str(output or config.run.output_dir),
        }
        if as_json:
            _emit_json(payload)
        else:
            console.print(f"[green]alignment dry run ok[/green] {_esc(recipe)}")
            for stage in workflow["stages"]:
                console.print(f"  - {_esc(stage['name'])}")
            console.print("\nNo models were loaded and nothing was downloaded.")
        return

    try:
        _require_training_stack("miniverl align")
        from miniverl.alignment import run_alignment

        payload = run_alignment(
            config,
            output_dir=output,
            run_id=run_id,
            local_files_only=offline,
            overwrite=overwrite,
        )
    except (MiniVerlError, ModuleNotFoundError) as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(payload)
        return
    console.print(f"[bold green]alignment complete[/bold green] {_esc(payload['run_dir'])}")
    console.print(f"  card {_esc(Path(str(payload['run_dir'])) / 'alignment-card.md')}")


# ----------------------------------------------------------------- train


@app.command()
def train(
    recipe: Path = typer.Argument(..., help="Path to a recipe YAML file."),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Parent directory for the run (default: run.output_dir)."
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run id."),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        help="Continue an existing run from its highest valid checkpoint.",
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume-from",
        help="Continue an existing run from this exact checkpoint directory.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Explicitly replace the whole target run directory.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate, resolve and print the plan without loading models or downloading anything.",
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Refuse network access; use only already-cached model files."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    make_report: bool = typer.Option(
        True, "--report/--no-report", help="Render report.html when the run finishes."
    ),
) -> None:
    """Train a student with SFT, offline KD or on-policy distillation."""
    from miniverl.config import RunConfig

    try:
        config = RunConfig.from_yaml(recipe)
    except (ValidationError, MiniVerlError) as exc:
        if isinstance(exc, MiniVerlError):
            _fail(exc)
        err_console.print(f"[red]invalid recipe[/red] {_esc(recipe)}\n{_esc(exc)}")
        raise typer.Exit(1) from None

    if resume is not None and resume_from is not None:
        _fail(ConfigError("--resume and --resume-from are mutually exclusive"))
        return
    if overwrite and (resume is not None or resume_from is not None):
        _fail(ConfigError("--overwrite cannot be combined with --resume or --resume-from"))
        return
    if (resume is not None or resume_from is not None) and (
        run_id is not None or output is not None
    ):
        _fail(
            ConfigError(
                "--resume/--resume-from cannot be combined with --run-id or --output",
                hint="the existing run directory already determines its id and output root",
            )
        )
        return

    if dry_run:
        steps_per_cycle = max(
            1,
            (config.train.rollouts_per_cycle + config.train.gradient_accumulation_steps - 1)
            // config.train.gradient_accumulation_steps,
        )
        plan = {
            "dry_run": True,
            "recipe": str(recipe),
            "run_name": config.run.name,
            "mode": config.run.mode.value,
            "backend": config.models.backend.value,
            "student": config.models.student.model_id,
            "student_revision": config.models.student.revision,
            "teacher": config.models.teacher.model_id,
            "teacher_revision": config.models.teacher.revision,
            "downloads_required": config.models.backend.value == "hf",
            "planned_optimizer_steps": steps_per_cycle
            * (config.train.cycles + config.train.sft_warmup_cycles),
            "planned_rollouts": config.train.rollouts_per_cycle * config.train.cycles,
            "output_dir": str(output or config.run.output_dir),
            "resume": str(resume) if resume is not None else None,
            "resume_from": str(resume_from) if resume_from is not None else None,
            "overwrite": overwrite,
            "resolved_config": json.loads(json.dumps(config.model_dump(mode="json"), default=str)),
        }
        if as_json:
            _emit_json(plan)
        else:
            console.print(f"[green]dry run ok[/green] {_esc(recipe)}")
            for key in (
                "mode",
                "backend",
                "student",
                "teacher",
                "downloads_required",
                "planned_optimizer_steps",
                "planned_rollouts",
                "output_dir",
            ):
                console.print(f"  [dim]{key}[/dim] {_esc(plan[key])}")
            console.print("\nNo models were loaded and nothing was downloaded.")
        return

    try:
        _require_training_stack("miniverl train")
        from miniverl.trainer import OPDTrainer

        explicit_id = run_id or config.run.run_id
        if overwrite and explicit_id and not as_json:
            target = Path(output or config.run.output_dir) / make_run_id(
                config.run.name,
                explicit=explicit_id,
            )
            if target.exists():
                console.print(
                    f"[yellow]overwrite[/yellow] replacing whole run directory {_esc(target)}"
                )
        with OPDTrainer.from_config(
            config,
            output_dir=output,
            run_id=run_id,
            local_files_only=offline,
            overwrite=overwrite,
            resume=resume,
            resume_from=resume_from,
        ) as trainer:
            result = trainer.train()
            paths = trainer.paths
            report_path: Path | None = None
            if make_report and config.report.enabled:
                from miniverl.reporting import ReportData, write_markdown, write_report

                report_path = write_report(
                    paths.root,
                    paths.report_html,
                    max_trajectories=config.report.max_trajectories,
                    max_tokens=config.report.max_tokens_per_trajectory,
                )
                write_markdown(ReportData.from_run(paths.root), paths.summary_md)
    except (MiniVerlError, ModuleNotFoundError) as exc:
        _fail(exc)
        return

    payload = {**result.to_dict(), "report": str(report_path) if report_path else None}
    if as_json:
        _emit_json(payload)
        return
    console.print()
    console.print(f"[bold green]run complete[/bold green] {_esc(paths.root)}")
    baseline = (result.baseline_eval or {}).get("success_rate")
    final = (result.eval or {}).get("success_rate")
    console.print(
        f"  task success {_fmt_pct(baseline)} -> {_fmt_pct(final)} "
        f"in {result.global_step} optimizer steps ({result.duration_seconds:.1f} s)"
    )
    console.print(f"  eval  miniverl eval --run {_esc(paths.root)}")
    console.print(f"  report {_esc(report_path or paths.report_html)}")


# ------------------------------------------------------------------- eval


@app.command("eval")
def eval_command(
    run: Path = typer.Option(..., "--run", help="Run directory produced by `miniverl train`."),
    split: Optional[str] = typer.Option(
        None, "--split", help="Split to evaluate: train, eval or test."
    ),
    tasks: Optional[int] = typer.Option(None, "--tasks", help="Limit the number of tasks."),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", help="Specific checkpoint directory (default: the latest)."
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write the eval JSON."),
    tag: str = typer.Option("standalone", "--tag", help="Label recorded with the results."),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Refuse network access; use only local or already-cached model and adapter files.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Re-evaluate a finished run deterministically."""
    try:
        _require_training_stack("miniverl eval")
        from miniverl.evaluation.evaluator import evaluate_run

        payload = evaluate_run(
            run,
            split=split,
            tasks=tasks,
            checkpoint=checkpoint,
            out=out,
            tag=tag,
            local_files_only=offline,
        )
    except (MiniVerlError, ModuleNotFoundError) as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(payload)
        return
    console.print(f"[bold]eval[/bold] {_esc(payload['run_dir'])} split={_esc(payload['split'])}")
    table = Table(show_header=False, box=None)
    for key in (
        "tasks",
        "success_rate",
        "strict_task_success_rate",
        "lenient_diagnostic_success_rate",
        "avg_turns",
        "avg_tool_calls",
        "emitted_tool_calls",
        "parsed_tool_calls",
        "parse_valid_tool_call_rate",
        "tool_execution_success_rate",
        "tool_execution_error_rate",
        "final_answer_format_validity_rate",
        "protocol_token_accuracy",
        "generated_tokens_per_task",
        "rollout_tokens_per_second",
        "seconds",
    ):
        table.add_row(f"[dim]{key}[/dim]", _esc(payload.get(key)))
    console.print(table)
    console.print(f"  failures {_esc(payload.get('failure_categories'))}")
    console.print(f"  written  {_esc(payload.get('written_to'))}")


# ----------------------------------------------------------- verl bridge


@app.command("import-verl")
def import_verl_command(
    source: Path = typer.Argument(..., help="Pinned verl YAML configuration."),
    profile: str = typer.Option(..., "--profile", help="Documented bridge profile."),
    target_verl: str = typer.Option(..., "--target-verl", help="Pinned verl tag or commit."),
    out: Path = typer.Option(..., "--out", help="New miniVERL recipe path."),
    environment: Optional[str] = typer.Option(
        None, "--environment", help="Explicit registered miniVERL environment."
    ),
    teacher_model: Optional[str] = typer.Option(
        None, "--teacher-model", help="Explicit frozen teacher model identity."
    ),
    teacher_adapter: Optional[str] = typer.Option(
        None, "--teacher-adapter", help="Optional local teacher adapter path."
    ),
    loss_profile: Optional[str] = typer.Option(
        None, "--loss-profile", help="Explicit miniVERL distillation objective profile."
    ),
    schedule_mapping: Optional[str] = typer.Option(
        None, "--schedule-mapping", help="Explicit schedule-unit mapping."
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing <stem>.yaml/.template.yaml/.import-report.json family.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Import the documented whitelist, never generic verl YAML."""
    try:
        from miniverl.bridge.config import import_verl_config

        report = import_verl_config(
            source,
            profile=profile,
            target_verl=target_verl,
            out=out,
            environment=environment,
            teacher_model=teacher_model,
            teacher_adapter=teacher_adapter,
            loss_profile=loss_profile,
            schedule_mapping=schedule_mapping,
            overwrite=overwrite,
        )
    except MiniVerlError as exc:
        _fail(exc)
        return
    written = out.parent / str(report["generated_path"])
    report_file = out.parent / str(report["report_path"])
    payload = {"written": str(written), "report": str(report_file), **report}
    if as_json:
        _emit_json(payload)
        return
    style = "green" if report["status"] == "accepted" else "yellow"
    console.print(f"[{style}]verl profile {report['status']}[/{style}] {_esc(written)}")
    console.print(f"  profile {_esc(profile)}")
    console.print(f"  report  {_esc(report_file)}")


@app.command("convert-dataset")
def convert_dataset_command(
    source: Path = typer.Argument(..., help="Source Parquet dataset."),
    out: Path = typer.Option(..., "--out", help="New Parquet dataset."),
    from_format: Optional[str] = typer.Option(None, "--from", help="Source format."),
    to_format: Optional[str] = typer.Option(None, "--to", help="Destination format."),
    max_prompt_characters: Optional[int] = typer.Option(
        None,
        "--max-prompt-characters",
        min=1,
        help="Optional character-only risk bound; rows are never truncated.",
    ),
    allow_rejected_rows: bool = typer.Option(
        False,
        "--allow-rejected-rows",
        help="Publish an explicitly partial dataset instead of failing on invalid rows.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing <name>.parquet/.miniverl.json/.report.json family.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Convert the official prompt schema while preserving chat structure.

    Any invalid row fails the conversion unless --allow-rejected-rows is given.
    """
    if (from_format is None) == (to_format is None):
        _fail(ConfigError("choose exactly one of --from verl-parquet or --to verl-parquet"))
        return
    selected = from_format or to_format
    if selected != "verl-parquet":
        _fail(ConfigError(f"unsupported dataset format {selected!r}; expected 'verl-parquet'"))
        return
    direction: Literal["from-verl-parquet", "to-verl-parquet"] = (
        "from-verl-parquet" if from_format else "to-verl-parquet"
    )
    try:
        from miniverl.bridge.dataset import convert_dataset

        report = convert_dataset(
            source,
            out=out,
            direction=direction,
            max_prompt_characters=max_prompt_characters,
            allow_rejected_rows=allow_rejected_rows,
            overwrite=overwrite,
        )
    except MiniVerlError as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json({"written": str(out), **report})
        return
    console.print(f"[green]dataset converted[/green] {_esc(out)}")
    console.print(
        f"  accepted {_esc(report['accepted_rows'])}; rejected {_esc(report['rejected_rows'])}"
    )
    if report["partial_conversion"]:
        console.print(
            "  [yellow]partial conversion[/yellow]: this dataset is incomplete; "
            "only the accepted rows are lossless"
        )
    console.print(f"  sha256  {_esc(report['output_sha256'])}")


@app.command("export-verl")
def export_verl_command(
    run: Path = typer.Option(..., "--run", help="Source miniVERL run directory."),
    target_verl: str = typer.Option(..., "--target-verl", help="Pinned verl tag or commit."),
    out: Path = typer.Option(..., "--out", help="New scale-out bundle directory."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Export a self-checking miniVERL-defined Level-3 artifact bundle."""
    try:
        from miniverl.bridge.export import export_verl_bundle

        report = export_verl_bundle(run, target_verl=target_verl, out=out)
    except MiniVerlError as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json({"written": str(out), **report})
        return
    console.print(f"[green]verl bundle exported[/green] {_esc(out)}")
    console.print(f"  profile {_esc(report['profile'])}")
    console.print("  launchable: false")
    console.print("  distributed execution: not tested")


@bridge_app.command("doctor")
def bridge_doctor_command(
    bundle: Path = typer.Argument(..., help="Exported verl bundle directory."),
    require_verl: bool = typer.Option(
        False,
        "--require-verl",
        help="Fail unless the exact commit is installed from a VCS direct URL.",
    ),
    require_tokenizer_load: bool = typer.Option(
        False,
        "--require-tokenizer-load",
        help="Fail unless the tokenizer loads from the local snapshot and its identity checks out.",
    ),
    require_adapter_payload: bool = typer.Option(
        False,
        "--require-adapter-payload",
        help="Fail unless every adapter tensor payload is validated, not just the header.",
    ),
    trust_and_import_reward_code: bool = typer.Option(
        False,
        "--trust-and-import-reward-code",
        help=(
            "UNSAFE: execute the bundle's reward Python in this process. "
            "Only for bundles you produced yourself."
        ),
    ),
    scan_dataset_text: bool = typer.Option(
        False,
        "--scan-dataset-text",
        help="Run the bounded heuristic scan over string-like Parquet fields.",
    ),
    require_complete_metadata_scan: bool = typer.Option(
        False,
        "--require-complete-metadata-scan",
        help=(
            "Fail unless every portable metadata file was actually inspected. "
            "Without this, an incomplete scan that found nothing is reported as "
            "heuristic_incomplete rather than failed."
        ),
    ),
    sentinel: list[str] = typer.Option(
        [],
        "--sentinel",
        help="Extra literal string to search for during --scan-dataset-text. Repeatable.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Verify pins, standard artifacts, schema, scaffold, hashes and smoke status.

    Reward code is statically inspected and never executed by default.
    """
    from miniverl.bridge.doctor import inspect_bridge_bundle

    if trust_and_import_reward_code:
        # Printed before the import happens, so the warning survives a crash
        # caused by the untrusted module itself.
        console.print(
            "[red]WARNING[/red]: --trust-and-import-reward-code executes this bundle's "
            "Python in the current process with your privileges.\n"
            "         A subprocess would not be a security sandbox either. "
            "Only use it on bundles you produced yourself."
        )
    payload = inspect_bridge_bundle(
        bundle,
        require_verl=require_verl,
        require_tokenizer_load=require_tokenizer_load,
        require_adapter_payload=require_adapter_payload,
        trust_and_import_reward_code=trust_and_import_reward_code,
        scan_dataset_text=scan_dataset_text,
        require_complete_metadata_scan=require_complete_metadata_scan,
        sentinels=tuple(sentinel),
    )
    if as_json:
        _emit_json(payload)
    else:
        style = "green" if payload["verdict"] == "ok" else "red"
        console.print(f"[{style}]{_esc(payload['verdict'])}[/{style}] {_esc(bundle)}")
        for key in (
            "target_verl",
            "model_adapter_loadability",
            "tokenizer_verification_level",
            "parquet_schema",
            "config_profile",
            "reward_verification_level",
            "artifact_hashes",
            "local_smoke_status",
            "distributed_execution_status",
        ):
            console.print(f"  {_esc(key)}: {_esc(payload[key])}")
        if payload["reward_code_executed"]:
            console.print("  [red]reward code from this bundle was executed[/red]")
        else:
            console.print("  reward code: statically inspected, never executed")
        console.print("  privacy scopes:")
        console.print(f"    portable metadata: {_esc(payload['portable_metadata_privacy'])}")
        console.print(f"    dataset content:   {_esc(payload['dataset_content_privacy'])}")
        console.print(f"    model weights:     {_esc(payload['model_weight_privacy'])}")
        if payload["dataset_content_privacy"] == "not_inspected":
            console.print(
                "    [yellow]not_inspected does not mean passed[/yellow]; "
                "re-run with --scan-dataset-text"
            )
    if payload["verdict"] != "ok":
        raise typer.Exit(1)


# -------------------------------------------------------------- benchmark


@app.command()
def benchmark(
    config_path: Optional[Path] = typer.Argument(None, help="Benchmark config YAML."),
    output: Optional[Path] = typer.Option(None, "--output", help="Output directory."),
    export_community: Optional[Path] = typer.Option(
        None,
        "--export-community",
        help="Write a privacy-safe, schema-validated community submission template.",
    ),
    notes: str = typer.Option("", "--notes", help="Free-text notes stored in the result."),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Refuse network access; use only local or already-cached model and adapter files.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume validated per-seed/per-arm run directories after a partial benchmark.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run a matched-budget comparison across training modes."""
    if export_community is not None:
        if config_path is not None:
            _fail(ConfigError("--export-community does not accept a benchmark config argument"))
            return
        from miniverl.bridge.community import export_community_submission, validate_submission

        payload = export_community_submission(export_community)
        problems = validate_submission(export_community)
        if problems:
            _fail(ConfigError("invalid community submission: " + "; ".join(problems)))
            return
        if as_json:
            _emit_json({"written": str(export_community), "submission": payload})
        else:
            console.print(f"[green]community template exported[/green] {_esc(export_community)}")
            console.print("  status not_measured; add claims only from retained artifacts")
        return
    if config_path is None:
        _fail(ConfigError("benchmark requires CONFIG_PATH or --export-community OUTPUT"))
        return
    try:
        _require_training_stack("miniverl benchmark")
        from miniverl.evaluation.benchmark import run_benchmark
        from miniverl.evaluation.schema import BenchmarkConfig

        spec = BenchmarkConfig.from_yaml(config_path)
        result = run_benchmark(
            spec,
            output_dir=output,
            notes=notes,
            local_files_only=offline,
            resume=resume,
        )
    except (ValidationError, MiniVerlError) as exc:
        if isinstance(exc, MiniVerlError):
            _fail(exc)
        err_console.print(f"[red]invalid benchmark config[/red] {_esc(config_path)}\n{_esc(exc)}")
        raise typer.Exit(1) from None
    if as_json:
        _emit_json(result.model_dump(mode="json"))
        return
    table = Table(title=f"benchmark {result.name}")
    for column in ("arm", "mode", "loss mode", "steps", "success", "gen tok/task", "seconds"):
        table.add_column(column, justify="right" if column != "arm" else "left")
    for arm in result.arms:
        table.add_row(
            _esc(arm.name),
            _esc(arm.mode),
            _esc(arm.loss_mode),
            str(arm.optimizer_steps),
            f"{arm.success_rate * 100:.1f}%",
            f"{arm.generated_tokens_per_task:.1f}",
            f"{arm.seconds:.1f}",
        )
    console.print(table)
    if len(result.seeds) == 1:
        console.print(
            "[yellow]single seed[/yellow] - no statistical significance is claimed. "
            "Add more entries to `seeds:` for a variance estimate."
        )


# ---------------------------------------------------------------- inspect


@app.command("inspect")
def inspect_command(
    path: Path = typer.Argument(..., help="Path to a trajectories.jsonl file."),
    limit: int = typer.Option(5, "--limit", help="Number of trajectories to show."),
    trajectory: Optional[str] = typer.Option(
        None, "--trajectory", help="Inspect one trajectory id."
    ),
    show_spans: bool = typer.Option(False, "--spans", help="Print the span table of --trajectory."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate and summarize trajectories, including token provenance."""
    try:
        from miniverl.inspection import iter_spans_for_display, summarize_file

        summary = summarize_file(path, limit=limit, trajectory_id=trajectory)
    except MiniVerlError as exc:
        _fail(exc)
        return
    if summary.trajectories == 0:
        _fail(
            MiniVerlError(
                f"no trajectories matched in {path}",
                hint="drop --trajectory, or check the id with `miniverl inspect <file>`",
            )
        )
        return
    if as_json:
        _emit_json(summary.to_dict())
        return

    console.print(f"[bold]{_esc(path)}[/bold]")
    console.print(
        f"  {summary.trajectories} trajectories | {summary.tokens} tokens | "
        f"{summary.model_tokens} model tokens ({summary.model_token_fraction * 100:.1f}%) | "
        f"{summary.critical_tokens} critical"
    )
    if summary.graded:
        console.print(
            f"  graded {summary.graded} | solved {summary.solved} "
            f"({(summary.success_rate or 0) * 100:.1f}%)"
        )
    console.print(f"  policy versions {_esc(summary.policy_versions)}")
    console.print(f"  termination {_esc(summary.termination_reasons)}")
    console.print(f"  tools {_esc(summary.tools_used or {})}")
    fingerprints = [f[:12] + "..." for f in summary.tokenizer_fingerprints]
    console.print(f"  tokenizer {_esc(fingerprints)}")

    provenance = Table(title="tokens by span type (only assistant_* can enter the loss)")
    provenance.add_column("span type")
    provenance.add_column("tokens", justify="right")
    provenance.add_column("in loss")
    from miniverl.schemas.trajectory import MODEL_GENERATED_SPAN_TYPES

    trainable = {s.value for s in MODEL_GENERATED_SPAN_TYPES}
    for name, count in sorted(summary.tokens_by_span_type.items(), key=lambda kv: -kv[1]):
        provenance.add_row(
            _esc(name),
            str(count),
            "[green]yes[/green]" if name in trainable else "[dim]no (context)[/dim]",
        )
    console.print(provenance)

    table = Table(title=f"first {len(summary.samples)} trajectories")
    for column in ("trajectory", "task", "term", "solved", "turns", "tokens", "model", "crit"):
        left = column in {"trajectory", "task", "term"}
        table.add_column(
            column,
            justify="left" if left else "right",
            overflow="fold" if left else "ellipsis",
        )
    for record in summary.samples:
        table.add_row(
            _esc(record.trajectory_id),
            _esc(record.task_id),
            _esc(record.termination_reason),
            "-" if record.solved is None else ("yes" if record.solved else "no"),
            str(record.turns),
            str(record.tokens),
            str(record.model_tokens),
            str(record.critical_tokens),
        )
    console.print(table)

    if show_spans and trajectory:
        span_table = Table(title=f"spans of {trajectory}")
        for column in ("span", "range", "tokens", "in loss", "text"):
            span_table.add_column(column, overflow="fold")
        for row in iter_spans_for_display(path, trajectory):
            span_table.add_row(
                _esc(row["span_type"]),
                _esc(f"[{row['start']}, {row['end']})"),
                str(row["tokens"]),
                "yes" if row["in_loss"] else "no",
                _esc(row["text"][:400]),
            )
        console.print(span_table)


# ----------------------------------------------------------------- report


@app.command()
def report(
    run: Path = typer.Argument(..., help="Run directory."),
    out: Optional[Path] = typer.Option(None, "--out", help="HTML output path."),
    markdown: Optional[Path] = typer.Option(None, "--markdown", help="Markdown output path."),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="JSON summary output path."),
    max_trajectories: int = typer.Option(5, "--max-trajectories", help="Trajectories to render."),
    max_tokens: int = typer.Option(400, "--max-tokens", help="Tokens per trajectory to render."),
    lock_timeout: float = typer.Option(
        0.0,
        "--lock-timeout",
        min=0.0,
        help="Seconds to wait if the run is being mutated (default: fail fast).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON to stdout."),
) -> None:
    """Render a self-contained offline HTML report from a run directory."""
    try:
        from miniverl.reporting import (
            ReportData,
            render_summary_json,
            write_markdown,
            write_report,
        )
        from miniverl.utils.locking import RunLock

        with RunLock(run.parent, run.name, timeout=lock_timeout):
            data = ReportData.from_run(
                run,
                max_trajectories=max_trajectories,
                max_tokens=max_tokens,
            )
            html_path = write_report(
                run,
                out,
                max_trajectories=max_trajectories,
                max_tokens=max_tokens,
            )
            markdown_path = write_markdown(data, markdown or Path(run) / "summary.md")
            summary = render_summary_json(data)
            json_path = None
            if json_out is not None:
                from miniverl.utils.runs import write_json

                json_path = write_json(json_out, summary)
    except MiniVerlError as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(
            {
                "html": str(html_path),
                "markdown": str(markdown_path),
                "json": str(json_path) if json_path else None,
                "summary": summary,
            }
        )
        return
    console.print(f"[green]report written[/green] {_esc(html_path)}")
    console.print(f"[green]summary written[/green] {_esc(markdown_path)}")
    if json_path:
        console.print(f"[green]json written[/green] {_esc(json_path)}")
    console.print("  the HTML file is self-contained and works offline")


# ------------------------------------------------------------------ cache


@cache_app.command("stats")
def cache_stats(
    path: Path = typer.Argument(..., help="Teacher-cache directory."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Recompute shard checksums (slower, safer)."
    ),
) -> None:
    """Print compression statistics and provenance for a teacher-target cache."""
    try:
        from miniverl.cache.stats import compute_stats, format_stats

        stats = compute_stats(path, verify_checksums=verify)
    except MiniVerlError as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(stats)
        return
    # markup=False: paths and bracketed shapes must print verbatim.
    console.print(format_stats(stats), markup=False, highlight=False)


@cache_app.command("validate")
def cache_validate(
    path: Path = typer.Argument(..., help="Teacher-cache directory."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Verify cache structure and checksums; exit non-zero on any problem."""
    try:
        from miniverl.cache.store import TeacherCache

        cache = TeacherCache.open(path, verify_checksums=True)
        problems = cache.validate(verify_checksums=True)
    except MiniVerlError as exc:
        if as_json:
            _emit_json({"path": str(path), "valid": False, "problems": [str(exc)]})
            raise typer.Exit(1) from None
        _fail(exc)
        return
    payload = {
        "path": str(path),
        "valid": not problems,
        "entries": len(cache),
        "shards": len(cache.index.shards),
        "problems": problems,
    }
    if as_json:
        _emit_json(payload)
    else:
        if problems:
            err_console.print(f"[red]invalid[/red] {_esc(path)}")
            for problem in problems:
                err_console.print(f"  - {_esc(problem)}")
        else:
            console.print(
                f"[green]valid[/green] {_esc(path)}: {len(cache)} entries in "
                f"{len(cache.index.shards)} shard(s), all checksums match"
            )
    if problems:
        raise typer.Exit(1)


# --------------------------------------------------------- export-adapter


@app.command("export-adapter")
def export_adapter_command(
    run: Path = typer.Option(..., "--run", help="Source miniVERL run directory."),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Checkpoint directory (defaults to <run>/checkpoints/final).",
    ),
    out: Path = typer.Option(..., "--out", help="New standard PEFT adapter directory."),
    offline: bool = typer.Option(
        False,
        "--offline",
        "--local-files-only",
        help="Refuse network access while loading the base model and tokenizer.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Export a miniVERL LoRA checkpoint as a standard frozen PEFT adapter."""
    try:
        _require_training_stack("miniverl export-adapter")
        from miniverl.models.adapter_io import export_adapter

        source_checkpoint = checkpoint or run / "checkpoints" / "final"
        manifest, destination = export_adapter(
            run,
            source_checkpoint,
            out,
            local_files_only=offline,
        )
    except MiniVerlError as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json({"written": str(destination), "manifest": manifest})
        return
    console.print(f"[green]adapter exported[/green] {_esc(destination)}")
    console.print("  adapter_config.json")
    console.print("  adapter_model.safetensors")
    console.print("  miniverl_adapter_manifest.json")


# ------------------------------------------------------- export-benchmark


@app.command("export-benchmark")
def export_benchmark(
    run: Path = typer.Argument(..., help="Run directory to export."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output JSON path."),
    notes: str = typer.Option("", "--notes", help="Free-text notes for the submission."),
    lock_timeout: float = typer.Option(
        0.0,
        "--lock-timeout",
        min=0.0,
        help="Seconds to wait if the run is being mutated (default: fail fast).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Export a schema-validated, sanitized hardware result for a pull request."""
    try:
        from miniverl.evaluation.export import export_run
        from miniverl.utils.locking import RunLock

        with RunLock(run.parent, run.name, timeout=lock_timeout):
            payload, destination = export_run(run, out=out, notes=notes)
    except MiniVerlError as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json({"written": str(destination), "result": payload})
        return
    console.print(f"[green]exported[/green] {_esc(destination)}")
    console.print(
        "  open a pull request adding this file under benchmarks/results/ "
        "(see benchmarks/README.md)"
    )


@app.command("schema")
def schema_command(
    out: Optional[Path] = typer.Option(None, "--out", help="Write the JSON Schema to a file."),
    recoverybench: bool = typer.Option(
        False,
        "--recoverybench",
        help="Emit the RecoveryBench-v3 publication schema.",
    ),
) -> None:
    """Print the benchmark-result JSON Schema.

    Both output paths emit byte-identical text. They did not always: ``--out``
    went through :func:`~miniverl.utils.runs.write_json`, which sorts keys and
    appends a trailing newline, while stdout went through Rich, which preserves
    insertion order. CI regenerates this schema and byte-diffs it against the
    committed copy, so whichever path a contributor happens to use has to give
    the same bytes or the check fails on formatting rather than on drift.
    """
    from miniverl.evaluation.schema import json_schema, recovery_json_schema
    from miniverl.utils.runs import canonical_json, write_text

    text = canonical_json(recovery_json_schema() if recoverybench else json_schema())
    if out is not None:
        write_text(out, text)
        console.print(f"[green]schema written[/green] {_esc(out)}")
        return
    # Written to the underlying binary buffer rather than through Rich or text
    # mode, so that redirecting the output produces exactly the bytes `--out`
    # would have written. Text-mode stdout on Windows would re-expand every
    # newline to CRLF and reintroduce the platform difference this avoids.
    data = text.encode("utf-8")
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:  # captured stdout in tests has no binary buffer
        sys.stdout.write(text)
    else:
        buffer.write(data)
        buffer.flush()


def run() -> None:  # pragma: no cover - console-script shim
    """Entry point used by the ``miniverl`` console script."""
    try:
        app()
    except MiniVerlError as exc:
        _fail(exc)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    run()
