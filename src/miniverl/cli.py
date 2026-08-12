"""miniVERL command line interface.

A thin layer: every command parses arguments, calls one library function, and
renders the result.  No training logic lives here.

Only the heavy commands import torch, and they do it inside the command body,
so ``miniverl --help``, ``doctor``, ``validate``, ``inspect``, ``report`` and
``cache`` all work from a bare ``pip install miniverl``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
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
        "Auditable single-GPU alignment and distillation runtime.\n\n"
        "Run native local workflows, inspect every artifact, and exchange standard "
        "HF/PEFT/Parquet artifacts through a bounded verl artifact bridge."
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
alignment_suite_app = typer.Typer(
    help=(
        "Prepare, validate and report the pinned external alignment suite. "
        "Needs the alignment-benchmarks extra."
    ),
    no_args_is_help=True,
)
app.add_typer(alignment_suite_app, name="alignment-suite")
evidence_app = typer.Typer(
    help="Show and validate evidence packaged with the installed wheel.", no_args_is_help=True
)
app.add_typer(evidence_app, name="evidence")
data_app = typer.Typer(help="Create and inspect portable prompt data.", no_args_is_help=True)
app.add_typer(data_app, name="data")

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
    """miniVERL: an auditable single-GPU alignment and distillation runtime."""
    from miniverl.utils.logging import configure_logging

    configure_logging(log_level)


@data_app.command("sample")
def data_sample_command(
    out: Path = typer.Option(..., "--out", help="Output Parquet path."),
    format_name: str = typer.Option(
        "verl-parquet", "--format", help="Portable output format (verl-parquet only)."
    ),
    rows: int = typer.Option(4, "--rows", min=1, max=1024, help="Number of sample prompts."),
) -> None:
    """Create a small reward-free verl-style Parquet prompt dataset."""
    if format_name != "verl-parquet":
        _fail(ConfigError("--format must be verl-parquet"))
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        from miniverl.errors import MissingDependencyError

        _fail(MissingDependencyError(exc.name or "pyarrow", "bridge", "Parquet sample data"))
        return
    prompts = [
        "Explain why exact provenance matters in one concise sentence.",
        "Give one safe way to recover from a CUDA out-of-memory error.",
        "What does token-mean loss aggregation mean?",
        "State one limitation of a single-GPU training runtime.",
    ]
    records = []
    for index in range(rows):
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": "Answer clearly and briefly."},
                    {"role": "user", "content": prompts[index % len(prompts)]},
                ],
                "data_source": "miniverl_quickstart",
                "ability": "short_answer",
                "extra_info": {"sample_index": index},
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    console.print(f"[green]wrote[/green] {_esc(out)} ({rows} rows, sha256 {digest})")


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
        (
            "single-GPU CUDA training (native recipes)",
            "gpu_training",
            "install a CUDA build of torch",
        ),
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
    environment_config = config.environment
    if environment_config is not None:
        try:
            environment = make_environment(environment_config.name, **environment_config.params)
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
        "source_kind": config.source.kind.value,
        "environment": environment_config.name if environment_config is not None else None,
        "difficulty": environment_config.difficulty if environment_config is not None else None,
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
        "eval_tasks": (
            config.effective_eval_tasks if environment_config is not None else config.eval.tasks
        ),
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
        if config.require_environment("qualify-teacher").name != "sqlite_recovery":
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


# ------------------------------------------------------------- evidence


@evidence_app.command("show")
def evidence_show(
    study_id: str = typer.Argument(..., help="Packaged study identifier."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show a packaged, typed study result without a repository checkout."""
    from miniverl.evidence import show_builtin_study

    try:
        payload = show_builtin_study(study_id)
    except (MiniVerlError, OSError, ValidationError) as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(payload)
        return
    console.print(f"[bold]{_esc(study_id)}[/bold]")
    console.print_json(json.dumps(payload["result"], allow_nan=False))


@evidence_app.command("validate")
def evidence_validate(
    study_id: str = typer.Argument(..., help="Packaged study identifier."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate packaged result, schema, preregistration and task evidence."""
    from miniverl.evidence import validate_builtin_study

    try:
        payload = validate_builtin_study(study_id)
    except (MiniVerlError, OSError, ValidationError) as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(payload)
    elif payload["valid"]:
        console.print(
            f"[green]valid[/green] {_esc(study_id)} · {_esc(payload['task_rows'])} task rows"
        )
    else:
        for problem in payload["problems"]:
            err_console.print(f"[red]invalid[/red] {_esc(problem)}")
        raise typer.Exit(1)


# ----------------------------------------------------------------- pilot


@app.command()
def pilot(
    recipe: Optional[Path] = typer.Argument(
        None, help="Alignment recipe containing bounded pilot evidence."
    ),
    study_result: Optional[Path] = typer.Option(
        None,
        "--study-result",
        help="Schema-validated external-study result; does not load a model.",
    ),
    builtin_study: Optional[str] = typer.Option(
        None,
        "--builtin-study",
        help="Packaged external-study result; works from an installed wheel.",
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional JSON output path."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Recommend a method from explicit pilot evidence without loading a model."""
    from miniverl.utils.runs import write_json_atomic

    payload: dict[str, Any]
    try:
        selected = sum(value is not None for value in (recipe, study_result, builtin_study))
        if selected > 1:
            raise ConfigError(
                "miniverl pilot accepts exactly one of a recipe, --study-result, or --builtin-study"
            )
        builtin = None
        if builtin_study is not None:
            from miniverl.evidence import get_builtin_study

            builtin = get_builtin_study(builtin_study)
            study_result = builtin.result_path
        if study_result is not None:
            from miniverl.alignment_external.result import load_alignment_external_result

            result = load_alignment_external_result(study_result)
            if result.study_status != "terminated_at_checkpoint_selection":
                raise ConfigError(
                    "this pilot evidence path currently requires "
                    "study_status=terminated_at_checkpoint_selection"
                )
            payload = {
                "study_status": result.study_status,
                "recommendation": "do_not_continue",
                "recommendation_scope": "do_not_continue_this_study",
                "method_recommendation": "insufficient_evidence",
                "reasons": [
                    "no candidate satisfied the retained-utility gate",
                    "no starting checkpoint was selected",
                    "teacher qualification was not run",
                    "no continuation method was authorized",
                    "the reserved final test was not accessed",
                ],
                "evidence": {
                    "path": str(study_result),
                    "builtin_study": builtin.study_id if builtin is not None else None,
                    "sha256": hashlib.sha256(study_result.read_bytes()).hexdigest(),
                    "preregistration": result.preregistration.model_dump(mode="json"),
                    "task_evidence": result.checkpoint_selection.task_evidence.model_dump(
                        mode="json"
                    ),
                },
                "universal_claim": False,
            }
        else:
            if recipe is None:
                raise ConfigError(
                    "miniverl pilot requires a recipe, --study-result or --builtin-study"
                )
            from miniverl.alignment import PilotEvidence, recommend_alignment_method
            from miniverl.config import RunConfig

            config = RunConfig.from_yaml(recipe)
            if config.alignment is None:
                raise ConfigError("miniverl pilot requires a recipe with an alignment section")
            evidence = config.alignment.pilot or PilotEvidence()
            recommendation = recommend_alignment_method(evidence)
            payload = recommendation.model_dump(mode="json")
        if out is not None:
            write_json_atomic(out, payload)
    except (ValidationError, MiniVerlError) as exc:
        if isinstance(exc, MiniVerlError):
            _fail(exc)
        source = recipe if recipe is not None else study_result or builtin_study
        err_console.print(f"[red]invalid pilot evidence[/red] {_esc(source)}\n{_esc(exc)}")
        raise typer.Exit(1) from None
    if as_json:
        _emit_json(payload)
        return
    recommendation_text = payload.get("method_recommendation", payload["recommendation"])
    console.print(f"[bold]recommendation[/bold] {_esc(recommendation_text)}")
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


# ---------------------------------------------------------- verl-shaped plan/run


@app.command("plan")
def plan_command(
    config: str = typer.Option(..., "--config", help="Resolved YAML path or builtin profile."),
    profile: str = typer.Option(
        "verl-opd-v0.8-single-gpu-v1", "--profile", help="Pinned compatibility profile."
    ),
    overrides: list[str] = typer.Option([], "--set", help="Repeatable dotted key=value override."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    offline: bool = typer.Option(False, "--offline", help="Do not access the network."),
    probe: bool = typer.Option(
        False, "--probe", help="Load models for a bounded probe (not available in v0.8.0)."
    ),
) -> None:
    """Plan pinned single-GPU verl-style OPD without loading model weights."""
    del offline
    try:
        from miniverl.bridge.opd_runtime import build_system_plan
        from miniverl.bridge.opd_v08 import (
            VERL_OPD_V08_PROFILE,
            load_verl_opd_v08_source,
        )

        if profile != VERL_OPD_V08_PROFILE:
            raise ConfigError(
                f"unsupported OPD profile {profile!r}", hint=f"use --profile {VERL_OPD_V08_PROFILE}"
            )
        if probe:
            raise ConfigError(
                "--probe is not implemented in v0.8.0",
                hint="run without --probe for the weight-free estimate",
            )
        compiled = load_verl_opd_v08_source(config, overrides=overrides)
        plan = build_system_plan(compiled)
        payload = plan.model_dump(mode="json")
    except MiniVerlError as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(payload)
        return
    console.print(f"[green]executable: {str(plan.executable).lower()}[/green]")
    console.print(f"  profile {_esc(plan.profile)}")
    console.print(f"  verl {_esc(plan.upstream['tag'])} @ {_esc(plan.upstream['commit'][:12])}")
    console.print(f"  student {_esc(plan.student['model_id'])} @ {_esc(plan.student['revision'])}")
    console.print(f"  teacher {_esc(plan.teacher['model_id'])} @ {_esc(plan.teacher['revision'])}")
    console.print(
        f"  loss {_esc(plan.loss['mode'])}, {_esc(plan.loss['aggregation'])}, "
        f"top-k {_esc(plan.loss['top_k'])}"
    )
    console.print(
        f"  placement {_esc(plan.local_execution['strategy'])}: "
        f"{_esc(plan.local_execution['reason'])}"
    )
    console.print("  memory estimates (not measurements)")
    for key, value in plan.memory.items():
        console.print(f"    {_esc(key)}: {_esc(value)}")
    console.print("  time to first update: unknown (not measured by plan)")
    console.print(f"  plan sha256 {_esc(plan.compiled_digest)}")


@app.command("run")
def verl_run_command(
    config: str = typer.Option(..., "--config", help="Resolved YAML path or builtin profile."),
    profile: str = typer.Option(
        "verl-opd-v0.8-single-gpu-v1", "--profile", help="Pinned compatibility profile."
    ),
    overrides: list[str] = typer.Option([], "--set", help="Repeatable dotted key=value override."),
    output: Optional[Path] = typer.Option(None, "--output", help="Parent run directory."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run id."),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume an existing run."),
    offline: bool = typer.Option(False, "--offline", help="Use cached model files only."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compile native config only."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Execute the pinned verl v0.8 pure-OPD subset on one local CUDA GPU."""
    try:
        from miniverl.bridge.opd_runtime import build_system_plan, compile_native_run_config
        from miniverl.bridge.opd_v08 import (
            VERL_OPD_V08_PROFILE,
            load_verl_opd_v08_source,
        )

        if profile != VERL_OPD_V08_PROFILE:
            raise ConfigError(
                f"unsupported OPD profile {profile!r}", hint=f"use --profile {VERL_OPD_V08_PROFILE}"
            )
        compiled = load_verl_opd_v08_source(config, overrides=overrides)
        system_plan = build_system_plan(compiled)
        native = compile_native_run_config(compiled, system_plan=system_plan)
        if dry_run:
            payload = {
                "dry_run": True,
                "compatibility": compiled.model_dump(mode="json"),
                "system_plan": system_plan.model_dump(mode="json"),
                "resolved_native_config": native.model_dump(mode="json"),
            }
            if as_json:
                _emit_json(payload)
            else:
                console.print("[green]run dry run ok[/green]")
                console.print(f"  plan sha256 {_esc(compiled.compiled_digest)}")
                console.print("  no model weights loaded")
            return
        _require_training_stack("miniverl run")
        from miniverl.trainer import OPDTrainer
        from miniverl.utils.runs import read_jsonl, write_json_atomic

        construction_started = time.perf_counter()
        trainer_instance = OPDTrainer.from_config(
            native,
            output_dir=output,
            run_id=run_id,
            local_files_only=offline,
            resume=resume,
        )
        construction_seconds = time.perf_counter() - construction_started
        with trainer_instance as trainer:
            write_json_atomic(
                trainer.paths.root / "verl-source-config.json",
                compiled.source.model_dump(mode="json"),
            )
            write_json_atomic(
                trainer.paths.root / "verl-compatibility-report.json",
                compiled.model_dump(mode="json"),
            )
            write_json_atomic(
                trainer.paths.root / "local-execution-plan.json",
                system_plan.model_dump(mode="json"),
            )
            result = trainer.train()
            paths = trainer.paths
            if resume is not None:
                measurements = {
                    "schema_version": 1,
                    "status": "measured",
                    "resume_load_seconds": round(construction_seconds, 4),
                    "resume_from": str(resume),
                    "global_optimizer_step": result.global_step,
                    "distributed_execution": False,
                }
                write_json_atomic(paths.root / "verl-resume-measurements.json", measurements)
                cycle_rows = []
            else:
                cycle_rows = [
                    row for row in read_jsonl(paths.metrics) if row.get("phase") == "opd_cycle"
                ]
            first_cycle = cycle_rows[0] if cycle_rows else {}
            final_metrics = result.final_metrics
            checkpoint = paths.checkpoints / "final"
            checkpoint_bytes = sum(
                item.stat().st_size for item in checkpoint.rglob("*") if item.is_file()
            )
            run_bytes = sum(item.stat().st_size for item in paths.root.rglob("*") if item.is_file())
            adapter_file = checkpoint / "adapter.safetensors"
            fresh_measurements = {
                "schema_version": 1,
                "status": "measured",
                "hardware": {
                    "device": trainer.plan.device,
                    "gpu_count": 1,
                    "distributed_execution": False,
                },
                "construction_seconds": round(construction_seconds, 4),
                "time_to_first_rollout_seconds": round(
                    construction_seconds + float(first_cycle.get("rollout_seconds") or 0.0), 4
                ),
                "time_to_first_teacher_target_batch_seconds": round(
                    construction_seconds
                    + float(first_cycle.get("rollout_seconds") or 0.0)
                    + float(first_cycle.get("teacher_scoring_seconds") or 0.0),
                    4,
                ),
                "time_to_first_optimizer_update_seconds": round(
                    construction_seconds + float(first_cycle.get("seconds") or 0.0), 4
                ),
                "rollout_tokens_per_second": (
                    round(
                        float((first_cycle.get("rollouts") or {}).get("generated_tokens") or 0)
                        / float(first_cycle.get("rollout_seconds") or 1.0),
                        2,
                    )
                ),
                "teacher_scored_positions_per_second": first_cycle.get(
                    "teacher_scored_positions_per_second"
                ),
                "update_selected_positions_per_second": final_metrics.get(
                    "train_selected_tokens_per_second"
                ),
                "peak_allocated_gib": (final_metrics.get("memory") or {}).get("peak_allocated_gib"),
                "peak_reserved_gib": (final_metrics.get("memory") or {}).get("peak_reserved_gib"),
                "checkpoint_bytes": checkpoint_bytes,
                "run_artifacts_bytes_before_measurement_manifest": run_bytes,
                "adapter_sha256": (
                    hashlib.sha256(adapter_file.read_bytes()).hexdigest()
                    if adapter_file.is_file()
                    else None
                ),
                "runtime_correctness_only": True,
                "alignment_quality_claim": False,
            }
            if resume is None:
                measurements = fresh_measurements
                write_json_atomic(paths.root / "verl-reference-measurements.json", measurements)
    except (MiniVerlError, ModuleNotFoundError, ValidationError) as exc:
        _fail(exc)
        return
    payload = {**result.to_dict(), "run_dir": str(paths.root), "measurements": measurements}
    if as_json:
        _emit_json(payload)
    else:
        console.print(f"[bold green]OPD run complete[/bold green] {_esc(paths.root)}")
        console.print(f"  optimizer steps {_esc(result.global_step)}")
        console.print("  distributed execution false")


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


@bridge_app.command("compile-opd")
def bridge_compile_opd_command(
    config: str = typer.Option(..., "--config", help="Resolved YAML path or builtin profile."),
    profile: str = typer.Option(
        "verl-opd-v0.8-single-gpu-v1",
        "--profile",
        help="Pinned miniVERL compatibility profile.",
    ),
    overrides: list[str] = typer.Option(
        [],
        "--set",
        help="Repeatable dotted key=value override. No Hydra or shell evaluation.",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Atomically write the compiled machine-readable plan.",
    ),
    inspect_unsupported: bool = typer.Option(
        False,
        "--inspect-unsupported",
        help="Return a non-executable report instead of failing on unsupported values.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Compile the pinned single-GPU verl v0.8 OPD config subset offline."""
    try:
        from miniverl.bridge.opd_v08 import VERL_OPD_V08_PROFILE, load_verl_opd_v08_source

        if profile != VERL_OPD_V08_PROFILE:
            raise ConfigError(
                f"unsupported OPD profile {profile!r}",
                hint=f"use --profile {VERL_OPD_V08_PROFILE}",
            )
        plan = load_verl_opd_v08_source(
            config,
            overrides=overrides,
            require_executable=not inspect_unsupported,
        )
        payload = plan.model_dump(mode="json")
        if out is not None:
            from miniverl.utils.runs import write_json_atomic

            write_json_atomic(out, payload)
    except MiniVerlError as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(payload)
        return
    style = "green" if plan.executable else "red"
    console.print(f"[{style}]config semantics executable: {str(plan.executable).lower()}[/{style}]")
    console.print(f"  profile: {_esc(plan.profile)}")
    console.print(f"  verl: {_esc(plan.upstream['tag'])} @ {_esc(plan.upstream['commit'][:12])}")
    console.print(f"  plan sha256: {_esc(plan.compiled_digest)}")
    console.print("  runtime execution: not provided by this config-only command")
    if out is not None:
        console.print(f"  written: {_esc(out)}")


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


@alignment_suite_app.command("prepare")
def alignment_suite_prepare(
    profile: Path = typer.Option(
        ...,
        "--profile",
        exists=True,
        dir_okay=False,
        help="Evaluation profile, e.g. benchmarks/external-alignment/profile-v1.yaml",
    ),
    out: Path = typer.Option(..., "--out", help="Directory to write suite-manifest.json into."),
    registry: Optional[Path] = typer.Option(
        None, "--registry", help="Endpoint registry; defaults to the committed one."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve and report without writing the manifest."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the manifest summary as JSON."),
) -> None:
    """Freeze one evaluation suite.

    This is the step that needs network access: it resolves each endpoint at
    its pinned revision and selects tasks from benchmark metadata alone. Every
    later step runs offline against the manifest written here.
    """
    import yaml as _yaml

    from miniverl.alignment_external.suite import prepare_suite

    payload = _yaml.safe_load(profile.read_text(encoding="utf-8"))
    try:
        manifest = prepare_suite(
            profile=payload,
            out=out,
            registry_path=registry,
            resolver=_hub_task_resolver(),
            dry_run=dry_run,
        )
    except (ValueError, OSError) as exc:
        from miniverl.errors import ConfigError

        _fail(ConfigError(str(exc)))

    summary = {
        "profile_id": manifest["profile_id"],
        "manifest_digest": manifest["manifest_digest"],
        "generation_tasks_per_model": manifest["generation_tasks_per_model"],
        "endpoints": {entry["id"]: entry["selected_tasks"] for entry in manifest["endpoints"]},
        "written": None if dry_run else str(Path(out) / "suite-manifest.json"),
    }
    if as_json:
        _emit_json(summary)
    else:
        console.print(summary)


@alignment_suite_app.command("validate")
def alignment_suite_validate(
    results: Path = typer.Argument(..., exists=True, help="Task-level JSONL result file."),
    manifest: Path = typer.Option(
        ..., "--manifest", exists=True, dir_okay=False, help="suite-manifest.json to check against."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the problem list as JSON."),
) -> None:
    """Check a finished result set against the suite it claims to come from."""
    from miniverl.alignment_external.suite import validate_results

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    problems = validate_results(manifest_payload, rows)

    if as_json:
        _emit_json({"rows": len(rows), "problems": problems, "valid": not problems})
    elif problems:
        for problem in problems:
            err_console.print(f"[red]-[/red] {problem}")
    else:
        console.print(f"[green]{len(rows)} result row(s) match the suite manifest[/green]")
    if problems:
        raise typer.Exit(code=1)


@alignment_suite_app.command("report")
def alignment_suite_report(
    results: Path = typer.Argument(..., exists=True, help="Task-level JSONL result file."),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Aggregate task rows into per-endpoint metrics."""
    from miniverl.alignment_external.suite import report_suite

    rows = [
        json.loads(line)
        for line in results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = report_suite(rows)
    if as_json:
        _emit_json(report)
    else:
        console.print(report)


def _hub_task_resolver() -> Any:
    """Resolve upstream task ids for an endpoint. Requires the benchmark extra."""

    def resolve(endpoint: dict[str, Any]) -> tuple[list[str], list[str] | None]:
        if endpoint.get("dataset") is None:
            # A miniVERL-internal measurement: deterministic task ids, no Hub.
            count = 256
            return [f"{endpoint['id']}-{index:04d}" for index in range(count)], None
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - dependency boundary
            from miniverl.errors import MissingDependencyError

            raise MissingDependencyError(
                "datasets", "alignment-benchmarks", "external benchmark preparation"
            ) from exc

        dataset = load_dataset(
            endpoint["dataset"],
            endpoint.get("config"),
            split=endpoint["split"],
            revision=endpoint["revision"],
        )
        task_ids = [f"{endpoint['id']}-{index:05d}" for index in range(dataset.num_rows)]
        field = endpoint.get("strata_field")
        strata = [str(value) for value in dataset[field]] if field else None
        return task_ids, strata

    return resolve


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
