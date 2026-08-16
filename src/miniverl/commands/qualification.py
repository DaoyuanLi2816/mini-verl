"""Torch-free exact-commit GPU qualification inspection commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

qualification_app = typer.Typer(
    help="Validate exact-commit GPU qualification artifacts.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


@qualification_app.command("validate")
def qualification_validate_command(
    path: Path = typer.Argument(..., help="qualification.json from a GPU workflow artifact."),
    commit: str | None = typer.Option(None, "--commit", help="Required exact source SHA."),
    wheel_sha256: str | None = typer.Option(
        None, "--wheel-sha256", help="Required release-wheel SHA-256."
    ),
    known_good_sha256: str | None = typer.Option(
        None, "--known-good-sha256", help="Required known-good manifest SHA-256."
    ),
    required_gpu_name: str | None = typer.Option(
        None, "--required-gpu-name", help="Required measured GPU name."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate schema, privacy, hashes and optional release bindings."""
    from miniverl.qualification import validate_qualification_file

    problems = validate_qualification_file(
        path,
        expected_commit=commit,
        expected_wheel_sha256=wheel_sha256,
        expected_known_good_sha256=known_good_sha256,
        required_gpu_name=required_gpu_name,
    )
    payload = {"valid": not problems, "problems": problems}
    if as_json:
        console.print_json(json.dumps(payload, allow_nan=False))
    elif problems:
        for problem in problems:
            err_console.print(f"[red]invalid[/red] {escape(problem)}")
    else:
        console.print("[green]valid[/green] exact-commit GPU qualification")
    if problems:
        raise typer.Exit(1)
