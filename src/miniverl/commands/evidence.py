"""Portable hardware-evidence commands; intentionally torch-free."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

hardware_app = typer.Typer(
    help="Create and validate private-by-default hardware records.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _fail(exc: Exception) -> None:
    err_console.print(f"[red]error[/red] {escape(str(exc))}")
    raise typer.Exit(1)


@hardware_app.command("record")
def hardware_record_command(
    run: Path = typer.Option(..., "--run", help="Completed miniVERL run directory."),
    out: Path = typer.Option(..., "--out", help="New portable JSON record."),
    consent_to_publish: bool = typer.Option(
        False,
        "--consent-to-publish",
        help="Record publication consent; no upload is performed.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Derive a schema-validated record without uploading or exposing paths."""
    from miniverl.evidence.hardware import build_hardware_record, write_hardware_record

    try:
        record = build_hardware_record(
            run,
            consent_to_publish=consent_to_publish,
        )
        destination = write_hardware_record(record, out)
    except Exception as exc:
        _fail(exc)
        return
    payload = record.model_dump(mode="json")
    if as_json:
        console.print_json(json.dumps({"written": str(destination), "record": payload}))
    else:
        console.print(f"[green]hardware record written[/green] {escape(str(destination))}")
        console.print("  schema-valid and unreviewed; nothing was uploaded")


@hardware_app.command("validate")
def hardware_validate_command(
    path: Path = typer.Argument(..., help="Standalone hardware-record JSON."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate schema and privacy without promoting the record to measured docs."""
    from miniverl.evidence.hardware import validate_hardware_record

    problems = validate_hardware_record(path)
    payload = {
        "path": str(path),
        "valid": not problems,
        "review_status": "unreviewed",
        "problems": problems,
    }
    if as_json:
        console.print_json(json.dumps(payload))
    elif not problems:
        console.print(f"[green]valid[/green] {escape(str(path))}")
        console.print("  schema validation does not confer maintainer review or publication")
    else:
        for problem in problems:
            err_console.print(f"[red]invalid[/red] {escape(problem)}")
    if problems:
        raise typer.Exit(1)
