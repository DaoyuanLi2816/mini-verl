"""Profile registry and compatibility-introspection commands."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from miniverl.errors import MiniVerlError, SerializationError

profiles_app = typer.Typer(
    help="Inspect the closed registry of versioned compatibility profiles.",
    no_args_is_help=True,
)
compat_app = typer.Typer(
    help="Explain and check fields against one documented compatibility profile.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _esc(value: object) -> str:
    return escape(str(value))


def _fail(exc: Exception) -> None:
    if isinstance(exc, MiniVerlError):
        err_console.print(f"[red]error[/red] {_esc(exc.message)}")
        if exc.hint:
            err_console.print(f"[yellow]hint[/yellow]  {_esc(exc.hint)}")
    else:
        err_console.print(f"[red]error[/red] {_esc(exc)}")
    raise typer.Exit(1)


def _emit_json(payload: Any) -> None:
    try:
        console.print_json(json.dumps(payload, default=str, allow_nan=False))
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        _fail(SerializationError(f"command result is not finite JSON: {exc}"))


@profiles_app.command("list")
def profiles_list_command(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List built-ins; third-party code loading is intentionally unsupported."""
    from miniverl.bridge.profiles import list_profiles

    payload = [item.model_dump(mode="json") for item in list_profiles()]
    if as_json:
        _emit_json(payload)
        return
    table = Table(title="miniVERL compatibility profiles")
    table.add_column("profile", style="bold")
    table.add_column("objective")
    table.add_column("teacher target")
    table.add_column("status")
    for item in payload:
        table.add_row(
            _esc(item["name"]),
            _esc(item["objective"]),
            _esc(item["teacher_target"]),
            _esc(item["status"]),
        )
    console.print(table)


@profiles_app.command("show")
def profiles_show_command(
    profile: str = typer.Argument(..., help="Built-in compatibility profile name."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show immutable identity, algorithm boundary and copyable examples."""
    from miniverl.bridge.profiles import get_profile

    try:
        payload = get_profile(profile).show()
    except Exception as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(payload)
        return
    console.print(f"[bold]{_esc(payload['name'])}[/bold]")
    console.print(f"  objective: {_esc(payload['objective'])}")
    console.print(f"  teacher target: {_esc(payload['teacher_target'])}")
    console.print(f"  identity: {_esc(payload['identity']['digest'])}")
    console.print("\n[bold]Copyable profile YAML[/bold]")
    console.print(payload["minimal_yaml"], markup=False)
    console.print("[bold]Copyable override invocation[/bold]")
    console.print(payload["override_invocation"], markup=False)


@profiles_app.command("schema")
def profiles_schema_command(
    profile: str = typer.Argument(..., help="Built-in compatibility profile name."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Print the exact JSON Schema accepted by a profile compiler."""
    from miniverl.bridge.profiles import get_profile

    try:
        payload = get_profile(profile).config_schema()
    except Exception as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(payload)
    else:
        console.print(json.dumps(payload, indent=2, sort_keys=True), markup=False)


@compat_app.command("explain")
def compat_explain_command(
    field: str = typer.Argument(..., help="Resolved upstream field path."),
    profile: str = typer.Option(..., "--profile", help="Compatibility profile name."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Explain whether one source field is accepted, effective or informational."""
    from miniverl.bridge.profiles import get_profile

    try:
        payload = get_profile(profile).explain(field).model_dump(mode="json")
    except Exception as exc:
        _fail(exc)
        return
    if as_json:
        _emit_json(payload)
        return
    console.print(f"[bold]{_esc(field)}[/bold]")
    console.print(f"  classification: {_esc(payload['classification'])}")
    console.print(f"  local target: {_esc(payload['local_target'] or 'none')}")
    console.print(f"  accepted: {_esc(payload['field_accepted'])}")
    console.print(f"  effective: {_esc(payload['field_effective'])}")
    console.print(f"  reason: {_esc(payload['reason'])}")


@compat_app.command("check")
def compat_check_command(
    profile: str = typer.Option(..., "--profile", help="Compatibility profile name."),
    config: str = typer.Option(..., "--config", help="Resolved YAML path or builtin profile."),
    accept_local_reinterpretations: bool = typer.Option(
        False,
        "--accept-local-reinterpretations",
        help="Accept documented high-risk local reinterpretations.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Compile a config and report algorithm, field and applicability status."""
    from miniverl.bridge.profiles import check_profile

    try:
        report = check_profile(
            profile,
            config,
            accept_local_reinterpretations=accept_local_reinterpretations,
        )
    except Exception as exc:
        _fail(exc)
        return
    payload = report.model_dump(mode="json")
    if as_json:
        _emit_json(payload)
    else:
        style = "green" if report.status == "compatible" else "yellow"
        console.print(f"[{style}]{_esc(report.status)}[/{style}] {_esc(profile)}")
        for key, value in report.summary.items():
            console.print(f"  {_esc(key)}: {_esc(value)}")
