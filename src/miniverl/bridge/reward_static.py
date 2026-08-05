"""Static, non-executing inspection of an exported reward scaffold.

``miniverl bridge doctor`` inspects bundles that arrive from other people. Up to
v0.6.2 the reward check imported ``reward/reward_or_verifier_scaffold.py`` with
``spec.loader.exec_module``, so *any* top-level statement in an untrusted bundle
ran with the user's privileges as soon as they asked for a diagnosis -- while the
report described the result as a "side-effect-free import".

This module replaces that with an AST walk. Parsing does not evaluate the code:
``ast.parse`` builds a tree from source text and never executes a statement, so a
bundle can no longer act merely by being inspected.

Static inspection proves the *interface* is present and that importing the module
would not obviously run code. It proves nothing about whether the reward logic is
correct, meaningful, or safe to run later.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

__all__ = [
    "REWARD_LEVELS",
    "REQUIRED_REWARD_PARAMETERS",
    "inspect_reward_scaffold",
]

#: Ordered weakest to strongest. The default doctor path stops at
#: ``interface_statically_verified``; the strongest level requires an explicit
#: opt-in that executes untrusted code and is never reached by default.
REWARD_LEVELS = (
    "not_present",
    "syntax_valid",
    "interface_statically_verified",
    "trusted_dynamic_import_verified",
)

#: verl calls ``compute_score(data_source, solution_str, ground_truth, extra_info)``.
REQUIRED_REWARD_PARAMETERS = ("data_source", "solution_str", "ground_truth")
_OPTIONAL_REWARD_PARAMETER = "extra_info"

#: The generated fail-closed scaffold carries this sentinel inside a raised
#: message. It is found by reading the parsed constants, never by running it.
_PLACEHOLDER_SENTINEL = "complete and test reward_or_verifier_scaffold"

#: Names whose top-level use signals real side effects rather than a declaration.
_SIDE_EFFECT_HINTS = {
    "subprocess": "subprocess",
    "os": "filesystem_or_process",
    "shutil": "filesystem",
    "pathlib": "filesystem",
    "socket": "network",
    "urllib": "network",
    "requests": "network",
    "httpx": "network",
    "open": "filesystem",
    "eval": "dynamic_evaluation",
    "exec": "dynamic_evaluation",
    "compile": "dynamic_evaluation",
    "__import__": "dynamic_import",
}


def _finding(category: str, line: int, detail: str) -> dict[str, Any]:
    return {"category": category, "line": int(line), "detail": detail}


def _is_literal(node: ast.expr | None) -> bool:
    """Whether an expression is a pure literal that cannot run user code."""
    if node is None:
        return True
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _call_hint(node: ast.expr) -> str:
    """Best-effort category for a call that would run during import."""
    target: ast.expr = node
    while isinstance(target, ast.Attribute):
        target = target.value
    name = target.id if isinstance(target, ast.Name) else ""
    return _SIDE_EFFECT_HINTS.get(name, "executable_expression")


def _contains_call(node: ast.expr | None) -> ast.Call | None:
    """First call expression inside ``node``, if any."""
    if node is None:
        return None
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            return child
    return None


def _check_signature(function: ast.FunctionDef) -> list[dict[str, Any]]:
    """Verify the parameter contract without evaluating defaults."""
    findings: list[dict[str, Any]] = []
    args = function.args
    positional = [arg.arg for arg in (*args.posonlyargs, *args.args)]
    required = list(REQUIRED_REWARD_PARAMETERS)
    if positional[: len(required)] != required:
        findings.append(
            _finding(
                "signature_mismatch",
                function.lineno,
                f"compute_score must start with {', '.join(required)}; got "
                f"({', '.join(positional) or 'no positional parameters'})",
            )
        )
        return findings
    extra = positional[len(required) :]
    if extra and extra[0] != _OPTIONAL_REWARD_PARAMETER:
        findings.append(
            _finding(
                "signature_mismatch",
                function.lineno,
                f"the fourth parameter must be {_OPTIONAL_REWARD_PARAMETER!r}; got {extra[0]!r}",
            )
        )
    # Every parameter past the required three must be optional so verl can call
    # the function with three positional arguments.
    if len(extra) > len(args.defaults):
        findings.append(
            _finding(
                "signature_mismatch",
                function.lineno,
                "parameters after ground_truth must have defaults",
            )
        )
    if args.vararg is not None or args.kwarg is not None:
        findings.append(
            _finding(
                "signature_not_statically_verifiable",
                function.lineno,
                "*args/**kwargs hide the parameter contract; declare it explicitly",
            )
        )
    for default in (*args.defaults, *args.kw_defaults):
        if not _is_literal(default):
            call = _contains_call(default)
            findings.append(
                _finding(
                    "default_argument_executes",
                    getattr(default, "lineno", function.lineno),
                    "default argument values are evaluated at definition time; "
                    f"this one is {_call_hint(call.func) if call else 'not a literal'}",
                )
            )
    return findings


def _check_top_level(body: list[ast.stmt], *, context: str) -> list[dict[str, Any]]:
    """Allow only declarations; every other top-level form runs at import."""
    findings: list[dict[str, Any]] = []
    for statement in body:
        if isinstance(statement, ast.Expr):
            # A bare string is a docstring; anything else is evaluated.
            if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            call = _contains_call(statement.value)
            findings.append(
                _finding(
                    "top_level_call" if call else "top_level_expression",
                    statement.lineno,
                    f"{context} expression runs at import ({_call_hint(call.func) if call else 'evaluated'})",
                )
            )
        elif isinstance(statement, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = statement.value
            if isinstance(statement, ast.AugAssign) or not _is_literal(value):
                call = _contains_call(value)
                findings.append(
                    _finding(
                        "non_literal_assignment",
                        statement.lineno,
                        f"{context} assignment is not a literal and runs at import "
                        f"({_call_hint(call.func) if call else 'evaluated expression'})",
                    )
                )
        elif isinstance(statement, ast.ClassDef):
            if statement.decorator_list:
                findings.append(
                    _finding(
                        "decorator",
                        statement.lineno,
                        f"class {statement.name} has a decorator that is evaluated at import",
                    )
                )
            findings.extend(_check_top_level(statement.body, context=f"class {statement.name}"))
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.decorator_list:
                findings.append(
                    _finding(
                        "decorator",
                        statement.lineno,
                        f"function {statement.name} has a decorator that is evaluated at import",
                    )
                )
            for default in (*statement.args.defaults, *statement.args.kw_defaults):
                if not _is_literal(default):
                    findings.append(
                        _finding(
                            "default_argument_executes",
                            getattr(default, "lineno", statement.lineno),
                            f"default argument of {statement.name} is evaluated at import",
                        )
                    )
        else:
            findings.append(
                _finding(
                    "top_level_statement",
                    statement.lineno,
                    f"{type(statement).__name__} is executable {context} code; the scaffold "
                    "may only declare a docstring, imports, functions, classes and literals",
                )
            )
    return findings


def _dynamic_assignment(tree: ast.Module) -> ast.stmt | None:
    """Top-level statement that binds ``compute_score`` to something dynamic."""
    for statement in tree.body:
        targets: list[ast.expr] = []
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "compute_score":
                return statement
    return None


def inspect_reward_scaffold(path: str | Path, *, trust_and_import: bool = False) -> dict[str, Any]:
    """Statically verify the reward interface. Never executes the file by default.

    ``trust_and_import`` re-enables the historical dynamic import for trusted
    developer workflows. It runs untrusted code in this process; it is not a
    sandbox, and the returned report says so.
    """
    source_path = Path(path)
    check: dict[str, Any] = {
        "status": "fail",
        "verification_level": "not_present",
        "code_executed": False,
        "untrusted_code_executed": False,
        "findings": [],
        "signature": None,
        "implementation_complete": False,
        "scope": (
            "static AST inspection only; the file is parsed, never imported or executed. "
            "This does not prove the reward logic is correct or safe to run."
        ),
    }
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        check["detail"] = f"no reward scaffold at {source_path.name}: {exc}"
        return check

    try:
        tree = ast.parse(source, filename=str(source_path))
    except (SyntaxError, ValueError) as exc:
        check["detail"] = f"reward scaffold is not valid Python: {exc}"
        check["findings"] = [_finding("syntax_error", getattr(exc, "lineno", 0) or 0, str(exc))]
        return check

    check["verification_level"] = "syntax_valid"
    findings = _check_top_level(tree.body, context="module")

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "compute_score"
    ]
    dynamic = _dynamic_assignment(tree)
    if dynamic is not None:
        findings.append(
            _finding(
                "dynamic_compute_score",
                dynamic.lineno,
                "compute_score is bound by assignment; the interface cannot be verified "
                "without executing the module",
            )
        )
    elif not functions:
        findings.append(
            _finding(
                "missing_compute_score",
                0,
                "no top-level `def compute_score(...)` exists",
            )
        )
    else:
        function = functions[-1]
        if isinstance(function, ast.AsyncFunctionDef):
            findings.append(
                _finding(
                    "async_compute_score",
                    function.lineno,
                    "compute_score must be a synchronous function; verl calls it directly",
                )
            )
        else:
            findings.extend(_check_signature(function))
            check["signature"] = [
                arg.arg for arg in (*function.args.posonlyargs, *function.args.args)
            ]

    # The fail-closed placeholder is detected from parsed string constants, so a
    # scaffold never has to run to prove it is still a placeholder.
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    check["implementation_complete"] = not any(
        _PLACEHOLDER_SENTINEL in constant for constant in constants
    )

    check["findings"] = findings
    if findings:
        categories = sorted({item["category"] for item in findings})
        check["detail"] = "static inspection rejected the reward scaffold: " + ", ".join(categories)
        return check

    check["status"] = "ok"
    check["verification_level"] = "interface_statically_verified"
    check["detail"] = (
        "compute_score is declared with the expected parameters and the module "
        "contains no top-level executable statement; the file was not imported"
    )

    if trust_and_import:
        check.update(_trusted_dynamic_import(source_path))
    return check


def _trusted_dynamic_import(path: Path) -> dict[str, Any]:
    """Execute the scaffold on explicit request. This is not a security sandbox."""
    import importlib.util
    import sys

    result: dict[str, Any] = {
        "code_executed": True,
        "untrusted_code_executed": True,
        "isolation": (
            "none; the module runs in this interpreter with the caller's privileges. "
            "A subprocess would not be a security sandbox either."
        ),
    }
    try:
        spec = importlib.util.spec_from_file_location("_miniverl_exported_reward", path)
        if spec is None or spec.loader is None:
            raise ImportError("could not create an import specification")
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        if not callable(getattr(module, "compute_score", None)):
            raise ImportError("compute_score is not callable after import")
    except BaseException as exc:  # the imported module may raise anything
        result["status"] = "fail"
        result["detail"] = f"trusted dynamic import failed: {exc}"
        return result
    result["status"] = "ok"
    result["verification_level"] = "trusted_dynamic_import_verified"
    result["detail"] = (
        "compute_score imported and is callable; untrusted code from this bundle was "
        "executed in this process on explicit request"
    )
    return result
