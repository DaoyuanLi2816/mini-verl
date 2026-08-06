"""Static, non-executing inspection of an exported reward scaffold.

``miniverl bridge doctor`` inspects bundles that arrive from other people. Up to
v0.6.2 the reward check imported ``reward/reward_or_verifier_scaffold.py`` with
``spec.loader.exec_module``, so *any* top-level statement in an untrusted bundle
ran with the user's privileges as soon as they asked for a diagnosis -- while the
report described the result as a "side-effect-free import".

This module replaces that with an AST walk. Parsing does not evaluate the code:
``ast.parse`` builds a tree from source text and never executes a statement, so a
bundle can no longer act merely by being inspected.

The policy covers *definition-time* expressions, not only top-level statements.
Class bases, class keywords such as ``metaclass=``, decorators, default
arguments, annotations and type-parameter bounds are all evaluated when a module
is imported, so a scaffold whose body is otherwise inert can still run code
through any of them. The v0.6.3 release candidate audited only decorators and
defaults and therefore accepted ``class Hidden(exploit())`` with no finding at
all.

What this proves is deliberately narrow: the expected interface is *shaped*
correctly and no executable definition-time expression forbidden by this policy
is present. It does not prove the reward body is correct, that imported modules
are side-effect free, or that the file is safe to import.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_AST_DEPTH",
    "MAX_AST_NODES",
    "MAX_FINDINGS",
    "MAX_SOURCE_BYTES",
    "REWARD_LEVELS",
    "REQUIRED_REWARD_PARAMETERS",
    "inspect_reward_scaffold",
]

#: Ordered weakest to strongest. The default doctor path stops at
#: ``interface_shape_verified`` -- deliberately weaker wording than the release
#: candidate's ``interface_statically_verified``, which read as a safety claim.
#: The strongest level requires an explicit opt-in that executes untrusted code
#: and is never reached by default.
REWARD_LEVELS = (
    "not_present",
    "syntax_valid",
    "interface_shape_verified",
    "trusted_dynamic_import_verified",
)

#: verl calls ``compute_score(data_source, solution_str, ground_truth, extra_info)``.
REQUIRED_REWARD_PARAMETERS = ("data_source", "solution_str", "ground_truth")
_OPTIONAL_REWARD_PARAMETER = "extra_info"

#: Fail-closed bounds. A hostile bundle must produce a bounded diagnostic rather
#: than exhaust the process that merely asked for a diagnosis.
MAX_SOURCE_BYTES = 1_000_000
MAX_AST_NODES = 200_000
MAX_AST_DEPTH = 60
MAX_FINDINGS = 50

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

#: Expression forms that run code where they appear. A bare ``Name``,
#: ``Attribute`` or ``Subscript`` is how ordinary type annotations and base
#: classes are written, so flagging those would reject every normal scaffold.
_EXECUTABLE_FORMS: tuple[type[ast.AST], ...] = (
    ast.Call,
    ast.Lambda,
    ast.Await,
    ast.NamedExpr,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Yield,
    ast.YieldFrom,
)


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


def _executable_form(node: ast.expr | None) -> ast.AST | None:
    """First sub-expression that would run code where ``node`` appears."""
    if node is None:
        return None
    for child in ast.walk(node):
        if isinstance(child, _EXECUTABLE_FORMS):
            return child
    return None


def _hint_for(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _call_hint(node.func)
    return type(node).__name__.lower()


def _audit_definition_expression(
    node: ast.expr | None,
    *,
    category: str,
    fallback_line: int,
    description: str,
) -> list[dict[str, Any]]:
    """Flag an expression that is evaluated when the module is imported."""
    offender = _executable_form(node)
    if offender is None:
        return []
    line = getattr(node, "lineno", fallback_line)
    return [
        _finding(
            category,
            line,
            f"{description} is evaluated at import time ({_hint_for(offender)})",
        )
    ]


def _audit_type_params(node: ast.AST, *, owner: str) -> list[dict[str, Any]]:
    """Python 3.12 type-parameter bounds and defaults are definition-time code."""
    findings: list[dict[str, Any]] = []
    for parameter in getattr(node, "type_params", ()) or ():
        for attribute in ("bound", "default_value"):
            findings.extend(
                _audit_definition_expression(
                    getattr(parameter, attribute, None),
                    category="annotation_executes",
                    fallback_line=getattr(node, "lineno", 0),
                    description=f"the type-parameter {attribute} of {owner}",
                )
            )
    return findings


def _audit_annotations(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict[str, Any]]:
    """Parameter and return annotations run unless they are strings."""
    findings: list[dict[str, Any]] = []
    args = function.args
    parameters = [
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        *(item for item in (args.vararg, args.kwarg) if item is not None),
    ]
    for parameter in parameters:
        findings.extend(
            _audit_definition_expression(
                parameter.annotation,
                category="annotation_executes",
                fallback_line=function.lineno,
                description=f"the annotation of parameter {parameter.arg!r}",
            )
        )
    findings.extend(
        _audit_definition_expression(
            function.returns,
            category="annotation_executes",
            fallback_line=function.lineno,
            description=f"the return annotation of {function.name}",
        )
    )
    findings.extend(_audit_type_params(function, owner=f"function {function.name}"))
    return findings


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
    if len(extra) > 1:
        findings.append(
            _finding(
                "signature_mismatch",
                function.lineno,
                f"compute_score accepts at most one parameter after "
                f"{required[-1]}; got {len(extra)}",
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
    # Keyword-only parameters were never inspected before v0.6.3 final, so a
    # required keyword-only ``extra_info`` -- which raises TypeError on verl's
    # three-argument call -- passed as a verified interface.
    for index, keyword_only in enumerate(args.kwonlyargs):
        if keyword_only.arg != _OPTIONAL_REWARD_PARAMETER:
            findings.append(
                _finding(
                    "signature_mismatch",
                    function.lineno,
                    f"unexpected keyword-only parameter {keyword_only.arg!r}; compute_score "
                    f"accepts only {_OPTIONAL_REWARD_PARAMETER!r} past {required[-1]}",
                )
            )
        elif args.kw_defaults[index] is None:
            findings.append(
                _finding(
                    "signature_mismatch",
                    function.lineno,
                    f"keyword-only {_OPTIONAL_REWARD_PARAMETER!r} must have a default; verl "
                    "calls compute_score with three positional arguments",
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
    findings.extend(_audit_annotations(function))
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
                    f"{context} expression runs at import "
                    f"({_call_hint(call.func) if call else 'evaluated'})",
                )
            )
        elif isinstance(statement, (ast.Import, ast.ImportFrom, ast.Pass)):
            if isinstance(statement, ast.ImportFrom) and (statement.level or 0) > 0:
                findings.append(
                    _finding(
                        "relative_import",
                        statement.lineno,
                        "the scaffold must be self-contained; a relative import pulls in "
                        "bundle-local code this check never inspected",
                    )
                )
            continue
        elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(statement, ast.AnnAssign):
                findings.extend(
                    _audit_definition_expression(
                        statement.annotation,
                        category="annotation_executes",
                        fallback_line=statement.lineno,
                        description=f"the {context} variable annotation",
                    )
                )
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
            # Bases and keywords -- including ``metaclass=`` -- are evaluated
            # when the class statement executes.
            for base in statement.bases:
                findings.extend(
                    _audit_definition_expression(
                        base,
                        category="class_base_executes",
                        fallback_line=statement.lineno,
                        description=f"a base class of {statement.name}",
                    )
                )
            for keyword in statement.keywords:
                findings.extend(
                    _audit_definition_expression(
                        keyword.value,
                        category="class_keyword_executes",
                        fallback_line=statement.lineno,
                        description=(
                            f"the {keyword.arg or '**kwargs'} class keyword of {statement.name}"
                        ),
                    )
                )
            findings.extend(_audit_type_params(statement, owner=f"class {statement.name}"))
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
            findings.extend(_audit_annotations(statement))
        elif type(statement).__name__ == "TypeAlias":  # Python 3.12+
            findings.extend(
                _audit_definition_expression(
                    getattr(statement, "value", None),
                    category="annotation_executes",
                    fallback_line=statement.lineno,
                    description=f"the {context} type alias",
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


def _ast_metrics(tree: ast.AST) -> tuple[int, int]:
    """Bounded node count and depth. Stops early once a bound is exceeded."""
    nodes = 0
    deepest = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        nodes += 1
        deepest = max(deepest, depth)
        if nodes > MAX_AST_NODES or deepest > MAX_AST_DEPTH:
            break
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return nodes, deepest


def _imported_modules(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return sorted(set(names))


def inspect_reward_scaffold(path: str | Path, *, trust_and_import: bool = False) -> dict[str, Any]:
    """Statically verify the reward interface shape. Never executes by default.

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
        "findings_truncated": False,
        "signature": None,
        "implementation_complete": False,
        "imports_present": [],
        "import_runtime_safety": "not_verified",
        "limits": {
            "max_source_bytes": MAX_SOURCE_BYTES,
            "max_ast_nodes": MAX_AST_NODES,
            "max_ast_depth": MAX_AST_DEPTH,
            "max_findings": MAX_FINDINGS,
        },
        "scope": (
            "static AST inspection only; the file is parsed, never imported or executed. "
            "It shows the expected interface shape is present and that no definition-time "
            "expression forbidden by this policy was found. It does not prove the reward "
            "logic is correct, that imported modules are side-effect free, or that the "
            "file is safe to import."
        ),
    }

    def _finish(findings: list[dict[str, Any]]) -> dict[str, Any]:
        check["findings_truncated"] = len(findings) > MAX_FINDINGS
        check["findings"] = findings[:MAX_FINDINGS]
        return check

    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        check["detail"] = f"no reward scaffold at {source_path.name}: {exc}"
        return check

    if len(raw) > MAX_SOURCE_BYTES:
        check["detail"] = (
            f"reward scaffold is {len(raw)} bytes; this static check refuses to parse "
            f"more than {MAX_SOURCE_BYTES}"
        )
        return _finish([_finding("source_too_large", 0, "source exceeds the inspection bound")])

    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        check["detail"] = f"reward scaffold is not valid UTF-8: {exc.reason}"
        return _finish([_finding("encoding_error", 0, "source is not decodable as UTF-8")])
    # CPython strips a UTF-8 BOM when it reads a source file, so a scaffold
    # saved by a Windows editor imports fine. Handing the leading U+FEFF to
    # ast.parse instead reports a perfectly good file as a syntax error.
    source = source.lstrip("﻿")

    try:
        tree = ast.parse(source, filename=str(source_path))
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        check["detail"] = f"reward scaffold is not parseable Python: {type(exc).__name__}"
        return _finish(
            [_finding("syntax_error", getattr(exc, "lineno", 0) or 0, type(exc).__name__)]
        )

    nodes, depth = _ast_metrics(tree)
    if nodes > MAX_AST_NODES:
        check["detail"] = f"reward scaffold exceeds {MAX_AST_NODES} AST nodes"
        return _finish([_finding("ast_too_large", 0, "syntax tree exceeds the inspection bound")])
    if depth > MAX_AST_DEPTH:
        check["detail"] = f"reward scaffold nests deeper than {MAX_AST_DEPTH} AST levels"
        return _finish([_finding("ast_too_deep", 0, "syntax tree exceeds the depth bound")])

    check["verification_level"] = "syntax_valid"
    check["imports_present"] = _imported_modules(tree)

    try:
        findings = _check_top_level(tree.body, context="module")
    except RecursionError:
        check["detail"] = "reward scaffold nests too deeply to inspect"
        return _finish([_finding("ast_too_deep", 0, "static inspection exceeded its depth bound")])

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

    if findings:
        categories = sorted({item["category"] for item in findings})
        check["detail"] = "static inspection rejected the reward scaffold: " + ", ".join(categories)
        return _finish(findings)

    _finish(findings)
    check["status"] = "ok"
    check["verification_level"] = "interface_shape_verified"
    check["detail"] = (
        "the expected compute_score interface was found and no executable definition-time "
        "expression forbidden by this static policy was detected; the reward body and any "
        "imported modules were not executed and are not proven safe"
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
