"""SQLite environment: read-only queries against a seeded in-memory database.

Safety model
------------
Three independent mechanisms, all of them enforced rather than advisory:

1. **An SQLite authorizer.**  The connection denies every action by default and
   allows only ``SQLITE_SELECT``, ``SQLITE_READ`` on the two known tables, and
   ``SQLITE_FUNCTION`` for the whitelisted aggregate functions.  ``ATTACH``,
   ``PRAGMA``, ``INSERT``, ``UPDATE``, ``DELETE``, ``CREATE`` and ``DROP`` are
   refused by the engine itself, not by string matching.
2. **A progress handler.**  Queries are aborted after a bounded number of VM
   instructions, so a cartesian product cannot hang a training run.
3. **Structural limits.**  One statement per call, a length cap, and a row cap
   on the result.

The database lives in ``:memory:`` and is rebuilt from the task seed, so no
file on the user's disk is ever opened.
"""

from __future__ import annotations

import contextlib
import json
import random
import sqlite3
from typing import Any

from miniverl.environments.base import (
    FailureCategory,
    Observation,
    OracleAction,
    OracleActionKind,
    StepResult,
    Task,
    ToolCall,
    ToolEnvironment,
    ToolSpec,
    VerificationResult,
)
from miniverl.errors import ToolEnvironmentError

__all__ = ["SqliteEnvironment", "SCHEMA_DDL", "MAX_ROWS", "MAX_SQL_CHARS"]

MAX_ROWS = 20
MAX_SQL_CHARS = 400
MAX_VM_STEPS = 200_000
_PROGRESS_INTERVAL = 1000

ALLOWED_TABLES = ("customers", "orders")
ALLOWED_FUNCTIONS = frozenset(
    {"count", "sum", "avg", "min", "max", "round", "abs", "lower", "upper", "length", "total"}
)

SCHEMA_DDL = """CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount INTEGER NOT NULL,
    status TEXT NOT NULL
);"""

_CITIES = ("Lyon", "Osaka", "Porto", "Quito", "Riga", "Split")
_NAMES = (
    "Ada",
    "Bo",
    "Cyd",
    "Dov",
    "Eli",
    "Fay",
    "Gus",
    "Hal",
    "Ida",
    "Jem",
    "Kit",
    "Lou",
)
_STATUSES = ("shipped", "pending", "cancelled")

_SQLITE_SELECT = getattr(sqlite3, "SQLITE_SELECT", 21)
_SQLITE_READ = getattr(sqlite3, "SQLITE_READ", 20)
_SQLITE_FUNCTION = getattr(sqlite3, "SQLITE_FUNCTION", 31)
_SQLITE_OK = getattr(sqlite3, "SQLITE_OK", 0)
_SQLITE_DENY = getattr(sqlite3, "SQLITE_DENY", 1)


class _ReadOnlyGuard:
    """Authorizer callback plus an instruction budget."""

    def __init__(self) -> None:
        self.steps = 0
        self.denied: str | None = None

    def authorize(
        self,
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        trigger: str | None,
    ) -> int:
        """Allow only whitelisted read actions."""
        if action == _SQLITE_SELECT:
            return _SQLITE_OK
        if action == _SQLITE_READ:
            if arg1 in ALLOWED_TABLES:
                return _SQLITE_OK
            self.denied = f"reading table {arg1!r} is not permitted"
            return _SQLITE_DENY
        if action == _SQLITE_FUNCTION:
            fn = (arg2 or "").lower()
            if fn in ALLOWED_FUNCTIONS:
                return _SQLITE_OK
            self.denied = f"function {fn!r} is not permitted"
            return _SQLITE_DENY
        self.denied = f"SQL action {action} is not permitted; this database is read-only"
        return _SQLITE_DENY

    def progress(self) -> int:
        """Abort once the instruction budget is exhausted."""
        self.steps += _PROGRESS_INTERVAL
        return 1 if self.steps > MAX_VM_STEPS else 0


class SqliteEnvironment(ToolEnvironment):
    """Answer questions about a seeded synthetic sales database."""

    name = "sqlite"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._task: Task | None = None
        self._conn: sqlite3.Connection | None = None
        self._guard = _ReadOnlyGuard()
        self._steps = 0

    # -- tools ----------------------------------------------------------

    def tool_specs(self) -> list[ToolSpec]:
        """The two read-only database tools."""
        return [
            ToolSpec(
                name="schema",
                description="Return the CREATE TABLE statements for the database.",
                parameters={},
                required=(),
                example={},
            ),
            ToolSpec(
                name="query",
                description=(
                    f"Run one read-only SELECT statement and return up to {MAX_ROWS} rows as JSON."
                ),
                parameters={"sql": "a single SELECT statement"},
                required=("sql",),
                example={"sql": "SELECT count(*) FROM orders WHERE status = 'shipped'"},
            ),
        ]

    # -- episode ---------------------------------------------------------

    def reset(self, task: Task) -> Observation:
        """Rebuild the in-memory database for ``task``."""
        self.close()
        self._task = task
        self._steps = 0
        self._guard = _ReadOnlyGuard()
        self._conn = self._build_database(int(task.metadata["db_seed"]))
        return Observation(text=task.prompt, state_id="sql:0")

    def close(self) -> None:
        """Release the in-memory connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with contextlib.suppress(Exception):
            self.close()

    @staticmethod
    def _rows(seed: int) -> tuple[list[tuple[int, str, str]], list[tuple[int, int, int, str]]]:
        rng = random.Random(f"sqlite:{seed}")
        n_customers = rng.randrange(4, 8)
        customers = [
            (
                i + 1,
                _NAMES[(seed + i * 5) % len(_NAMES)] + str(i + 1),
                _CITIES[rng.randrange(len(_CITIES))],
            )
            for i in range(n_customers)
        ]
        orders: list[tuple[int, int, int, str]] = []
        for order_id in range(1, rng.randrange(9, 18)):
            orders.append(
                (
                    order_id,
                    rng.randrange(1, n_customers + 1),
                    rng.randrange(10, 500),
                    _STATUSES[rng.randrange(len(_STATUSES))],
                )
            )
        return customers, orders

    def _build_database(self, seed: int) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        customers, orders = self._rows(seed)
        conn.executescript(SCHEMA_DDL)
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", customers)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
        conn.commit()
        # Lock the connection down only *after* the fixture is loaded.
        conn.set_authorizer(self._guard.authorize)
        conn.set_progress_handler(self._guard.progress, _PROGRESS_INTERVAL)
        return conn

    def step(self, call: ToolCall) -> StepResult:
        """Execute one read-only tool."""
        self._steps += 1
        state_id = f"sql:{self._steps}"
        if self._conn is None:
            raise ToolEnvironmentError("step() called before reset()")
        if call.name == "schema":
            return StepResult(ok=True, result=SCHEMA_DDL, state_id=state_id)
        if call.name == "query":
            sql = call.arguments.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                return StepResult(
                    ok=False,
                    error="'sql' must be a non-empty string",
                    state_id=state_id,
                    failure_category=FailureCategory.INVALID_TOOL_CALL,
                )
            try:
                rows = self._run_query(sql)
            except ToolEnvironmentError as exc:
                return StepResult(
                    ok=False,
                    error=exc.message,
                    state_id=state_id,
                    failure_category=FailureCategory.TOOL_ERROR,
                )
            return StepResult(
                ok=True,
                result=json.dumps(rows, ensure_ascii=False, sort_keys=False),
                state_id=state_id,
            )
        return StepResult(
            ok=False,
            error=f"unknown tool {call.name!r}; available tools: schema, query",
            state_id=state_id,
            failure_category=FailureCategory.UNKNOWN_TOOL,
        )

    def _run_query(self, sql: str) -> list[dict[str, Any]]:
        assert self._conn is not None
        text = sql.strip().rstrip(";").strip()
        if len(text) > MAX_SQL_CHARS:
            raise ToolEnvironmentError(
                f"query is {len(text)} characters, over the {MAX_SQL_CHARS} character limit"
            )
        if ";" in text:
            raise ToolEnvironmentError("only one SQL statement per call is allowed")
        if not text.lower().startswith(("select", "with")):
            raise ToolEnvironmentError("only SELECT statements are allowed")
        self._guard.steps = 0
        self._guard.denied = None
        try:
            cursor = self._conn.execute(text)
            rows = cursor.fetchmany(MAX_ROWS + 1)
            columns = [d[0] for d in (cursor.description or [])]
        except sqlite3.OperationalError as exc:
            if self._guard.denied:
                raise ToolEnvironmentError(self._guard.denied) from exc
            if "interrupted" in str(exc).lower():
                raise ToolEnvironmentError(
                    f"query exceeded the {MAX_VM_STEPS} instruction budget"
                ) from exc
            raise ToolEnvironmentError(f"SQL error: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            if self._guard.denied:
                raise ToolEnvironmentError(self._guard.denied) from exc
            raise ToolEnvironmentError(f"SQL error: {exc}") from exc
        truncated = len(rows) > MAX_ROWS
        out = [dict(zip(columns, row, strict=False)) for row in rows[:MAX_ROWS]]
        if truncated:
            out.append({"_note": f"result truncated to {MAX_ROWS} rows"})
        return out

    def verify(self, answer: str) -> VerificationResult:
        """Normalized comparison against the reference answer."""
        if self._task is None:
            raise ToolEnvironmentError("verify() called before reset()")
        expected = self._task.answer
        predicted = answer.strip().strip('"').strip()
        if _normalize(predicted) == _normalize(expected):
            return VerificationResult(
                solved=True, reward=1.0, expected=expected, predicted=predicted
            )
        return VerificationResult(
            solved=False,
            reward=0.0,
            expected=expected,
            predicted=predicted,
            failure_category=FailureCategory.WRONG_ANSWER,
            detail=f"expected {expected!r}, got {predicted!r}",
        )

    # -- tasks ------------------------------------------------------------

    def generate_task(self, index: int, seed: int, *, difficulty: str, split: str) -> Task:
        """Deterministically build one database question."""
        db_seed = seed * 6151 + index
        customers, orders = self._rows(db_seed)
        rng = random.Random(f"sqlite-task:{seed}:{difficulty}:{index}")
        # The database id is part of the prompt so a prompt identifies exactly
        # one instance, which keeps the splits provably disjoint.
        header = f"Database #{db_seed}."

        if difficulty == "easy":
            status = _STATUSES[rng.randrange(len(_STATUSES))]
            answer = str(sum(1 for o in orders if o[3] == status))
            prompt = (
                f"{header} How many rows in the orders table have status "
                f"'{status}'? Report the count."
            )
            sql = f"SELECT count(*) AS n FROM orders WHERE status = '{status}'"
            kind = "count"
        elif difficulty == "medium":
            customer = customers[rng.randrange(len(customers))]
            total = sum(o[2] for o in orders if o[1] == customer[0])
            answer = str(total)
            prompt = (
                f"{header} What is the total order amount for the customer named "
                f"'{customer[1]}'? Report the number."
            )
            sql = (
                "SELECT total(o.amount) AS total FROM orders o "
                "JOIN customers c ON c.id = o.customer_id "
                f"WHERE c.name = '{customer[1]}'"
            )
            kind = "join_sum"
        else:
            by_city: dict[str, int] = {}
            city_of = {c[0]: c[2] for c in customers}
            for order in orders:
                city = city_of[order[1]]
                by_city[city] = by_city.get(city, 0) + order[2]
            best = sorted(by_city.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            answer = best[0]
            prompt = (
                f"{header} Which city has the highest total order amount? "
                "Report the city name only (ties broken alphabetically)."
            )
            sql = (
                "SELECT c.city AS city, total(o.amount) AS total FROM orders o "
                "JOIN customers c ON c.id = o.customer_id "
                "GROUP BY c.city ORDER BY total DESC, c.city ASC LIMIT 1"
            )
            kind = "group_by"

        return Task(
            task_id=f"sql-{split}-{index}",
            prompt=prompt,
            answer=answer,
            difficulty=difficulty,
            split=split,
            metadata={"kind": kind, "db_seed": db_seed, "reference_sql": sql},
        )

    def oracle_actions(self, task: Task) -> list[OracleAction]:
        """Reference query sequence."""
        actions = []
        if task.difficulty != "easy":
            actions.append(
                OracleAction(OracleActionKind.TOOL_CALL, tool_name="schema", arguments={})
            )
        actions.append(
            OracleAction(
                OracleActionKind.TOOL_CALL,
                tool_name="query",
                arguments={"sql": str(task.metadata["reference_sql"])},
            )
        )
        actions.append(OracleAction(OracleActionKind.FINAL, answer=task.answer))
        return actions

    def privileged_context(self, task: Task) -> str | None:
        """Reveal the reference SQL and answer to the teacher only."""
        return (
            f"Verified reference SQL: {task.metadata['reference_sql']} "
            f"-> the answer is {task.answer}."
        )


def _normalize(text: str) -> str:
    """Case- and format-insensitive comparison key for answers."""
    cleaned = text.strip().lower().replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:.4f}"
