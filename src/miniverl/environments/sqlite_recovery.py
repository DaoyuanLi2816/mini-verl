"""RecoveryBench: deterministic recovery from SQLite tool errors.

This environment is separate from :mod:`miniverl.environments.sqlite_env` so
the published legacy SQLite task semantics remain unchanged.  Every database
is in memory, every identifier comes from the registry below, and SQLite's
authorizer enforces the read-only boundary after fixture construction.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import re
import sqlite3
from dataclasses import asdict, dataclass
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
from miniverl.environments.sqlite_env import (
    ALLOWED_FUNCTIONS,
    MAX_ROWS,
    MAX_SQL_CHARS,
    MAX_VM_STEPS,
)
from miniverl.errors import ToolEnvironmentError

__all__ = [
    "RECOVERY_SCHEMA_TEMPLATES",
    "RecoverySchemaTemplate",
    "SqliteRecoveryEnvironment",
    "recovery_template_registry_digest",
]

TEMPLATE_REGISTRY_VERSION = 1
_PROGRESS_INTERVAL = 1000
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_LABELS = ("Aster", "Birch", "Cedar", "Dahlia", "Elm", "Fir", "Grove", "Hazel")
_CATEGORIES = ("amber", "blue", "coral", "green")
_STATES = ("active", "pending", "closed")

_SQLITE_SELECT = getattr(sqlite3, "SQLITE_SELECT", 21)
_SQLITE_READ = getattr(sqlite3, "SQLITE_READ", 20)
_SQLITE_FUNCTION = getattr(sqlite3, "SQLITE_FUNCTION", 31)
_SQLITE_OK = getattr(sqlite3, "SQLITE_OK", 0)
_SQLITE_DENY = getattr(sqlite3, "SQLITE_DENY", 1)


@dataclass(frozen=True)
class RecoverySchemaTemplate:
    """One safe, versioned schema family assigned to exactly one split."""

    template_id: str
    split: str
    entity_table: str
    event_table: str
    entity_id: str
    label_column: str
    category_column: str
    event_id: str
    join_key: str
    amount_column: str
    state_column: str
    entity_decoy_column: str
    event_decoy_column: str
    relationship_layout: str = "direct_fk"
    link_table: str | None = None
    link_event_key: str | None = None
    link_entity_key: str | None = None
    version: int = TEMPLATE_REGISTRY_VERSION

    def __post_init__(self) -> None:
        if self.split not in {"train", "eval", "test"}:
            raise ValueError(f"invalid template split {self.split!r}")
        if self.relationship_layout not in {"direct_fk", "association_table"}:
            raise ValueError(f"invalid relationship layout {self.relationship_layout!r}")
        if self.relationship_layout == "association_table" and not all(
            (self.link_table, self.link_event_key, self.link_entity_key)
        ):
            raise ValueError("association-table templates require all link identifiers")
        for identifier in self.identifiers:
            if not _SAFE_IDENTIFIER.fullmatch(identifier):
                raise ValueError(f"unsafe internal SQL identifier {identifier!r}")

    @property
    def identifiers(self) -> tuple[str, ...]:
        values = (
            self.entity_table,
            self.event_table,
            self.entity_id,
            self.label_column,
            self.category_column,
            self.event_id,
            self.join_key,
            self.amount_column,
            self.state_column,
            self.entity_decoy_column,
            self.event_decoy_column,
            self.link_table,
            self.link_event_key,
            self.link_entity_key,
        )
        return tuple(value for value in values if value is not None)

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def tables(self) -> tuple[str, ...]:
        values = (self.entity_table, self.event_table, self.link_table)
        return tuple(value for value in values if value is not None)


def _template(
    template_id: str,
    split: str,
    entity_table: str,
    event_table: str,
    entity_id: str,
    label: str,
    category: str,
    event_id: str,
    join_key: str,
    amount: str,
    state: str,
    entity_decoy: str,
    event_decoy: str,
    *,
    association: tuple[str, str, str] | None = None,
) -> RecoverySchemaTemplate:
    return RecoverySchemaTemplate(
        template_id=template_id,
        split=split,
        entity_table=entity_table,
        event_table=event_table,
        entity_id=entity_id,
        label_column=label,
        category_column=category,
        event_id=event_id,
        join_key=join_key,
        amount_column=amount,
        state_column=state,
        entity_decoy_column=entity_decoy,
        event_decoy_column=event_decoy,
        relationship_layout="association_table" if association else "direct_fk",
        link_table=association[0] if association else None,
        link_event_key=association[1] if association else None,
        link_entity_key=association[2] if association else None,
    )


RECOVERY_SCHEMA_TEMPLATES: tuple[RecoverySchemaTemplate, ...] = (
    _template(
        "train_customers_orders",
        "train",
        "customers",
        "orders",
        "customer_id",
        "customer_name",
        "region",
        "order_id",
        "buyer_ref",
        "order_total",
        "order_state",
        "segment_note",
        "priority_code",
    ),
    _template(
        "train_members_bookings",
        "train",
        "members",
        "bookings",
        "member_id",
        "display_name",
        "chapter",
        "booking_id",
        "member_ref",
        "fee_amount",
        "booking_status",
        "signup_channel",
        "seat_zone",
    ),
    _template(
        "train_accounts_transactions",
        "train",
        "accounts",
        "transactions",
        "account_id",
        "account_label",
        "portfolio",
        "transaction_id",
        "account_ref",
        "net_amount",
        "transaction_state",
        "risk_band",
        "batch_code",
        association=("account_transaction_links", "transaction_ref", "account_ref"),
    ),
    _template(
        "train_vendors_invoices",
        "train",
        "vendors",
        "invoices",
        "vendor_id",
        "vendor_name",
        "market",
        "invoice_id",
        "vendor_ref",
        "invoice_value",
        "invoice_status",
        "tier_note",
        "ledger_code",
    ),
    _template(
        "eval_users_subscriptions",
        "eval",
        "users",
        "subscriptions",
        "user_id",
        "user_alias",
        "cohort",
        "subscription_id",
        "subscriber_ref",
        "monthly_value",
        "subscription_state",
        "locale_note",
        "plan_code",
    ),
    _template(
        "eval_patients_appointments",
        "eval",
        "patients",
        "appointments",
        "patient_id",
        "patient_alias",
        "clinic_group",
        "appointment_id",
        "patient_ref",
        "visit_cost",
        "appointment_state",
        "intake_note",
        "room_code",
        association=("patient_appointment_map", "appointment_ref", "patient_ref"),
    ),
    _template(
        "eval_teams_matches",
        "eval",
        "teams",
        "matches",
        "team_id",
        "team_name",
        "division",
        "match_id",
        "team_ref",
        "score_value",
        "match_state",
        "seed_note",
        "venue_code",
    ),
    _template(
        "eval_creators_royalties",
        "eval",
        "creators",
        "royalties",
        "creator_id",
        "creator_name",
        "genre_group",
        "royalty_id",
        "creator_ref",
        "royalty_amount",
        "royalty_state",
        "catalog_note",
        "territory_code",
    ),
    _template(
        "test_readers_loans",
        "test",
        "readers",
        "loans",
        "reader_id",
        "reader_name",
        "library_zone",
        "loan_id",
        "reader_ref",
        "late_fee",
        "loan_state",
        "membership_note",
        "shelf_code",
    ),
    _template(
        "test_hosts_reservations",
        "test",
        "hosts",
        "reservations",
        "host_id",
        "host_alias",
        "area_group",
        "reservation_id",
        "host_ref",
        "booking_value",
        "reservation_state",
        "license_note",
        "unit_code",
        association=("host_reservation_links", "reservation_ref", "host_ref"),
    ),
    _template(
        "test_merchants_payments",
        "test",
        "merchants",
        "payments",
        "merchant_id",
        "merchant_name",
        "sector",
        "payment_id",
        "merchant_ref",
        "payment_amount",
        "payment_state",
        "channel_note",
        "terminal_code",
    ),
    _template(
        "test_projects_expenses",
        "test",
        "projects",
        "expenses",
        "project_id",
        "project_name",
        "program",
        "expense_id",
        "project_ref",
        "expense_amount",
        "expense_state",
        "owner_note",
        "cost_code",
        association=("project_expense_map", "expense_ref", "project_ref"),
    ),
)


def recovery_template_registry_digest() -> str:
    payload = [asdict(template) for template in RECOVERY_SCHEMA_TEMPLATES]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _RecoveryReadOnlyGuard:
    def __init__(self, tables: tuple[str, ...]) -> None:
        self.tables = frozenset(tables)
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
        del db_name, trigger
        if action == _SQLITE_SELECT:
            return _SQLITE_OK
        if action == _SQLITE_READ:
            if arg1 in self.tables:
                return _SQLITE_OK
            self.denied = f"reading table {arg1!r} is not permitted"
            return _SQLITE_DENY
        if action == _SQLITE_FUNCTION:
            function = (arg2 or "").lower()
            if function in ALLOWED_FUNCTIONS:
                return _SQLITE_OK
            self.denied = f"function {function!r} is not permitted"
            return _SQLITE_DENY
        self.denied = f"SQL action {action} is not permitted; this database is read-only"
        return _SQLITE_DENY

    def progress(self) -> int:
        self.steps += _PROGRESS_INTERVAL
        return int(self.steps > MAX_VM_STEPS)


class SqliteRecoveryEnvironment(ToolEnvironment):
    """Versioned SQLite recovery tasks with explicit error interventions."""

    name = "sqlite_recovery"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._task: Task | None = None
        self._template: RecoverySchemaTemplate | None = None
        self._conn: sqlite3.Connection | None = None
        self._guard: _RecoveryReadOnlyGuard | None = None
        self._schema_ddl = ""
        self._steps = 0
        self._controlled_error_pending = False

    def tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="schema",
                description="Return the current in-memory database schema.",
                example={},
            ),
            ToolSpec(
                name="query",
                description=f"Run one read-only SELECT and return up to {MAX_ROWS} JSON rows.",
                parameters={"sql": "one SELECT or read-only WITH statement"},
                required=("sql",),
                example={"sql": "SELECT count(*) AS n FROM records"},
            ),
        ]

    def final_answer_example(self) -> str:
        return "4"

    def reset(self, task: Task) -> Observation:
        self.close()
        self._task = task
        self._steps = 0
        self._controlled_error_pending = (
            task.metadata.get("intervention_kind") == "controlled_schema_refresh"
        )
        self._template = self._template_by_id(str(task.metadata["template_id"]))
        self._guard = _RecoveryReadOnlyGuard(self._template.tables)
        self._conn, self._schema_ddl = self._build_database(
            self._template, int(task.metadata["database_seed"])
        )
        return Observation(text=task.prompt, state_id="recovery:0")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover
        with contextlib.suppress(Exception):
            self.close()

    @staticmethod
    def _template_by_id(template_id: str) -> RecoverySchemaTemplate:
        for template in RECOVERY_SCHEMA_TEMPLATES:
            if template.template_id == template_id:
                return template
        raise ToolEnvironmentError(f"unknown recovery template {template_id!r}")

    @staticmethod
    def _templates_for(split: str) -> tuple[RecoverySchemaTemplate, ...]:
        templates = tuple(
            template for template in RECOVERY_SCHEMA_TEMPLATES if template.split == split
        )
        if not templates:
            raise ToolEnvironmentError(f"no recovery templates registered for split {split!r}")
        return templates

    def _build_database(
        self, template: RecoverySchemaTemplate, seed: int
    ) -> tuple[sqlite3.Connection, str]:
        rng = random.Random(f"recovery-db:{template.digest}:{seed}")
        entity_rows = [
            (
                index,
                f"{_LABELS[(seed + index * 3) % len(_LABELS)]}-{index}",
                _CATEGORIES[rng.randrange(len(_CATEGORIES))],
                f"note-{rng.randrange(100, 999)}",
            )
            for index in range(1, 9)
        ]
        event_rows: list[tuple[Any, ...]] = []
        links: list[tuple[int, int]] = []
        for event_id in range(1, 25):
            entity_id = rng.randrange(1, len(entity_rows) + 1)
            common = (
                event_id,
                rng.randrange(15, 700),
                _STATES[rng.randrange(len(_STATES))],
                f"code-{rng.randrange(10, 99)}",
            )
            if template.relationship_layout == "direct_fk":
                event_rows.append((common[0], entity_id, common[1], common[2], common[3]))
            else:
                event_rows.append(common)
                links.append((event_id, entity_id))

        entity_ddl = (
            f"CREATE TABLE {template.entity_table} (\n"
            f"    {template.entity_id} INTEGER PRIMARY KEY,\n"
            f"    {template.label_column} TEXT NOT NULL,\n"
            f"    {template.category_column} TEXT NOT NULL,\n"
            f"    {template.entity_decoy_column} TEXT NOT NULL\n"
            ");"
        )
        if template.relationship_layout == "direct_fk":
            event_ddl = (
                f"CREATE TABLE {template.event_table} (\n"
                f"    {template.event_id} INTEGER PRIMARY KEY,\n"
                f"    {template.join_key} INTEGER NOT NULL,\n"
                f"    {template.amount_column} INTEGER NOT NULL,\n"
                f"    {template.state_column} TEXT NOT NULL,\n"
                f"    {template.event_decoy_column} TEXT NOT NULL\n"
                ");"
            )
            link_ddl = ""
        else:
            event_ddl = (
                f"CREATE TABLE {template.event_table} (\n"
                f"    {template.event_id} INTEGER PRIMARY KEY,\n"
                f"    {template.amount_column} INTEGER NOT NULL,\n"
                f"    {template.state_column} TEXT NOT NULL,\n"
                f"    {template.event_decoy_column} TEXT NOT NULL\n"
                ");"
            )
            link_ddl = (
                f"CREATE TABLE {template.link_table} (\n"
                f"    {template.link_event_key} INTEGER NOT NULL,\n"
                f"    {template.link_entity_key} INTEGER NOT NULL\n"
                ");"
            )
        schema = "\n".join(part for part in (entity_ddl, event_ddl, link_ddl) if part)
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        conn.executemany(f"INSERT INTO {template.entity_table} VALUES (?, ?, ?, ?)", entity_rows)
        placeholders = ", ".join("?" for _ in event_rows[0])
        conn.executemany(f"INSERT INTO {template.event_table} VALUES ({placeholders})", event_rows)
        if links:
            conn.executemany(f"INSERT INTO {template.link_table} VALUES (?, ?)", links)
        conn.commit()
        assert self._guard is not None
        conn.set_authorizer(self._guard.authorize)
        conn.set_progress_handler(self._guard.progress, _PROGRESS_INTERVAL)
        return conn, schema

    def step(self, call: ToolCall) -> StepResult:
        self._steps += 1
        state_id = f"recovery:{self._steps}"
        if self._conn is None or self._task is None:
            raise ToolEnvironmentError("step() called before reset()")
        if call.name == "schema":
            return StepResult(ok=True, result=self._schema_ddl, state_id=state_id)
        if call.name != "query":
            return StepResult(
                ok=False,
                error=f"unknown tool {call.name!r}; available tools: schema, query",
                state_id=state_id,
                failure_category=FailureCategory.UNKNOWN_TOOL,
                error_code="UNKNOWN_TOOL",
                retryable=False,
            )
        sql = call.arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return StepResult(
                ok=False,
                error="'sql' must be a non-empty string",
                state_id=state_id,
                failure_category=FailureCategory.INVALID_TOOL_CALL,
                error_code="INVALID_SQL_CALL",
                retryable=False,
            )
        try:
            rows = self._run_query(sql)
        except ToolEnvironmentError as exc:
            return StepResult(
                ok=False,
                error=exc.message,
                state_id=state_id,
                failure_category=FailureCategory.TOOL_ERROR,
                error_code="SQL_EXECUTION_ERROR",
                retryable=True,
                intervention=False,
                tool_result_metadata={"source": "sqlite"},
            )
        if self._controlled_error_pending:
            self._controlled_error_pending = False
            return StepResult(
                ok=False,
                error="The schema view changed. Inspect the current schema and retry.",
                state_id=state_id,
                failure_category=FailureCategory.TOOL_ERROR,
                error_code="SCHEMA_REFRESH_REQUIRED",
                retryable=True,
                intervention=True,
                tool_result_metadata={
                    "intervention_kind": "controlled_schema_refresh",
                    "occurrence": 1,
                },
            )
        try:
            result = json.dumps(rows, ensure_ascii=False, allow_nan=False)
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            return StepResult(
                ok=False,
                error=f"query result is not finite JSON: {exc}",
                state_id=state_id,
                failure_category=FailureCategory.TOOL_ERROR,
                error_code="RESULT_ENCODING_ERROR",
                retryable=False,
            )
        return StepResult(ok=True, result=result, state_id=state_id)

    def _run_query(self, sql: str) -> list[dict[str, Any]]:
        assert self._conn is not None and self._guard is not None
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
            columns = [item[0] for item in (cursor.description or [])]
        except sqlite3.DatabaseError as exc:
            if self._guard.denied:
                raise ToolEnvironmentError(self._guard.denied) from exc
            if "interrupted" in str(exc).lower():
                raise ToolEnvironmentError(
                    f"query exceeded the {MAX_VM_STEPS} instruction budget"
                ) from exc
            raise ToolEnvironmentError(f"SQL error: {exc}") from exc
        output = [dict(zip(columns, row, strict=False)) for row in rows[:MAX_ROWS]]
        if len(rows) > MAX_ROWS:
            output.append({"_note": f"result truncated to {MAX_ROWS} rows"})
        return output

    def generate_task(self, index: int, seed: int, *, difficulty: str, split: str) -> Task:
        templates = self._templates_for(split)
        template = templates[index % len(templates)]
        database_seed = seed * 1_000_003 + index * 97 + (index % len(templates))
        conn_guard = self._guard
        self._guard = _RecoveryReadOnlyGuard(template.tables)
        temporary, _ = self._build_database(template, database_seed)
        try:
            entity_rows = temporary.execute(
                f"SELECT {template.entity_id}, {template.label_column}, "
                f"{template.category_column} FROM {template.entity_table} ORDER BY {template.entity_id}"
            ).fetchall()
            event_rows = temporary.execute(
                f"SELECT {template.event_id}, {template.amount_column}, "
                f"{template.state_column} FROM {template.event_table} ORDER BY {template.event_id}"
            ).fetchall()
        finally:
            temporary.close()
            self._guard = conn_guard
        rng = random.Random(f"recovery-task:{template.digest}:{seed}:{difficulty}:{index}")
        task_kind = ("filter_count", "lookup_join_sum", "group_rank", "multi_filter")[index % 4]
        join = self._join_sql(template)
        if task_kind == "filter_count":
            state = _STATES[rng.randrange(len(_STATES))]
            answer = str(sum(1 for row in event_rows if row[2] == state))
            reference_sql = (
                f"SELECT count(*) AS answer FROM {template.event_table} "
                f"WHERE {template.state_column} = '{state}'"
            )
            question = (
                f"How many {template.event_table} rows have {template.state_column} '{state}'?"
            )
        elif task_kind == "lookup_join_sum":
            chosen = entity_rows[rng.randrange(len(entity_rows))]
            reference_sql = (
                f"SELECT total(e.{template.amount_column}) AS answer {join} "
                f"WHERE p.{template.label_column} = '{chosen[1]}'"
            )
            answer = str(self._scalar(temporary_seed=(template, database_seed), sql=reference_sql))
            question = (
                f"What is the total {template.amount_column} for {template.label_column} "
                f"'{chosen[1]}'?"
            )
        elif task_kind == "group_rank":
            reference_sql = (
                f"SELECT p.{template.category_column} AS answer, "
                f"total(e.{template.amount_column}) AS total_value {join} "
                f"GROUP BY p.{template.category_column} "
                "ORDER BY total_value DESC, answer ASC LIMIT 1"
            )
            answer = str(self._scalar(temporary_seed=(template, database_seed), sql=reference_sql))
            question = (
                f"Which {template.category_column} has the highest total "
                f"{template.amount_column}? Report the category only."
            )
        else:
            category = _CATEGORIES[rng.randrange(len(_CATEGORIES))]
            state = _STATES[rng.randrange(len(_STATES))]
            reference_sql = (
                f"SELECT count(*) AS answer {join} "
                f"WHERE p.{template.category_column} = '{category}' "
                f"AND e.{template.state_column} = '{state}'"
            )
            answer = str(self._scalar(temporary_seed=(template, database_seed), sql=reference_sql))
            question = (
                f"How many {template.event_table} rows belong to {template.category_column} "
                f"'{category}' and have {template.state_column} '{state}'?"
            )
        intervention_kind = (
            "controlled_schema_refresh",
            "natural_sql_error",
            "none",
        )[index % 3]
        expected_sequence = {
            "controlled_schema_refresh": "schema-query_error-schema-corrected_query-final",
            "natural_sql_error": "query_error-schema-corrected_query-final",
            "none": "schema-query-final",
        }[intervention_kind]
        natural_error_sql = f"SELECT stale_missing_column FROM {template.event_table}"
        prompt = (
            f"Recovery database {database_seed} uses template {template.template_id}. "
            f"{question} Use the database tools; report only the final value."
        )
        return Task(
            task_id=f"sql-recovery-{split}-{index}",
            prompt=prompt,
            answer=answer,
            difficulty=difficulty,
            split=split,
            metadata={
                "template_id": template.template_id,
                "template_version": template.version,
                "template_digest": template.digest,
                "template_registry_digest": recovery_template_registry_digest(),
                "database_seed": database_seed,
                "db_seed": database_seed,
                "task_kind": task_kind,
                "intervention_kind": intervention_kind,
                "expected_tool_sequence_class": expected_sequence,
                "reference_sql": reference_sql,
                "natural_error_sql": natural_error_sql,
            },
        )

    def _scalar(self, *, temporary_seed: tuple[RecoverySchemaTemplate, int], sql: str) -> Any:
        template, seed = temporary_seed
        old_guard = self._guard
        self._guard = _RecoveryReadOnlyGuard(template.tables)
        conn, _ = self._build_database(template, seed)
        try:
            row = conn.execute(sql).fetchone()
            if row is None:
                raise ToolEnvironmentError("reference query returned no rows")
            return row[0]
        finally:
            conn.close()
            self._guard = old_guard

    @staticmethod
    def _join_sql(template: RecoverySchemaTemplate) -> str:
        if template.relationship_layout == "direct_fk":
            return (
                f"FROM {template.event_table} e JOIN {template.entity_table} p "
                f"ON p.{template.entity_id} = e.{template.join_key}"
            )
        return (
            f"FROM {template.event_table} e JOIN {template.link_table} l "
            f"ON l.{template.link_event_key} = e.{template.event_id} "
            f"JOIN {template.entity_table} p "
            f"ON p.{template.entity_id} = l.{template.link_entity_key}"
        )

    def verify(self, answer: str) -> VerificationResult:
        if self._task is None:
            raise ToolEnvironmentError("verify() called before reset()")
        expected = self._task.answer
        predicted = answer.strip().strip('"').strip()
        normalized_predicted = _normalize(predicted)
        normalized_expected = _normalize(expected)
        if normalized_predicted is None:
            return VerificationResult(
                solved=False,
                reward=0.0,
                expected=expected,
                predicted=predicted,
                failure_category=FailureCategory.MALFORMED_ANSWER,
                detail="answer is not finite",
            )
        if normalized_predicted == normalized_expected:
            return VerificationResult(True, 1.0, expected, predicted)
        return VerificationResult(
            False,
            0.0,
            expected,
            predicted,
            failure_category=FailureCategory.WRONG_ANSWER,
            detail=f"expected {expected!r}, got {predicted!r}",
        )

    def oracle_actions(self, task: Task) -> list[OracleAction]:
        reference = OracleAction(
            OracleActionKind.TOOL_CALL,
            tool_name="query",
            arguments={"sql": str(task.metadata["reference_sql"])},
        )
        schema = OracleAction(OracleActionKind.TOOL_CALL, tool_name="schema", arguments={})
        intervention = task.metadata["intervention_kind"]
        if intervention == "controlled_schema_refresh":
            actions = [schema, reference, schema, reference]
        elif intervention == "natural_sql_error":
            actions = [
                OracleAction(
                    OracleActionKind.TOOL_CALL,
                    tool_name="query",
                    arguments={"sql": str(task.metadata["natural_error_sql"])},
                ),
                schema,
                reference,
            ]
        else:
            actions = [schema, reference]
        return [*actions, OracleAction(OracleActionKind.FINAL, answer=task.answer)]

    def privileged_context(self, task: Task) -> str | None:
        return (
            f"Template {task.metadata['template_id']} reference SQL: "
            f"{task.metadata['reference_sql']} -> answer {task.answer}. "
            f"Expected sequence: {task.metadata['expected_tool_sequence_class']}."
        )

    def trajectory_metadata(self, task: Task) -> dict[str, Any]:
        keys = (
            "template_id",
            "template_version",
            "template_digest",
            "template_registry_digest",
            "database_seed",
            "task_kind",
            "intervention_kind",
            "expected_tool_sequence_class",
        )
        return {key: task.metadata[key] for key in keys}


def _normalize(text: str) -> str | None:
    cleaned = text.strip().lower().replace(",", "")
    try:
        value = float(cleaned)
    except (OverflowError, ValueError):
        value = None
    if value is None:
        if cleaned in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
            return None
        return cleaned
    if not math.isfinite(value):
        return None
    if abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:.4f}"
