"""Bounded reader and prompt renderer for the supported verl Parquet subset."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from miniverl.config.models import PromptTruncation, VerlParquetSourceConfig
from miniverl.errors import ConfigError, MissingDependencyError

SplitName = Literal["train", "val"]
_PRESERVED_FIELDS = ("data_source", "ability", "reward_model", "extra_info")


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Parquet row contains metadata that is not canonical JSON: {exc}",
            hint="use JSON-compatible scalars, lists and mappings in preserved metadata fields",
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise MissingDependencyError("pyarrow", "train", "verl Parquet prompt loading") from exc
    return pq


@dataclass(frozen=True)
class PromptRecord:
    """One validated prompt row with its complete supported provenance."""

    prompt: str | list[dict[str, str]]
    data_source: Any
    ability: Any
    reward_model: Any
    extra_info: Any
    source_file: str
    source_row_index: int
    row_digest: str
    canonical_payload: str


@dataclass(frozen=True)
class PromptDatasetManifest:
    """Content and schema identity obtained through a bounded scan."""

    rows: dict[str, int]
    schema_digest: str
    content_digest: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class RenderedPrompt:
    """The single actor-rendered prompt shared by actor and teacher scoring."""

    record: PromptRecord
    text: str
    token_ids: tuple[int, ...]
    tokenizer_identity: dict[str, Any]
    rendered_prompt_digest: str
    prompt_token_count: int
    truncation_decision: str
    original_prompt_token_count: int


class VerlParquetDataset:
    """Read prompt rows by record batch; never load a complete table."""

    def __init__(self, config: VerlParquetSourceConfig) -> None:
        self.config = config

    def _files(self, split: SplitName) -> list[Path]:
        raw = self.config.train_files if split == "train" else self.config.val_files
        paths = [Path(item).resolve() for item in raw]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ConfigError(
                f"{split} Parquet file not found: {missing[0]}",
                hint="resolve recipe-relative paths before constructing the dataset",
            )
        return paths

    def _records(self, split: SplitName) -> Iterator[PromptRecord]:
        pq = _parquet()
        prompt_key = self.config.prompt_key
        for path in self._files(split):
            try:
                parquet = pq.ParquetFile(path)
            except Exception as exc:
                raise ConfigError(f"cannot open Parquet file {path}: {exc}") from exc
            if prompt_key not in parquet.schema_arrow.names:
                raise ConfigError(
                    f"Parquet file {path} has no prompt field {prompt_key!r}",
                    hint=f"available columns: {', '.join(parquet.schema_arrow.names)}",
                )
            source_index = 0
            columns = [
                prompt_key,
                *[name for name in _PRESERVED_FIELDS if name in parquet.schema_arrow.names],
            ]
            try:
                batches = parquet.iter_batches(
                    batch_size=self.config.row_batch_size, columns=columns
                )
                for batch in batches:
                    for row in batch.to_pylist():
                        yield self._validate_row(row, path=path, source_index=source_index)
                        source_index += 1
            except ConfigError:
                raise
            except Exception as exc:
                raise ConfigError(
                    f"failed reading {path} near row {source_index}: {exc}",
                    hint="no rows were silently skipped",
                ) from exc

    def _validate_row(self, row: dict[str, Any], *, path: Path, source_index: int) -> PromptRecord:
        prompt = row.get(self.config.prompt_key)
        location = f"{path} row {source_index}"
        if isinstance(prompt, str):
            if not self.config.allow_plain_string_prompts:
                raise ConfigError(
                    f"{location} contains a plain-string prompt without explicit opt-in",
                    hint="set source.allow_plain_string_prompts=true or store a chat message list",
                )
            if not prompt:
                raise ConfigError(f"{location} contains an empty prompt")
            validated: str | list[dict[str, str]] = prompt
        elif isinstance(prompt, list) and prompt:
            messages: list[dict[str, str]] = []
            for index, message in enumerate(prompt):
                if not isinstance(message, dict):
                    raise ConfigError(f"{location} prompt message {index} is not a mapping")
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or not role or not isinstance(content, str):
                    raise ConfigError(
                        f"{location} prompt message {index} requires string role and content"
                    )
                messages.append({"role": role, "content": content})
            validated = messages
        else:
            raise ConfigError(
                f"{location} field {self.config.prompt_key!r} must be a non-empty message list"
                " or an explicitly enabled plain string"
            )
        preserved = {name: row.get(name) for name in _PRESERVED_FIELDS}
        if self.config.use_task_rewards and preserved["reward_model"] is None:
            raise ConfigError(
                f"{location} has no reward_model but source.use_task_rewards=true",
                hint="provide reward_model per row or disable task rewards for pure OPD",
            )
        payload = {"prompt": validated, **preserved}
        canonical = _canonical(payload)
        return PromptRecord(
            prompt=validated,
            source_file=str(path),
            source_row_index=source_index,
            row_digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            canonical_payload=canonical,
            **preserved,
        )

    def iter_split(self, split: SplitName, *, epoch: int = 0) -> Iterator[PromptRecord]:
        """Yield all rows, optionally with a deterministic bounded-buffer shuffle."""
        records = self._records(split)
        if not self.config.shuffle:
            yield from records
            return
        rng = random.Random(self.config.seed ^ epoch ^ (0x56414C if split == "val" else 0))
        buffer: list[PromptRecord] = []
        for record in records:
            if len(buffer) < self.config.row_batch_size:
                buffer.append(record)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = record
        rng.shuffle(buffer)
        yield from buffer

    def inspect(self) -> PromptDatasetManifest:
        """Scan schemas and canonical row digests without retaining row payloads."""
        pq = _parquet()
        schema_items: list[dict[str, str]] = []
        content = hashlib.sha256()
        rows: dict[str, int] = {"train": 0, "val": 0}
        files: list[str] = []
        for split in ("train", "val"):
            typed_split: SplitName = split
            for path in self._files(typed_split):
                files.append(str(path))
                parquet = pq.ParquetFile(path)
                schema_items.append({"split": split, "schema": str(parquet.schema_arrow)})
            for record in self._records(typed_split):
                rows[split] += 1
                content.update(split.encode("ascii"))
                content.update(b"\0")
                content.update(record.row_digest.encode("ascii"))
                content.update(b"\n")
        return PromptDatasetManifest(
            rows=rows,
            schema_digest=_digest(schema_items),
            content_digest=content.hexdigest(),
            files=tuple(files),
        )


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    target = (
        tokenizer
        if callable(getattr(tokenizer, "apply_chat_template", None))
        else getattr(tokenizer, "_tok", None)
    )
    method = getattr(target, "apply_chat_template", None)
    if not callable(method):
        raise ConfigError(
            "the actor tokenizer has no chat template for message-list prompts",
            hint="use a tokenizer with apply_chat_template or explicitly supply plain strings",
        )
    try:
        rendered = method(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        rendered = method(messages)
    if not isinstance(rendered, str) or not rendered:
        raise ConfigError("the actor tokenizer chat template produced no text")
    return rendered


def render_prompt(
    record: PromptRecord,
    tokenizer: Any,
    config: VerlParquetSourceConfig,
) -> RenderedPrompt:
    """Render exactly once, then enforce the configured token bound exactly."""
    text = (
        _apply_chat_template(tokenizer, record.prompt)
        if isinstance(record.prompt, list)
        else record.prompt
    )
    token_ids = list(tokenizer.encode(text))
    original_count = len(token_ids)
    decision = "not_needed"
    if original_count > config.max_prompt_length:
        if config.truncation is PromptTruncation.ERROR:
            raise ConfigError(
                f"prompt {record.row_digest[:12]} has {original_count} tokens; "
                f"max_prompt_length={config.max_prompt_length}",
                hint="raise the bound or explicitly select source.truncation=left/right",
            )
        if config.truncation is PromptTruncation.LEFT:
            token_ids = token_ids[-config.max_prompt_length :]
        else:
            token_ids = token_ids[: config.max_prompt_length]
        text = tokenizer.decode(token_ids)
        decision = f"truncated_{config.truncation.value}"
    identity = dict(
        getattr(
            tokenizer,
            "identity",
            {"behavioral_fingerprint_v1": getattr(tokenizer, "fingerprint", "unknown")},
        )
    )
    return RenderedPrompt(
        record=record,
        text=text,
        token_ids=tuple(int(token) for token in token_ids),
        tokenizer_identity=identity,
        rendered_prompt_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        prompt_token_count=len(token_ids),
        truncation_decision=decision,
        original_prompt_token_count=original_count,
    )
