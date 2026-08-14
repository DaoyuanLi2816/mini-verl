"""Deterministic padded trajectory batches for selected-position training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from miniverl.losses.chunked import ChunkTargetProvider

__all__ = [
    "PaddedTrajectoryBatch",
    "build_padded_trajectory_batch",
    "concatenate_target_providers",
    "deterministic_length_batches",
    "deterministic_padded_token_batches",
    "normalize_trajectory_weights",
]


@dataclass(frozen=True)
class PaddedTrajectoryBatch:
    """A right-padded batch plus a flat map of selected prediction states.

    ``selected_offsets`` has length ``batch_size + 1``.  Its adjacent pairs
    delimit each trajectory's rows in the flattened selected-state tensor.
    Padding is represented only in ``input_ids``; selected positions are
    validated against the corresponding unpadded length.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: tuple[int, ...]
    selected_batch_indices: torch.Tensor
    selected_positions: torch.Tensor
    selected_offsets: tuple[int, ...]

    def with_input_ids(self, input_ids: torch.Tensor) -> PaddedTrajectoryBatch:
        """Return the same typed layout with replacement token values."""
        if input_ids.shape != self.input_ids.shape:
            raise ValueError(
                f"replacement input_ids shape {tuple(input_ids.shape)} does not match "
                f"{tuple(self.input_ids.shape)}"
            )
        return replace(self, input_ids=input_ids)

    @property
    def batch_size(self) -> int:
        """Number of trajectories."""
        return len(self.lengths)

    @property
    def num_selected_positions(self) -> int:
        """Number of flattened selected prediction positions."""
        return int(self.selected_positions.numel())


@dataclass(frozen=True)
class _ConcatenatedTargetProvider:
    """Flat view over trajectory-local target providers."""

    providers: tuple[ChunkTargetProvider, ...]
    offsets: tuple[int, ...]
    kind: str = "concatenated"

    def _pieces(
        self, start: int, end: int, student_logits: torch.Tensor, *, entropy: bool
    ) -> list[torch.Tensor]:
        output: list[torch.Tensor] = []
        consumed = 0
        for index, provider in enumerate(self.providers):
            provider_start = self.offsets[index]
            provider_end = self.offsets[index + 1]
            overlap_start = max(start, provider_start)
            overlap_end = min(end, provider_end)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - provider_start
            local_end = overlap_end - provider_start
            count = overlap_end - overlap_start
            if entropy:
                output.append(provider.teacher_entropy(local_start, local_end))
            else:
                logits = student_logits[consumed : consumed + count]
                output.append(provider.divergence(local_start, local_end, logits))
            consumed += count
        if consumed != end - start:
            raise ValueError(f"target-provider slice [{start}, {end}) covered {consumed} positions")
        return output

    def divergence(self, start: int, end: int, student_logits: torch.Tensor) -> torch.Tensor:
        """Delegate a global slice to its trajectory-local providers."""
        parts = self._pieces(start, end, student_logits, entropy=False)
        return torch.cat(parts) if parts else student_logits.new_zeros((0,), dtype=torch.float32)

    def teacher_entropy(self, start: int, end: int) -> torch.Tensor:
        """Delegate entropy collection using the same global-to-local map."""
        placeholder = torch.empty(end - start)
        parts = self._pieces(start, end, placeholder, entropy=True)
        return torch.cat(parts) if parts else placeholder.new_zeros((0,), dtype=torch.float32)


def build_padded_trajectory_batch(
    *,
    token_ids: Sequence[Sequence[int]],
    selected_positions: Sequence[Sequence[int]],
    pad_token_id: int,
    device: str | torch.device,
) -> PaddedTrajectoryBatch:
    """Right-pad trajectories and build the exact flat selected-state map."""
    if not token_ids:
        raise ValueError("a padded trajectory batch must contain at least one trajectory")
    if len(token_ids) != len(selected_positions):
        raise ValueError(
            f"selected_positions has {len(selected_positions)} rows for {len(token_ids)} trajectories"
        )
    lengths = tuple(len(sequence) for sequence in token_ids)
    if any(length < 1 for length in lengths):
        raise ValueError("padded trajectory batches cannot contain empty token sequences")
    max_length = max(lengths)
    ids = torch.full(
        (len(token_ids), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros((len(token_ids), max_length), dtype=torch.bool, device=device)
    flat_batches: list[int] = []
    flat_positions: list[int] = []
    offsets = [0]
    for batch_index, (sequence, positions, length) in enumerate(
        zip(token_ids, selected_positions, lengths, strict=True)
    ):
        ids[batch_index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[batch_index, :length] = True
        previous = -1
        for position in positions:
            if position < 0 or position >= length:
                raise ValueError(
                    f"selected position {position} is outside trajectory {batch_index} "
                    f"with length {length}"
                )
            if position <= previous:
                raise ValueError(
                    f"selected positions for trajectory {batch_index} must be strictly increasing"
                )
            previous = int(position)
            flat_batches.append(batch_index)
            flat_positions.append(int(position))
        offsets.append(len(flat_positions))
    return PaddedTrajectoryBatch(
        input_ids=ids,
        attention_mask=mask,
        lengths=lengths,
        selected_batch_indices=torch.tensor(flat_batches, dtype=torch.long, device=device),
        selected_positions=torch.tensor(flat_positions, dtype=torch.long, device=device),
        selected_offsets=tuple(offsets),
    )


def deterministic_length_batches(
    lengths: Sequence[int], *, batch_size: int
) -> tuple[tuple[int, ...], ...]:
    """Stable shortest-first bucketing, returning original trajectory indices."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if any(length < 1 for length in lengths):
        raise ValueError("trajectory lengths must all be positive")
    order = sorted(range(len(lengths)), key=lambda index: (int(lengths[index]), index))
    return tuple(
        tuple(order[start : start + batch_size]) for start in range(0, len(order), batch_size)
    )


def deterministic_padded_token_batches(
    lengths: Sequence[int],
    *,
    batch_size: int,
    max_padded_tokens: int | None,
    sort_by_length: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Stable shortest-first batches bounded by count and padded token footprint."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if any(length < 1 for length in lengths):
        raise ValueError("trajectory lengths must all be positive")
    if max_padded_tokens is not None and max_padded_tokens < 1:
        raise ValueError(f"max_padded_tokens must be >= 1, got {max_padded_tokens}")

    order = (
        sorted(range(len(lengths)), key=lambda index: (int(lengths[index]), index))
        if sort_by_length
        else list(range(len(lengths)))
    )
    batches: list[tuple[int, ...]] = []
    pending: list[int] = []
    for index in order:
        length = int(lengths[index])
        if max_padded_tokens is not None and length > max_padded_tokens:
            raise ValueError(
                f"trajectory {index} has {length} tokens, exceeding the physical update "
                f"limit of {max_padded_tokens} padded tokens"
            )
        candidate_size = len(pending) + 1
        exceeds_count = candidate_size > batch_size
        exceeds_tokens = (
            max_padded_tokens is not None and length * candidate_size > max_padded_tokens
        )
        if pending and (exceeds_count or exceeds_tokens):
            batches.append(tuple(pending))
            pending = []
        pending.append(index)
    if pending:
        batches.append(tuple(pending))
    return tuple(batches)


def concatenate_target_providers(
    providers: Sequence[ChunkTargetProvider], sizes: Sequence[int]
) -> ChunkTargetProvider:
    """Present trajectory-local supervision as one selected-position stream."""
    if not providers:
        raise ValueError("at least one target provider is required")
    if len(providers) != len(sizes):
        raise ValueError(f"received {len(providers)} providers for {len(sizes)} sizes")
    if any(size < 0 for size in sizes):
        raise ValueError("target-provider sizes must be non-negative")
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + int(size))
    return _ConcatenatedTargetProvider(tuple(providers), tuple(offsets))


def normalize_trajectory_weights(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """Normalize each trajectory independently before concatenating its tokens."""
    if not weights:
        return torch.zeros(0, dtype=torch.float32)
    normalized = []
    for row in weights:
        values = row.to(torch.float32)
        denominator = values.sum()
        normalized.append(
            values / denominator
            if bool(denominator > 0)
            else torch.zeros_like(values, dtype=torch.float32)
        )
    return torch.cat(normalized)
