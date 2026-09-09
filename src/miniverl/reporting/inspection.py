"""Torch-free run inspection shared by CLI, JSON and offline reports."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

from miniverl.errors import MiniVerlError, ReportError
from miniverl.utils.privacy import portable_payload


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def read_object(path: Path, *, text: str | None = None) -> dict[str, Any]:
    """Reject duplicate keys, non-finite values and non-object records."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8") if text is None else text, object_pairs_hook=_pairs
        )
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        _finite(value)
        return value
    except (OSError, ValueError, RecursionError) as exc:
        raise ReportError(f"invalid run artifact {path.name}: {exc}") from exc


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read every record, including records beyond the display limit."""
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            return [read_object(path, text=line) for line in handle if line.strip()]
    except (OSError, UnicodeError) as exc:
        raise ReportError(f"cannot read run artifact {path.name}: {exc}") from exc


def statistics(values: list[float]) -> dict[str, Any]:
    """Descriptive statistics; absence remains null."""
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": math.fsum(v / len(values) for v in values) if values else None,
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
    }


def _numbers(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in records:
        value = row.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ReportError(f"invalid numeric run field {key}")
            try:
                number = float(value)
            except OverflowError as exc:
                raise ReportError(f"numeric run field out of range: {key}") from exc
            if not math.isfinite(number):
                raise ReportError(f"non-finite numeric run field: {key}")
            values.append(number)
    return values


def inspect_artifacts(
    root: Path,
    manifest: dict[str, Any],
    metrics: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive training status while checking declared artifact integrity."""
    for key in (
        "objective",
        "config_digests",
        "final_checkpoint",
        "final_memory",
        "models",
        "profile_identity",
        "memory",
    ):
        if manifest.get(key) is not None and not isinstance(manifest[key], dict):
            raise ReportError(f"invalid run manifest object: {key}")
    for name, digest in (manifest.get("config_digests") or {}).items():
        filename = {
            "submitted": "config.submitted.yaml",
            "validated": "config.validated.yaml",
            "legacy_original": "config.original.yaml",
            "resolved": "config.resolved.yaml",
        }.get(name)
        if filename and digest is not None:
            path = root / filename
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ReportError(f"run config checksum mismatch: {filename}")
    checkpoint = manifest.get("final_checkpoint")
    checkpoint_summary = None
    if checkpoint:
        from miniverl.training.checkpoint import validate_checkpoint

        directory = checkpoint.get("directory")
        if (
            not isinstance(directory, str)
            or directory in {".", ".."}
            or Path(directory).name != directory
        ):
            raise ReportError("unsafe checkpoint directory")
        path = root / "checkpoints" / directory
        if path.is_symlink() or not path.is_dir():
            raise ReportError("missing or unsafe checkpoint directory")
        try:
            validated = validate_checkpoint(path)
        except (MiniVerlError, ValueError, OSError) as exc:
            raise ReportError(f"invalid checkpoint: {exc}") from exc
        if validated.content_digest != checkpoint.get("digest"):
            raise ReportError("checkpoint digest disagrees with run manifest")
        if manifest.get("status") == "completed":
            if validated.state.artifact_cursors:
                from miniverl.training.journal import capture_logs

                if capture_logs(root) != validated.state.artifact_cursors:
                    raise ReportError("completed run logs disagree with checkpoint bindings")
            for key in (
                "global_step",
                "actor_update_count",
                "critic_update_count",
                "policy_version",
            ):
                if key in manifest and manifest[key] != getattr(validated.state, key):
                    raise ReportError(f"checkpoint and completed run disagree on {key}")
        checkpoint_summary = {
            "directory": directory,
            "integrity": validated.integrity,
            "digest": validated.content_digest,
            "actor_updates": validated.state.actor_update_count,
            "critic_updates": validated.state.critic_update_count,
        }

    actor = [
        r for r in metrics if r.get("phase") in {"rl", "opd", "sft", "offline_kd", "sft_warmup"}
    ]
    critic = [r for r in metrics if r.get("phase") == "ppo_critic"]
    cycles = [r for r in metrics if str(r.get("phase", "")).endswith("_cycle")]
    rewards = read_records(root / "rewards.jsonl")
    advantages = read_records(root / "advantages.jsonl")
    rollout_summary: dict[str, Any] = {}
    if manifest.get("mode") == "rl" and (root / "trajectories.jsonl").exists():
        from miniverl.trajectory.io import read_trajectories

        try:
            trajectories = read_trajectories(root / "trajectories.jsonl")
        except MiniVerlError as exc:
            raise ReportError(f"invalid training trajectories: {exc}") from exc
        ids = {t.trajectory_id for t in trajectories}
        if len(ids) != len(trajectories):
            raise ReportError("duplicate training trajectory identity")
        for records, label in ((rewards, "reward"), (advantages, "advantage")):
            recorded = [row.get("trajectory_id") for row in records]
            if any(not isinstance(key, str) for key in recorded) or len(set(recorded)) != len(
                recorded
            ):
                raise ReportError(f"invalid or duplicate {label} identity")
            if not set(recorded).issubset(ids):
                raise ReportError(f"{label} record has no training trajectory")
            if manifest.get("status") == "completed" and set(recorded) != ids:
                raise ReportError(f"completed run is missing {label} records")
        rollout_summary = {
            "trajectories": len(trajectories),
            "prompt_groups": len({t.prompt_group_id for t in trajectories}),
            "generated_tokens": sum(sum(t.model_generated_mask) for t in trajectories),
        }
    rl = [r["verl_rl"] for r in actor if isinstance(r.get("verl_rl"), dict)]
    memory = [
        r["memory"]
        for r in metrics
        if isinstance(r.get("memory"), dict) and r["memory"].get("cuda_available")
    ]
    final_memory = manifest.get("final_memory") or {}
    if final_memory.get("cuda_available"):
        memory.append(final_memory)
    peaks = _numbers(memory, "peak_reserved_gib")
    for records, key in ((actor, "step"), (critic, "critic_update_count")):
        counters = _numbers(records, key)
        if any(b <= a for a, b in pairwise(counters)):
            raise ReportError(f"non-increasing {key} in run metrics")
        if manifest.get("status") == "completed" and counters:
            final = manifest.get("global_step" if key == "step" else key)
            if final is not None and final != counters[-1]:
                raise ReportError(f"completed run disagrees with metric {key}")
    reward_stats = statistics(_numbers(rewards, "raw_reward"))
    token_stats = {}
    for key in ("advantages", "returns"):
        values: list[float] = []
        for row in advantages:
            series = row.get(key)
            if series is None:
                continue
            if not isinstance(series, list) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) for v in series
            ):
                raise ReportError(f"invalid {key} vector")
            values.extend(_numbers([{"value": v} for v in series], "value"))
        token_stats[key] = statistics(values)
    return portable_payload(
        {
            "schema_version": 1,
            "run_id": manifest.get("run_id", root.name),
            "status": manifest.get("status", "unknown"),
            "mode": manifest.get("mode"),
            "algorithm": (manifest.get("objective") or {}).get("name"),
            "profile": manifest.get("profile_identity") or {},
            "models": manifest.get("models") or {},
            "local_placement": manifest.get("memory") or {},
            "rollouts": rollout_summary,
            "miniverl_version": manifest.get("miniverl_version"),
            "updates": {
                "actor_records": len(actor),
                "critic_records": len(critic),
                "actor_updates": manifest.get("actor_update_count"),
                "critic_updates": manifest.get("critic_update_count"),
                "policy_version": manifest.get("policy_version"),
                "completed_cycle_records": len(cycles),
            },
            "rewards": reward_stats,
            "reward_trend": _numbers(rewards, "raw_reward"),
            **token_stats,
            "actor_metrics": {
                key: statistics(_numbers(rl, key))
                for key in sorted(
                    {
                        k
                        for row in rl
                        for k, v in row.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
                )
            },
            "policy_loss": statistics(_numbers(actor, "loss")),
            "value_loss": statistics(_numbers(critic, "value_loss")),
            "memory": {
                "status": "measured" if peaks else "not_measured",
                "peak_reserved_gib": max(peaks) if peaks else None,
            },
            "events": [
                e
                for e in events
                if any(
                    word in str(e.get("event", ""))
                    for word in ("oom", "retry", "resume", "checkpoint", "downshift")
                )
            ],
            "checkpoint": checkpoint_summary,
            "resumed_from": manifest.get("resumed_from"),
            "export": {
                "checkpoint_verified": checkpoint_summary is not None,
                "status": "checkpoint_available_for_export"
                if checkpoint_summary
                else "no_verified_checkpoint",
                "launchable": False,
                "distributed_execution_tested": False,
                "next_command": "miniverl export-verl --run <run> --target-verl v0.9.0 --out <bundle>",
            },
        }
    )


def inspect_run(root: Path) -> dict[str, Any]:
    """Read one run without loading models or writing report files."""
    from miniverl.utils.runs import RunPaths

    paths = RunPaths.open(root)
    return inspect_artifacts(
        paths.root,
        read_object(paths.manifest),
        read_records(paths.metrics),
        read_records(paths.events),
    )


def human_summary(payload: dict[str, Any]) -> list[str]:
    """Readable defaults; the JSON view retains complete identities and event detail."""
    profile = payload["profile"]
    updates = payload["updates"]
    lines = [
        f"profile: {profile.get('profile_name', profile.get('profile', 'native'))}",
        f"updates: actor {updates['actor_records']}; critic {updates['critic_records']}; "
        f"policy version {updates['policy_version']}",
        f"rollouts: {payload['rollouts']}",
    ]
    for role, model in payload["models"].items():
        if isinstance(model, dict) and model.get("model_id"):
            lines.append(f"{role}: {model['model_id']} @ {model.get('revision') or 'unspecified'}")
    lowerings = profile.get("local_lowering", [])
    if lowerings:
        lines.append(f"local lowerings: {len(lowerings)} recorded fields (see --json)")
    for name in ("rewards", "advantages", "returns", "policy_loss", "value_loss"):
        stats = payload[name]
        if stats["count"]:
            lines.append(
                f"{name}: n={stats['count']}; mean={stats['mean']:.5g}; "
                f"range [{stats['min']:.5g}, {stats['max']:.5g}]; "
                f"first/last={stats['first']:.5g}/{stats['last']:.5g}"
            )
    for name, stats in payload["actor_metrics"].items():
        if any(word in name for word in ("kl", "entropy", "clip")) and stats["count"]:
            lines.append(f"{name}: mean={stats['mean']:.5g}")
    lines.append(
        f"peak reserved GiB: {payload['memory']['peak_reserved_gib']} ({payload['memory']['status']})"
    )
    lines.append(f"checkpoint: {payload['checkpoint']}; resumed from: {payload['resumed_from']}")
    event_counts: dict[str, int] = {}
    for event in payload["events"]:
        name = str(event.get("event"))
        event_counts[name] = event_counts.get(name, 0) + 1
    lines.append(f"recovery/checkpoint events: {event_counts}")
    lines.append(f"export: {payload['export']['status']}; distributed execution not tested")
    return lines
