"""Typed compiler for miniVERL's pinned, single-GPU verl v0.8 OPD subset.

This module compiles configuration only. It does not import verl, allocate model
weights, execute an inference engine, or claim distributed compatibility.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from miniverl.bridge.contract import VERL_COMMIT, VERL_REPOSITORY, VERL_TAG
from miniverl.bridge.interpolation import reject_interpolation
from miniverl.errors import ConfigError

__all__ = [
    "CompiledLocalExecutionPlan",
    "VERL_OPD_V08_PROFILE",
    "VerlOPDV08Profile",
    "compile_verl_opd_v08",
    "load_verl_opd_v08",
    "load_verl_opd_v08_source",
    "parse_overrides",
    "publish_imported_verl_opd_v08",
]

VERL_OPD_V08_PROFILE = "verl-opd-v0.8-single-gpu-v1"

FieldClassification = Literal[
    "exact",
    "semantically_conformant",
    "locally_reinterpreted",
    "derived",
    "informational_only",
    "unsupported",
]
RuntimeMode = Literal["dual_model_resident", "shared_backbone", "swap", "auto"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


def _finite(value: float, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


class VerlOPDDataConfig(_StrictModel):
    train_files: list[str]
    val_files: list[str] = Field(default_factory=list)
    prompt_key: str = "prompt"
    train_batch_size: int = Field(ge=1)
    max_prompt_length: int = Field(ge=1)
    max_response_length: int = Field(ge=1)
    filter_overlong_prompts: bool = False
    truncation: Literal["error", "left", "right"] = "error"
    shuffle: bool = True
    seed: int | None = None

    @field_validator("train_files", "val_files", mode="before")
    @classmethod
    def _normalize_paths(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("train_files", "val_files")
    @classmethod
    def _nonempty_paths(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Parquet paths must not be empty")
        return value


class VerlOPDModelConfig(_StrictModel):
    path: str = Field(min_length=1)
    use_remove_padding: bool = False
    enable_gradient_checkpointing: bool = False
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    target_modules: list[str] = Field(default_factory=list)
    lora_adapter_path: str | None = None


class VerlOPDOptimConfig(_StrictModel):
    lr: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    lr_warmup_steps: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _numbers_are_finite(self) -> VerlOPDOptimConfig:
        _finite(self.lr, "actor_rollout_ref.actor.optim.lr")
        _finite(self.weight_decay, "actor_rollout_ref.actor.optim.weight_decay")
        return self


class VerlOPDFSDPConfig(_StrictModel):
    param_offload: bool = False
    optimizer_offload: bool = False


class VerlOPDActorConfig(_StrictModel):
    optim: VerlOPDOptimConfig
    use_torch_compile: bool = False
    fsdp_config: VerlOPDFSDPConfig = Field(default_factory=VerlOPDFSDPConfig)
    loss_agg_mode: str = "token-mean"
    use_kl_loss: bool = False
    ppo_mini_batch_size: int = Field(default=1, ge=1)
    ppo_max_token_len_per_gpu: int = Field(default=16384, ge=1)
    use_dynamic_bsz: bool = False


class VerlOPDRolloutConfig(_StrictModel):
    name: str
    n: int = Field(default=1, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    tensor_model_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.5, gt=0, le=1)
    max_model_len: int | None = Field(default=None, ge=1)
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    max_num_seqs: int | None = Field(default=None, ge=1)
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _numbers_are_finite(self) -> VerlOPDRolloutConfig:
        _finite(self.temperature, "actor_rollout_ref.rollout.temperature")
        _finite(self.top_p, "actor_rollout_ref.rollout.top_p")
        _finite(
            self.gpu_memory_utilization,
            "actor_rollout_ref.rollout.gpu_memory_utilization",
        )
        return self


class VerlOPDActorRolloutRefConfig(_StrictModel):
    model: VerlOPDModelConfig
    actor: VerlOPDActorConfig
    rollout: VerlOPDRolloutConfig


class VerlOPDAlgorithmConfig(_StrictModel):
    adv_estimator: str | None = None
    use_kl_in_reward: bool = False


class VerlOPDTeacherInferenceConfig(_StrictModel):
    name: str
    dtype: str = "bfloat16"
    tensor_model_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    pipeline_model_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.5, gt=0, le=1)
    max_model_len: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _numbers_are_finite(self) -> VerlOPDTeacherInferenceConfig:
        _finite(
            self.gpu_memory_utilization,
            "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization",
        )
        return self


class VerlOPDTeacherConfig(_StrictModel):
    model_path: str = Field(min_length=1)
    num_replicas: int = Field(default=1, ge=0)
    inference: VerlOPDTeacherInferenceConfig


class VerlOPDTeacherModelsConfig(_StrictModel):
    teacher_model: VerlOPDTeacherConfig


class VerlOPDLossConfig(_StrictModel):
    loss_mode: str
    topk: int = Field(ge=1)
    use_task_rewards: bool = False
    distillation_loss_coef: float = Field(default=1.0, ge=0)
    loss_max_clamp: float | None = Field(default=None, ge=0)
    log_prob_min_clamp: float | None = None
    use_policy_gradient: bool = False

    @model_validator(mode="after")
    def _numbers_are_finite(self) -> VerlOPDLossConfig:
        _finite(self.distillation_loss_coef, "distillation_loss_coef")
        if self.loss_max_clamp is not None:
            _finite(self.loss_max_clamp, "loss_max_clamp")
        if self.log_prob_min_clamp is not None:
            _finite(self.log_prob_min_clamp, "log_prob_min_clamp")
        return self


class VerlOPDDistillationConfig(_StrictModel):
    enabled: bool
    teacher_key: str = "data_source"
    n_gpus_per_node: int = Field(default=1, ge=0)
    nnodes: int = Field(default=1, ge=0)
    teacher_models: VerlOPDTeacherModelsConfig
    distillation_loss: VerlOPDLossConfig


class VerlOPDTrainerConfig(_StrictModel):
    project_name: str = "miniverl"
    experiment_name: str = "verl-opd"
    save_freq: int = Field(default=-1, ge=-1)
    test_freq: int = Field(default=-1, ge=-1)
    total_epochs: int = Field(default=1, ge=1)
    total_training_steps: int | None = Field(default=None, ge=1)
    n_gpus_per_node: int = Field(default=1, ge=1)
    nnodes: int = Field(default=1, ge=1)
    balance_batch: bool = False
    logger: str | list[str] = "console"
    val_before_train: bool = False


class MiniVerlRuntimeExtensions(_StrictModel):
    mode: RuntimeMode = "auto"


class MiniVerlMemoryExtensions(_StrictModel):
    vram_limit_gib: float = Field(default=16.0, gt=0)
    headroom_gib: float = Field(default=1.5, ge=0)

    @model_validator(mode="after")
    def _numbers_are_finite(self) -> MiniVerlMemoryExtensions:
        _finite(self.vram_limit_gib, "miniverl.memory.vram_limit_gib")
        _finite(self.headroom_gib, "miniverl.memory.headroom_gib")
        if self.headroom_gib >= self.vram_limit_gib:
            raise ValueError("headroom_gib must be smaller than vram_limit_gib")
        return self


class MiniVerlBatchingExtensions(_StrictModel):
    rollout_batch_size: int = Field(default=1, ge=1)
    teacher_score_batch_size: int = Field(default=1, ge=1)
    update_trajectory_batch_size: int = Field(default=1, ge=1)


class MiniVerlTeacherAdapterExtensions(_StrictModel):
    path: str | None = None
    revision: str | None = None


class MiniVerlLocalExtensions(_StrictModel):
    student_revision: str | None = None
    teacher_revision: str | None = None
    runtime: MiniVerlRuntimeExtensions = Field(default_factory=MiniVerlRuntimeExtensions)
    memory: MiniVerlMemoryExtensions = Field(default_factory=MiniVerlMemoryExtensions)
    batching: MiniVerlBatchingExtensions = Field(default_factory=MiniVerlBatchingExtensions)
    teacher_adapter: MiniVerlTeacherAdapterExtensions = Field(
        default_factory=MiniVerlTeacherAdapterExtensions
    )


class VerlOPDV08Profile(_StrictModel):
    """The public, typed input surface; deliberately separate from RunConfig."""

    data: VerlOPDDataConfig
    actor_rollout_ref: VerlOPDActorRolloutRefConfig
    algorithm: VerlOPDAlgorithmConfig
    distillation: VerlOPDDistillationConfig
    trainer: VerlOPDTrainerConfig
    miniverl: MiniVerlLocalExtensions = Field(default_factory=MiniVerlLocalExtensions)


class OverrideRecord(_StrictModel):
    expression: str
    field: str
    value: Any
    source_kind: Literal["overrides_file", "set", "trailing"] = "set"
    source: str = "--set"
    order: int = 0
    previous_value: Any = None
    previous_source: str = "base_config"
    final_value: Any = None
    effective: bool = False


class CompatibilityEntry(_StrictModel):
    upstream_field: str
    source_value: Any
    local_target: str | None
    classification: FieldClassification
    reason: str
    semantic_risk: Literal["none", "low", "medium", "high"]
    user_confirmation_required: bool
    executable: bool


class CompiledLocalExecutionPlan(_StrictModel):
    schema_version: Literal[2] = 2
    profile: Literal["verl-opd-v0.8-single-gpu-v1"] = "verl-opd-v0.8-single-gpu-v1"
    upstream: dict[str, str]
    source_digest: str
    compiled_digest: str
    source_leaf_fields: list[str]
    source: VerlOPDV08Profile
    overrides: list[OverrideRecord]
    compatibility: list[CompatibilityEntry]
    reinterpretation_acceptance: dict[str, Any]
    local_execution: dict[str, Any]
    executable: bool


@dataclass(frozen=True)
class _Rule:
    target: str | None
    classification: FieldClassification
    reason: str
    risk: Literal["none", "low", "medium", "high"] = "none"
    confirmation: bool = False


def _rule(
    target: str | None,
    classification: FieldClassification,
    reason: str,
    risk: Literal["none", "low", "medium", "high"] = "none",
    confirmation: bool = False,
) -> _Rule:
    return _Rule(target, classification, reason, risk, confirmation)


_FIELD_RULES: dict[str, _Rule] = {
    "data.train_files": _rule("source.train_files", "exact", "the same Parquet paths are consumed"),
    "data.val_files": _rule("source.val_files", "exact", "the same validation paths are consumed"),
    "data.prompt_key": _rule("source.prompt_key", "exact", "the same row field is selected"),
    "data.train_batch_size": _rule(
        "scheduler.logical_batch_size",
        "locally_reinterpreted",
        "logical examples per update; physical one-GPU batches are separately bounded",
        "medium",
    ),
    "data.max_prompt_length": _rule("source.max_prompt_length", "exact", "same token bound"),
    "data.max_response_length": _rule("source.max_response_length", "exact", "same token bound"),
    "data.filter_overlong_prompts": _rule(
        "source.filter_overlong_prompts", "semantically_conformant", "same filtering intent"
    ),
    "data.truncation": _rule("source.truncation", "semantically_conformant", "same named policy"),
    "data.shuffle": _rule(
        "source.shuffle", "semantically_conformant", "deterministic local shuffle"
    ),
    "data.seed": _rule("source.seed", "exact", "same integer seed"),
    "actor_rollout_ref.model.path": _rule("student.model_id", "exact", "same model identity"),
    "actor_rollout_ref.model.use_remove_padding": _rule(
        "student.padding",
        "semantically_conformant",
        "the local padded runtime removes padding logically",
    ),
    "actor_rollout_ref.model.enable_gradient_checkpointing": _rule(
        "student.gradient_checkpointing", "semantically_conformant", "same memory technique"
    ),
    "actor_rollout_ref.model.lora_rank": _rule(
        "student.lora.r", "semantically_conformant", "same PEFT rank"
    ),
    "actor_rollout_ref.model.lora_alpha": _rule(
        "student.lora.alpha", "semantically_conformant", "same PEFT scale"
    ),
    "actor_rollout_ref.model.target_modules": _rule(
        "student.lora.target_modules", "semantically_conformant", "same module names"
    ),
    "actor_rollout_ref.model.lora_adapter_path": _rule(
        "student.adapter.path", "exact", "same optional adapter artifact identity"
    ),
    "actor_rollout_ref.actor.optim.lr": _rule("optimizer.lr", "exact", "same learning rate"),
    "actor_rollout_ref.actor.optim.weight_decay": _rule(
        "optimizer.weight_decay", "exact", "same optimizer coefficient"
    ),
    "actor_rollout_ref.actor.optim.lr_warmup_steps": _rule(
        "optimizer.lr_warmup_steps", "exact", "same optimizer-step count"
    ),
    "actor_rollout_ref.actor.use_torch_compile": _rule(
        None,
        "informational_only",
        "recorded for provenance; the local runtime selects its own compilation path",
    ),
    "actor_rollout_ref.actor.fsdp_config.param_offload": _rule(
        None,
        "informational_only",
        "false is a harmless distributed-training no-op; true is unsupported",
    ),
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": _rule(
        None,
        "informational_only",
        "false is a harmless distributed-training no-op; true is unsupported",
    ),
    "actor_rollout_ref.actor.loss_agg_mode": _rule(
        "loss.reduction", "semantically_conformant", "token-mean is the only executable v0.8 mode"
    ),
    "actor_rollout_ref.actor.use_kl_loss": _rule(
        "loss.actor_reference_kl", "exact", "must remain disabled for this profile"
    ),
    "actor_rollout_ref.actor.ppo_mini_batch_size": _rule(
        "batching.update_trajectory_batch_size",
        "locally_reinterpreted",
        "used as a logical update batch, not a PPO mini-batch",
        "high",
    ),
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": _rule(
        "batching.update_token_budget",
        "locally_reinterpreted",
        "one-GPU physical token budget",
        "medium",
    ),
    "actor_rollout_ref.actor.use_dynamic_bsz": _rule(
        "batching.dynamic_physical_batching",
        "locally_reinterpreted",
        "may change only physical execution",
        "medium",
    ),
    "actor_rollout_ref.rollout.name": _rule(
        "rollout.backend",
        "locally_reinterpreted",
        "source engine name is recorded; local execution does not claim vLLM/SGLang equivalence",
        "high",
    ),
    "actor_rollout_ref.rollout.n": _rule("rollout.n", "exact", "one generation per prompt"),
    "actor_rollout_ref.rollout.temperature": _rule(
        "rollout.temperature", "exact", "same sampling value"
    ),
    "actor_rollout_ref.rollout.top_p": _rule("rollout.top_p", "exact", "same sampling value"),
    "actor_rollout_ref.rollout.tensor_model_parallel_size": _rule(
        "placement.tensor_parallel", "locally_reinterpreted", "must be one on one GPU", "high"
    ),
    "actor_rollout_ref.rollout.gpu_memory_utilization": _rule(
        "memory.rollout_fraction",
        "locally_reinterpreted",
        "planner hint rather than an inference-server reservation",
        "high",
    ),
    "actor_rollout_ref.rollout.max_model_len": _rule(
        "rollout.max_model_len", "semantically_conformant", "same context ceiling"
    ),
    "actor_rollout_ref.rollout.max_num_batched_tokens": _rule(
        "batching.rollout_token_budget",
        "locally_reinterpreted",
        "local padded-token budget",
        "medium",
    ),
    "actor_rollout_ref.rollout.max_num_seqs": _rule(
        "batching.rollout_batch_limit", "locally_reinterpreted", "local sequence cap", "medium"
    ),
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": _rule(
        None,
        "informational_only",
        "no separate rollout log-prob worker exists in direct GKD OPD",
    ),
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": _rule(
        None,
        "informational_only",
        "no separate rollout log-prob worker exists in direct GKD OPD",
    ),
    "algorithm.adv_estimator": _rule(
        None,
        "unsupported",
        "policy-gradient advantage estimation is outside direct GKD OPD",
        "high",
    ),
    "algorithm.use_kl_in_reward": _rule(
        "loss.kl_in_reward", "exact", "must remain disabled; pure OPD has no reward"
    ),
    "distillation.enabled": _rule("loss.enabled", "exact", "must be enabled"),
    "distillation.teacher_key": _rule(
        "teacher.routing_metadata",
        "informational_only",
        "single-teacher local execution does not route",
    ),
    "distillation.n_gpus_per_node": _rule(
        "placement.teacher_phase",
        "locally_reinterpreted",
        "resource pool becomes one-GPU phases",
        "high",
    ),
    "distillation.nnodes": _rule(
        "placement.teacher_phase",
        "locally_reinterpreted",
        "resource pool becomes one local node",
        "high",
    ),
    "distillation.teacher_models.teacher_model.model_path": _rule(
        "teacher.model_id", "exact", "same teacher model identity"
    ),
    "distillation.teacher_models.teacher_model.num_replicas": _rule(
        "placement.teacher_phase",
        "locally_reinterpreted",
        "one frozen teacher role, no replicas",
        "high",
    ),
    "distillation.teacher_models.teacher_model.inference.name": _rule(
        "teacher.backend",
        "locally_reinterpreted",
        "recorded without engine-equivalence claim",
        "high",
    ),
    "distillation.teacher_models.teacher_model.inference.dtype": _rule(
        "teacher.dtype", "semantically_conformant", "same requested numerical dtype"
    ),
    "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size": _rule(
        "placement.teacher_tensor_parallel", "locally_reinterpreted", "must be one", "high"
    ),
    "distillation.teacher_models.teacher_model.inference.data_parallel_size": _rule(
        "placement.teacher_data_parallel", "locally_reinterpreted", "must be one", "high"
    ),
    "distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size": _rule(
        "placement.teacher_pipeline_parallel", "locally_reinterpreted", "must be one", "high"
    ),
    "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization": _rule(
        "memory.teacher_fraction",
        "locally_reinterpreted",
        "planner hint, not server reservation",
        "high",
    ),
    "distillation.teacher_models.teacher_model.inference.max_model_len": _rule(
        "teacher.max_model_len", "semantically_conformant", "same scoring context ceiling"
    ),
    "distillation.distillation_loss.loss_mode": _rule(
        "loss.mode", "semantically_conformant", "dedicated pinned forward_kl_topk path"
    ),
    "distillation.distillation_loss.topk": _rule("loss.top_k", "exact", "same teacher top-k"),
    "distillation.distillation_loss.use_task_rewards": _rule(
        "loss.use_task_rewards", "exact", "must remain disabled"
    ),
    "distillation.distillation_loss.distillation_loss_coef": _rule(
        None, "informational_only", "coefficient is inactive when task rewards are disabled"
    ),
    "distillation.distillation_loss.loss_max_clamp": _rule(
        "loss.loss_max_clamp", "semantically_conformant", "same optional final clamp"
    ),
    "distillation.distillation_loss.log_prob_min_clamp": _rule(
        "loss.log_prob_min_clamp", "semantically_conformant", "same optional log-prob clamp"
    ),
    "distillation.distillation_loss.use_policy_gradient": _rule(
        "loss.use_policy_gradient", "exact", "must remain disabled for direct GKD OPD"
    ),
    "trainer.project_name": _rule("run.project", "exact", "same provenance label"),
    "trainer.experiment_name": _rule("run.name", "exact", "same provenance label"),
    "trainer.save_freq": _rule(
        "checkpoint.interval",
        "locally_reinterpreted",
        "local optimizer-step interval",
        "medium",
        True,
    ),
    "trainer.test_freq": _rule(
        "evaluation.interval",
        "locally_reinterpreted",
        "local optimizer-step interval",
        "medium",
        True,
    ),
    "trainer.total_epochs": _rule(
        "schedule.dataset_passes",
        "locally_reinterpreted",
        "local bounded dataset passes",
        "medium",
        True,
    ),
    "trainer.total_training_steps": _rule(
        "schedule.optimizer_steps", "semantically_conformant", "explicit global optimizer-step cap"
    ),
    "trainer.n_gpus_per_node": _rule(
        "placement.device_count", "locally_reinterpreted", "always one local CUDA device", "high"
    ),
    "trainer.nnodes": _rule("placement.node_count", "locally_reinterpreted", "must be one", "high"),
    "trainer.balance_batch": _rule(
        None, "informational_only", "logical batches are already deterministic and locally ordered"
    ),
    "trainer.logger": _rule(
        None, "informational_only", "external logger selection is not executed by miniVERL"
    ),
    "trainer.val_before_train": _rule(
        None, "informational_only", "recorded; validation scheduling follows the local run contract"
    ),
    "miniverl.runtime.mode": _rule(None, "informational_only", "miniVERL-only local extension"),
    "miniverl.student_revision": _rule(
        "student.revision", "informational_only", "miniVERL-only immutable Hub revision"
    ),
    "miniverl.teacher_revision": _rule(
        "teacher.revision", "informational_only", "miniVERL-only immutable Hub revision"
    ),
    "miniverl.memory.vram_limit_gib": _rule(
        None, "informational_only", "miniVERL-only planner limit"
    ),
    "miniverl.memory.headroom_gib": _rule(
        None, "informational_only", "miniVERL-only planner headroom"
    ),
    "miniverl.batching.rollout_batch_size": _rule(
        None, "informational_only", "miniVERL-only physical batch"
    ),
    "miniverl.batching.teacher_score_batch_size": _rule(
        None, "informational_only", "miniVERL-only physical batch"
    ),
    "miniverl.batching.update_trajectory_batch_size": _rule(
        None, "informational_only", "miniVERL-only physical batch"
    ),
    "miniverl.teacher_adapter.path": _rule(
        None, "informational_only", "miniVERL-only teacher adapter extension"
    ),
    "miniverl.teacher_adapter.revision": _rule(
        None, "informational_only", "miniVERL-only teacher adapter extension"
    ),
}


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


_NUMERIC_FIELDS = {
    path
    for path in _FIELD_RULES
    if path.endswith(
        (
            ".lr",
            ".weight_decay",
            ".lr_warmup_steps",
            ".train_batch_size",
            ".max_prompt_length",
            ".max_response_length",
            ".lora_rank",
            ".lora_alpha",
            ".ppo_mini_batch_size",
            ".ppo_max_token_len_per_gpu",
            ".temperature",
            ".top_p",
            ".gpu_memory_utilization",
            ".max_model_len",
            ".max_num_batched_tokens",
            ".max_num_seqs",
            ".n",
            ".topk",
            ".distillation_loss_coef",
            ".loss_max_clamp",
            ".log_prob_min_clamp",
            ".save_freq",
            ".test_freq",
            ".total_epochs",
            ".total_training_steps",
            ".n_gpus_per_node",
            ".nnodes",
            ".num_replicas",
            ".tensor_model_parallel_size",
            ".data_parallel_size",
            ".pipeline_model_parallel_size",
            ".vram_limit_gib",
            ".headroom_gib",
            ".rollout_batch_size",
            ".teacher_score_batch_size",
            ".update_trajectory_batch_size",
        )
    )
}


def _reject_non_finite_source_numbers(flat: Mapping[str, Any]) -> None:
    spellings = {
        "nan",
        ".nan",
        "+nan",
        "-nan",
        "inf",
        ".inf",
        "+inf",
        "-.inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }
    for path in sorted(_NUMERIC_FIELDS & flat.keys()):
        value = flat[path]
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"verl OPD numeric field {path} must be finite")
        if isinstance(value, str) and value.strip().lower() in spellings:
            raise ConfigError(f"verl OPD numeric field {path} must be finite")


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not all(parts):
        raise ConfigError(f"invalid empty component in override field {path!r}")
    current = payload
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise ConfigError(f"override {path!r} crosses non-mapping field {part!r}")
        current = existing
    current[parts[-1]] = value


def _validate_override_value(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"override {field} must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_override_value(item, field=field)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ConfigError(f"override {field} mapping keys must be strings")
        for item in value.values():
            _validate_override_value(item, field=field)
        return
    raise ConfigError(
        f"override {field} uses unsupported YAML value type {type(value).__name__}",
        hint="use finite JSON/YAML strings, numbers, booleans, null, lists, or mappings",
    )


def _read_override_file(path: Path) -> list[str]:
    if path.suffix.lower() in {".sh", ".bash", ".ps1", ".cmd", ".bat"}:
        raise ConfigError(f"override file must contain data, not a shell script: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"cannot read override file {path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"override argv JSON {path} is invalid: {exc}") from exc
        if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
            raise ConfigError(f"override argv JSON {path} must be an array of strings")
        return payload
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_overrides(
    expressions: Sequence[str],
    *,
    source_kind: Literal["overrides_file", "set", "trailing"] = "set",
    source: str = "--set",
    start_order: int = 0,
) -> list[OverrideRecord]:
    """Parse repeatable ``key=value`` overrides without Hydra or shell evaluation."""
    records: list[OverrideRecord] = []
    for offset, expression in enumerate(expressions):
        if "=" not in expression:
            raise ConfigError(f"override {expression!r} must use key=value syntax")
        field, raw = expression.split("=", 1)
        field = field.strip()
        if not field or field.startswith("+") or field.startswith("~"):
            raise ConfigError(
                f"override field {field!r} is not supported",
                hint="use an explicit existing dotted field without Hydra +/~ operators",
            )
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"override {expression!r} has invalid YAML value: {exc}") from exc
        reject_interpolation(value, label=f"override {field}")
        _validate_override_value(value, field=field)
        records.append(
            OverrideRecord(
                expression=expression,
                field=field,
                value=value,
                source_kind=source_kind,
                source=source,
                order=start_order + offset,
            )
        )
    return records


def _resolve_overrides(
    payload: dict[str, Any],
    *,
    override_files: Sequence[Path],
    overrides: Sequence[str],
    trailing_overrides: Sequence[str],
) -> list[OverrideRecord]:
    records: list[OverrideRecord] = []
    order = 0
    for path in override_files:
        parsed = parse_overrides(
            _read_override_file(path),
            source_kind="overrides_file",
            source=str(path),
            start_order=order,
        )
        records.extend(parsed)
        order += len(parsed)
    parsed_set = parse_overrides(overrides, source_kind="set", source="--set", start_order=order)
    records.extend(parsed_set)
    order += len(parsed_set)
    records.extend(
        parse_overrides(
            trailing_overrides,
            source_kind="trailing",
            source="--",
            start_order=order,
        )
    )

    current_sources = dict.fromkeys(_flatten(payload), "base_config")
    for record in records:
        before = _flatten(payload)
        record.previous_value = before.get(record.field)
        record.previous_source = current_sources.get(record.field, "not_present")
        _set_path(payload, record.field, record.value)
        current_sources[record.field] = record.source
    final = _flatten(payload)
    last_order = {record.field: record.order for record in records}
    for record in records:
        record.final_value = final.get(record.field)
        record.effective = last_order[record.field] == record.order
    return records


def _semantic_blocker(path: str, value: Any) -> str | None:
    required = {
        "distillation.enabled": (True, "distillation must be enabled"),
        "distillation.distillation_loss.loss_mode": (
            "forward_kl_topk",
            "only pinned forward_kl_topk is supported",
        ),
        "distillation.distillation_loss.use_policy_gradient": (
            False,
            "policy-gradient OPD is unsupported",
        ),
        "distillation.distillation_loss.use_task_rewards": (
            False,
            "task-reward mixtures are unsupported",
        ),
        "actor_rollout_ref.actor.use_kl_loss": (
            False,
            "actor/reference KL loss is outside pure OPD",
        ),
        "algorithm.use_kl_in_reward": (False, "KL reward is outside pure OPD"),
        "actor_rollout_ref.actor.loss_agg_mode": (
            "token-mean",
            "only token-mean aggregation is supported",
        ),
        "actor_rollout_ref.rollout.n": (1, "one generation per prompt is required"),
        "actor_rollout_ref.rollout.tensor_model_parallel_size": (
            1,
            "tensor parallelism greater than one is unsupported",
        ),
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size": (
            1,
            "teacher tensor parallelism greater than one is unsupported",
        ),
        "distillation.teacher_models.teacher_model.inference.data_parallel_size": (
            1,
            "teacher data parallelism greater than one is unsupported",
        ),
        "distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size": (
            1,
            "teacher pipeline parallelism greater than one is unsupported",
        ),
        "trainer.nnodes": (1, "multi-node execution is unsupported"),
    }
    expected = required.get(path)
    if expected is not None and value != expected[0]:
        return expected[1]
    if path == "trainer.n_gpus_per_node" and value != 1:
        return "more than one local training GPU is unsupported"
    if path == "distillation.nnodes" and value not in {0, 1}:
        return "multi-node teacher execution is unsupported"
    if path == "algorithm.adv_estimator" and value is not None:
        return "policy-gradient advantage estimation is outside direct GKD OPD"
    if (
        path
        in {
            "actor_rollout_ref.actor.fsdp_config.param_offload",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
        }
        and value
    ):
        return "FSDP offload semantics are unsupported by the one-GPU local runtime"
    return None


def compile_verl_opd_v08(
    payload: Mapping[str, Any],
    *,
    override_files: Sequence[Path] = (),
    overrides: Sequence[str] = (),
    trailing_overrides: Sequence[str] = (),
    reinterpretations_accepted: bool = True,
    acceptance_source: str = "library_call",
    require_executable: bool = True,
) -> CompiledLocalExecutionPlan:
    """Compile one resolved documented profile into a deterministic local plan."""
    merged = copy.deepcopy(dict(payload))
    reject_interpolation(merged, label="verl OPD config")
    records = _resolve_overrides(
        merged,
        override_files=override_files,
        overrides=overrides,
        trailing_overrides=trailing_overrides,
    )
    reject_interpolation(merged, label="resolved verl OPD config")

    teachers = (
        merged.get("distillation", {}).get("teacher_models", {})
        if isinstance(merged.get("distillation"), Mapping)
        else {}
    )
    if isinstance(teachers, Mapping) and set(teachers) != {"teacher_model"}:
        raise ConfigError(
            "verl OPD config is not executable: multi-teacher routing is unsupported; "
            f"found {sorted(map(str, teachers))}"
        )

    flat = _flatten(merged)
    _reject_non_finite_source_numbers(flat)
    unknown = sorted(path for path in flat if path not in _FIELD_RULES)
    if unknown:
        raise ConfigError(
            f"verl OPD config is not executable: unsupported field {unknown[0]!r}",
            hint="the profile accepts only its documented resolved verl v0.8 subset",
        )
    try:
        source = VerlOPDV08Profile.model_validate(merged)
    except ValidationError as exc:
        finite = any("finite" in error.get("msg", "").lower() for error in exc.errors())
        message = "verl OPD numeric fields must be finite" if finite else "invalid verl OPD profile"
        raise ConfigError(message, hint=str(exc)) from exc

    compatibility: list[CompatibilityEntry] = []
    blockers: list[str] = []
    for path, value in sorted(flat.items()):
        rule = _FIELD_RULES[path]
        blocker = _semantic_blocker(path, value)
        if blocker:
            blockers.append(path)
            classification: FieldClassification = "unsupported"
            reason = blocker
            risk: Literal["none", "low", "medium", "high"] = "high"
        else:
            classification = rule.classification
            reason = rule.reason
            risk = rule.risk
        confirmation_required = rule.confirmation or (
            classification == "locally_reinterpreted" and risk == "high"
        )
        compatibility.append(
            CompatibilityEntry(
                upstream_field=path,
                source_value=value,
                local_target=rule.target,
                classification=classification,
                reason=reason,
                semantic_risk=risk,
                user_confirmation_required=confirmation_required,
                executable=blocker is None,
            )
        )

    executable = not blockers
    local_execution = {
        "device_count": 1,
        "distributed_execution": False,
        "compiler_scope": "config_semantics_only",
        "model_weights_loaded": False,
        "roles": ["actor_rollout", "teacher_scoring", "actor_update", "artifact"],
        "runtime_mode": source.miniverl.runtime.mode,
        "student_model": source.actor_rollout_ref.model.path,
        "teacher_model": source.distillation.teacher_models.teacher_model.model_path,
        "loss_mode": source.distillation.distillation_loss.loss_mode,
        "loss_reduction": source.actor_rollout_ref.actor.loss_agg_mode,
        "task_rewards": False,
        "policy_gradient": False,
    }
    high_risk = [item.upstream_field for item in compatibility if item.user_confirmation_required]
    common = {
        "upstream": {"repository": VERL_REPOSITORY, "tag": VERL_TAG, "commit": VERL_COMMIT},
        "source_digest": _canonical_digest(merged),
        "source_leaf_fields": sorted(flat),
        "source": source,
        "overrides": records,
        "compatibility": compatibility,
        "reinterpretation_acceptance": {
            "required_fields": high_risk,
            "accepted": reinterpretations_accepted or not high_risk,
            "source": acceptance_source if high_risk else "not_required",
        },
        "local_execution": local_execution,
        "executable": executable,
    }
    digest_payload = {
        key: value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        for key, value in common.items()
    }
    digest_payload["overrides"] = [item.model_dump(mode="json") for item in records]
    digest_payload["compatibility"] = [item.model_dump(mode="json") for item in compatibility]
    plan = CompiledLocalExecutionPlan(
        **common,
        compiled_digest=_canonical_digest(digest_payload),
    )
    if require_executable and not executable:
        raise ConfigError(
            "verl OPD config is not executable: unsupported semantics in " + ", ".join(blockers)
        )
    return plan


def load_verl_opd_v08(
    path: Path,
    *,
    override_files: Sequence[Path] = (),
    overrides: Sequence[str] = (),
    trailing_overrides: Sequence[str] = (),
    accept_local_reinterpretations: bool = False,
    require_executable: bool = True,
) -> CompiledLocalExecutionPlan:
    """Load a resolved YAML mapping; scripts and interpolations are never evaluated."""
    if path.suffix.lower() in {".sh", ".bash", ".ps1", ".cmd", ".bat"}:
        raise ConfigError("verl OPD input must be resolved YAML, not a shell script")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read resolved verl OPD YAML {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("verl OPD YAML must contain one mapping")
    return compile_verl_opd_v08(
        payload,
        override_files=override_files,
        overrides=overrides,
        trailing_overrides=trailing_overrides,
        reinterpretations_accepted=accept_local_reinterpretations,
        acceptance_source="cli_flag" if accept_local_reinterpretations else "not_accepted",
        require_executable=require_executable,
    )


def load_verl_opd_v08_source(
    source: str | Path,
    *,
    override_files: Sequence[Path] = (),
    overrides: Sequence[str] = (),
    trailing_overrides: Sequence[str] = (),
    accept_local_reinterpretations: bool = False,
    require_executable: bool = True,
) -> CompiledLocalExecutionPlan:
    """Load a path or the packaged Qwen3 quickstart profile."""
    source_text = str(source)
    if source_text != "builtin:qwen3-0.6b-1.7b-opd":
        return load_verl_opd_v08(
            Path(source),
            override_files=override_files,
            overrides=overrides,
            trailing_overrides=trailing_overrides,
            accept_local_reinterpretations=accept_local_reinterpretations,
            require_executable=require_executable,
        )
    from importlib.resources import files

    resource = files("miniverl").joinpath("resources/qwen3_0_6b_1_7b_opd.yaml")
    approval_resource = files("miniverl").joinpath("resources/qwen3_0_6b_1_7b_opd_approval.json")
    try:
        payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read packaged OPD profile: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("packaged OPD profile must contain one mapping")
    unaccepted = compile_verl_opd_v08(
        payload,
        override_files=override_files,
        overrides=overrides,
        trailing_overrides=trailing_overrides,
        reinterpretations_accepted=False,
        acceptance_source="packaged_approval_mismatch",
        require_executable=require_executable,
    )
    try:
        approval = json.loads(approval_resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read packaged OPD approval manifest: {exc}") from exc
    approved_values = approval.get("approved_high_risk_values")
    observed_values = {
        item.upstream_field: item.source_value
        for item in unaccepted.compatibility
        if item.user_confirmation_required
    }
    manifest_matches = (
        approval.get("schema_version") == 1
        and approval.get("profile") == VERL_OPD_V08_PROFILE
        and approved_values == observed_values
    )
    if not manifest_matches and not accept_local_reinterpretations:
        return unaccepted
    return compile_verl_opd_v08(
        payload,
        override_files=override_files,
        overrides=overrides,
        trailing_overrides=trailing_overrides,
        reinterpretations_accepted=True,
        acceptance_source=(
            "cli_flag" if accept_local_reinterpretations else "packaged_approval_manifest"
        ),
        require_executable=require_executable,
    )


def publish_imported_verl_opd_v08(
    source: str | Path,
    *,
    out: str | Path,
    overrides: Sequence[str] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Transactionally publish one canonical, round-trippable OPD profile family."""
    from miniverl.bridge.publish import (
        OutputTransaction,
        import_output_targets,
        reject_source_output_alias,
    )

    source_path = Path(source)
    targets = import_output_targets(out)
    reject_source_output_alias({"source config": source_path}, targets)
    compiled = load_verl_opd_v08(source_path, overrides=overrides)
    rendered = yaml.safe_dump(
        compiled.source.model_dump(mode="python"),
        sort_keys=False,
        allow_unicode=True,
        width=100,
    ).encode("utf-8")
    # Prove the exact bytes about to be published remain executable without
    # relying on the first in-memory model instance.
    reparsed = yaml.safe_load(rendered)
    validated = compile_verl_opd_v08(
        reparsed,
        reinterpretations_accepted=bool(compiled.reinterpretation_acceptance["accepted"]),
        acceptance_source=str(compiled.reinterpretation_acceptance["source"]),
    )
    report = {
        "schema_version": 1,
        "status": "accepted",
        "profile": VERL_OPD_V08_PROFILE,
        "target_verl": {"tag": VERL_TAG, "commit": VERL_COMMIT},
        "source_config_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "generated_profile_sha256": hashlib.sha256(rendered).hexdigest(),
        "generated_profile_validated": validated.executable,
        "environment_required": False,
        "compiled_digest": compiled.compiled_digest,
        "round_trip_compiled_digest": validated.compiled_digest,
        "field_classification": [item.model_dump(mode="json") for item in compiled.compatibility],
        "generated_path": targets["recipe"].name,
        "report_path": targets["report"].name,
        "claim": (
            "Runnable only through the documented pure-OPD single-GPU profile; "
            "not arbitrary verl YAML or distributed execution."
        ),
    }
    transaction = OutputTransaction(
        targets=targets,
        stem=targets["recipe"].stem,
        lock_root=targets["recipe"].parent,
        overwrite=overwrite,
    )
    transaction.begin()
    try:
        transaction.write_bytes("recipe", rendered)
        transaction.write_json("report", report)
        transaction.discard("template")
        transaction.commit()
    finally:
        transaction.close()
    return report
