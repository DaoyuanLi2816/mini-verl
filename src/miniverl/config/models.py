"""Pydantic v2 models describing a miniVERL run.

A :class:`RunConfig` is validated before anything is downloaded or allocated.
File-backed recipes retain exact source bytes in ``config.submitted.yaml``;
canonical validated, compatibility and runtime-resolved layers are written
separately so no normalized file is mislabeled as verbatim input.

Cross-field validation lives here on purpose: it is cheaper and far friendlier
to reject ``exact_full_vocab`` + ``top_k: 64`` at parse time than to discover
the contradiction three minutes into a GPU run.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from miniverl.errors import ConfigError
from miniverl.utils.runs import write_text

__all__ = [
    "TrainingMode",
    "OfflineKDTrajectorySource",
    "OPDFreshness",
    "AdapterSource",
    "ModelBackend",
    "Precision",
    "Quantization",
    "TeacherContextMode",
    "LossMode",
    "Divergence",
    "SelectorName",
    "MemoryStrategy",
    "OptimizerName",
    "LRSchedule",
    "ToyModelConfig",
    "LoRAConfig",
    "StudentModelConfig",
    "TeacherModelConfig",
    "TeacherAdapterConfig",
    "ModelsConfig",
    "LossConfig",
    "SelectionConfig",
    "RolloutConfig",
    "EnvironmentConfig",
    "TrainConfig",
    "MemoryConfig",
    "CacheConfig",
    "OfflineKDConfig",
    "EvalConfig",
    "ReportConfig",
    "RunMeta",
    "RunConfig",
    "CONFIG_SCHEMA_VERSION",
]

CONFIG_SCHEMA_VERSION = 1


class TrainingMode(str, Enum):
    """Which training loop to run."""

    SFT = "sft"
    OFFLINE_KD = "offline_kd"
    OPD = "opd"


class OfflineKDTrajectorySource(str, Enum):
    """Where the immutable state distribution for offline KD comes from."""

    ORACLE = "oracle"
    FROZEN_STUDENT = "frozen_student"
    PERSISTED = "persisted"


class OPDFreshness(str, Enum):
    """Whether each OPD update consumes a newly sampled rollout batch."""

    STRICT = "strict"
    REPLAY = "replay"


class AdapterSource(str, Enum):
    """Where a frozen PEFT teacher adapter is loaded from."""

    LOCAL = "local"
    HUB = "hub"


class ModelBackend(str, Enum):
    """Where student/teacher weights come from."""

    TOY = "toy"
    HF = "hf"


class Precision(str, Enum):
    """Compute dtype for model weights and activations."""

    AUTO = "auto"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


class Quantization(str, Enum):
    """Weight quantization scheme."""

    NONE = "none"
    NF4 = "nf4"
    INT8 = "int8"


class TeacherContextMode(str, Enum):
    """What the teacher is allowed to see."""

    STANDARD = "standard"
    PRIVILEGED_CONTEXT = "privileged_context"


class LossMode(str, Enum):
    """Vocabulary treatment of the divergence."""

    EXACT_FULL_VOCAB = "exact_full_vocab"
    BUCKETED_TOPK_TAIL = "bucketed_topk_tail"


class Divergence(str, Enum):
    """Which divergence to minimize between teacher and student."""

    FORWARD_KL = "forward_kl"
    REVERSE_KL = "reverse_kl"
    JSD = "jsd"


class SelectorName(str, Enum):
    """Teacher-position budget policy."""

    ALL_MODEL_TOKENS = "all_model_tokens"
    TOOL_AND_FINAL = "tool_and_final"
    UNIFORM_RATIO = "uniform_ratio"
    UNIFORM_BUDGET = "uniform_budget"
    HYBRID = "hybrid"


class MemoryStrategy(str, Enum):
    """How teacher and student share the accelerator."""

    RESIDENT = "resident"
    SWAP = "swap"
    AUTO = "auto"


class OptimizerName(str, Enum):
    """Supported optimizers."""

    ADAMW = "adamw"
    ADAMW_8BIT = "adamw8bit"


class LRSchedule(str, Enum):
    """Learning-rate schedules."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class ToyModelConfig(_Base):
    """Shape of the built-in tiny transformer used by the ``toy`` backend."""

    hidden_size: int = Field(default=64, ge=8, le=512)
    num_layers: int = Field(default=2, ge=1, le=8)
    num_heads: int = Field(default=4, ge=1, le=16)
    intermediate_size: int = Field(default=128, ge=8, le=2048)
    max_position_embeddings: int = Field(default=1024, ge=64, le=8192)

    @model_validator(mode="after")
    def _check_heads(self) -> ToyModelConfig:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by num_heads={self.num_heads}"
            )
        return self


class LoRAConfig(_Base):
    """QLoRA / LoRA adapter configuration for the student."""

    enabled: bool = True
    r: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = Field(default="none", pattern="^(none|all|lora_only)$")


class StudentModelConfig(_Base):
    """The policy being trained."""

    model_id: str
    revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    dtype: Precision = Precision.AUTO
    quantization: Quantization = Quantization.NONE
    attn_implementation: str = Field(default="sdpa", pattern="^(sdpa|eager)$")
    gradient_checkpointing: bool = False
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    toy: ToyModelConfig = Field(default_factory=ToyModelConfig)
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def _check_quant(self) -> StudentModelConfig:
        if self.quantization is not Quantization.NONE and not self.lora.enabled:
            raise ValueError(
                "a quantized student must be trained with LoRA adapters "
                "(set models.student.lora.enabled: true)"
            )
        return self


class TeacherAdapterConfig(_Base):
    """Standard PEFT adapter applied to the frozen teacher base model."""

    path: str = Field(min_length=1)
    source: AdapterSource = AdapterSource.LOCAL
    revision: str | None = None
    base_model_revision: str | None = None
    tokenizer_fingerprint: str | None = None
    require_policy_evaluation: bool = False
    minimum_strict_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _pin_hub_adapter(self) -> TeacherAdapterConfig:
        if self.source is AdapterSource.HUB and not self.revision:
            raise ValueError(
                "a Hub teacher adapter must pin adapter.revision; moving adapter "
                "branches are not reproducible"
            )
        if self.minimum_strict_success_rate is not None and not self.require_policy_evaluation:
            raise ValueError(
                "adapter.minimum_strict_success_rate requires "
                "adapter.require_policy_evaluation=true"
            )
        return self


class TeacherModelConfig(_Base):
    """The frozen scoring model."""

    model_id: str
    revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    dtype: Precision = Precision.AUTO
    quantization: Quantization = Quantization.NONE
    attn_implementation: str = Field(default="sdpa", pattern="^(sdpa|eager)$")
    mode: TeacherContextMode = TeacherContextMode.STANDARD
    toy: ToyModelConfig = Field(default_factory=ToyModelConfig)
    trust_remote_code: bool = False
    adapter: TeacherAdapterConfig | None = None
    #: Deterministic perturbation used only by the ``toy`` backend so the toy
    #: teacher is a *different* distribution from the toy student.
    toy_teacher_seed: int = Field(default=99, ge=0)
    #: ``toy`` backend only: cross-entropy steps fitting the teacher to oracle
    #: traces before it supervises anything.  A randomly initialized teacher is
    #: a uniform-noise oracle, so distilling from it would demonstrate nothing.
    #: The fit is an explicit, logged phase written into the run directory.
    toy_pretrain_steps: int = Field(default=120, ge=0, le=100000)
    toy_pretrain_lr: float = Field(default=3e-3, gt=0.0, le=1.0)


class ModelsConfig(_Base):
    """Student/teacher pair. The current implementation requires one tokenizer."""

    backend: ModelBackend = ModelBackend.TOY
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
    student: StudentModelConfig
    teacher: TeacherModelConfig


class LossConfig(_Base):
    """Divergence objective and its vocabulary treatment."""

    mode: LossMode = LossMode.BUCKETED_TOPK_TAIL
    divergence: Divergence = Divergence.REVERSE_KL
    temperature: float = Field(default=1.0, gt=0.0, le=20.0)
    scale_by_temperature_squared: bool = True
    top_k: int = Field(default=64, ge=1, le=262144)
    jsd_beta: float = Field(default=0.5, ge=0.0, le=1.0)
    tail_epsilon: float = Field(default=1e-9, gt=0.0, lt=1e-2)
    #: Number of selected prediction positions projected through the LM head at
    #: once.  Purely a memory/throughput knob -- it does not change the loss.
    chunk_size: int = Field(default=256, ge=1, le=65536)
    #: Guard rail: refuse `exact_full_vocab` above this vocabulary size unless
    #: explicitly overridden, because it materializes ``[chunk, vocab]`` fp32.
    exact_max_vocab: int = Field(default=8192, ge=1)
    allow_large_exact: bool = False
    #: NLL on the token sampled by the student in the current trajectory. This is
    #: deliberately not called supervised CE: the target is not an oracle token.
    sampled_token_nll_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_ce_weight(cls, value: Any) -> Any:
        """Accept the v0.1 key without relying on runtime alias-class identity."""
        if not isinstance(value, dict) or "ce_weight" not in value:
            return value
        migrated = dict(value)
        if "sampled_token_nll_weight" in migrated:
            return migrated
        migrated["sampled_token_nll_weight"] = migrated.pop("ce_weight")
        return migrated

    @property
    def ce_weight(self) -> float:
        """Compatibility accessor for v0.1 callers; new configs use the explicit name."""
        return self.sampled_token_nll_weight


class SelectionConfig(_Base):
    """Which model-generated positions the teacher is asked to score."""

    selector: SelectorName = SelectorName.ALL_MODEL_TOKENS
    ratio: float = Field(default=0.35, gt=0.0, le=1.0)
    max_positions_per_trajectory: int | None = Field(default=None, ge=1)
    #: Relative weight applied to critical (tool-call / final) tokens.
    critical_weight: float = Field(default=1.0, gt=0.0, le=100.0)
    other_weight: float = Field(default=1.0, gt=0.0, le=100.0)


class RolloutConfig(_Base):
    """Bounds and sampling parameters for student rollouts."""

    max_turns: int = Field(default=4, ge=1, le=64)
    max_new_tokens_per_turn: int = Field(default=64, ge=1, le=8192)
    max_total_tokens: int = Field(default=768, ge=16, le=131072)
    temperature: float = Field(default=1.0, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    # Maximum parse errors tolerated in an episode. The rollout terminates as
    # soon as the count reaches this limit; zero therefore terminates on the
    # first parse error.
    max_parse_errors: int = Field(default=2, ge=0, le=32)
    max_repeated_calls: int = Field(default=2, ge=1, le=32)


class EnvironmentConfig(_Base):
    """Which deterministic local environment to train and evaluate on."""

    name: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    train_tasks: int = Field(default=64, ge=1, le=100000)
    eval_tasks: int = Field(default=32, ge=1, le=100000)
    test_tasks: int = Field(default=32, ge=0, le=100000)
    split_seed: int = Field(default=7, ge=0)
    difficulty: str = Field(default="easy", pattern="^(easy|medium|hard)$")

    @field_validator("name")
    @classmethod
    def _known_environment(cls, value: str) -> str:
        """Check against the registry rather than a hard-coded list.

        A literal pattern here would have made
        :func:`miniverl.environments.registry.register` useless: a custom
        environment would register successfully and then be rejected by config
        validation. The import is inside the validator so that
        ``import miniverl.config`` stays light.
        """
        from miniverl.environments.registry import available_environments

        known = available_environments()
        if value not in known:
            raise ValueError(
                f"unknown environment {value!r}; registered environments are "
                f"{', '.join(known)}. Custom environments must be imported (which "
                "runs their @register decorator) before the recipe is validated."
            )
        return value


class TrainConfig(_Base):
    """Optimization schedule."""

    #: ``0`` is legal and means "train nothing": load, evaluate, checkpoint.
    #: The benchmark harness uses it for a pure cold-start arm.
    cycles: int = Field(default=4, ge=0, le=100000)
    rollouts_per_cycle: int = Field(default=8, ge=1, le=8192)
    #: Trajectories per optimizer step. The trainer runs one trajectory per forward
    #: pass (no padded batching), so this *is* the effective batch size and the
    #: number of steps per cycle is ``ceil(rollouts_per_cycle / this)``.
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=1024)
    #: ``strict`` permits one optimizer update per freshly sampled rollout
    #: batch. ``replay`` explicitly permits multiple updates from that batch and
    #: must never be reported as genuine on-policy distillation.
    opd_freshness: OPDFreshness = OPDFreshness.STRICT
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0, le=1e4)
    warmup_steps: int = Field(default=0, ge=0, le=100000)
    lr_schedule: LRSchedule = LRSchedule.CONSTANT
    optimizer: OptimizerName = OptimizerName.ADAMW
    adam_beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    adam_eps: float = Field(default=1e-8, gt=0.0, lt=1.0)
    save_every_cycles: int = Field(default=0, ge=0, le=100000)
    eval_every_cycles: int = Field(default=0, ge=0, le=100000)
    #: Optional SFT cold start executed before the KD/OPD loop.
    sft_warmup_cycles: int = Field(default=0, ge=0, le=100000)
    log_every_steps: int = Field(default=1, ge=1, le=10000)
    #: Optional continuation-only budgets. Each is checked after an optimizer
    #: step/cycle, so the exact nonnegative overshoot is part of the result.
    max_selected_training_tokens: int | None = Field(default=None, ge=1)
    max_wall_seconds: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _one_stop_budget(self) -> TrainConfig:
        if self.max_selected_training_tokens is not None and self.max_wall_seconds is not None:
            raise ValueError(
                "train accepts at most one continuation stop budget: selected tokens or wall time"
            )
        return self


class MemoryConfig(_Base):
    """Consumer-GPU memory controls."""

    strategy: MemoryStrategy = MemoryStrategy.AUTO
    #: Bounded retries on CUDA OOM.  Retries only halve the projection chunk
    #: size, which is mathematically neutral.
    oom_retries: int = Field(default=3, ge=0, le=10)
    min_chunk_size: int = Field(default=8, ge=1, le=4096)
    empty_cache_between_phases: bool = True
    reset_peak_stats_each_cycle: bool = True
    #: Fraction of total VRAM below which ``auto`` prefers ``swap``.
    auto_swap_vram_headroom_gb: float = Field(default=2.0, ge=0.0, le=1024.0)


class CacheConfig(_Base):
    """Teacher-target cache policy."""

    dir: str | None = None
    entries_per_shard: int = Field(default=32, ge=1, le=4096)
    #: ``float32`` round-trips the targets exactly; ``float16`` halves the
    #: log-probability payload at the cost of ~1e-3 relative precision.
    dtype: str = Field(default="float32", pattern="^(float32|float16)$")
    #: Reject entries whose ``policy_version`` differs from the consuming
    #: update.  Must stay ``True`` for genuine on-policy distillation.
    strict_policy_version: bool = True
    #: ``offline_kd`` only: allow the same fixed cache across every update.
    reuse_across_policy_versions: bool = False
    keep_cycles: int = Field(default=1, ge=1, le=100000)
    verify_checksums_on_load: bool = True


class OfflineKDConfig(_Base):
    """Explicit provenance and collection policy for fixed-state distillation."""

    trajectory_source: OfflineKDTrajectorySource = OfflineKDTrajectorySource.ORACLE
    dataset_path: str | None = None
    collection_seed: int = Field(default=1234, ge=0)
    collection_tasks: int | None = Field(default=None, ge=1, le=100000)
    task_schedule_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_source(self) -> OfflineKDConfig:
        if self.trajectory_source is OfflineKDTrajectorySource.PERSISTED and not self.dataset_path:
            raise ValueError(
                "offline_kd.trajectory_source=persisted requires offline_kd.dataset_path"
            )
        if self.trajectory_source is not OfflineKDTrajectorySource.PERSISTED and self.dataset_path:
            raise ValueError(
                "offline_kd.dataset_path is only valid when trajectory_source=persisted"
            )
        return self


class EvalConfig(_Base):
    """Deterministic evaluation settings."""

    enabled: bool = True
    baseline_enabled: bool = True
    tasks: int | None = Field(default=None, ge=1)
    split: str = Field(default="eval", pattern="^(train|eval|test)$")
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    max_turns: int | None = Field(default=None, ge=1)
    seed: int = Field(default=0, ge=0)


class ReportConfig(_Base):
    """Report generation knobs."""

    enabled: bool = True
    max_trajectories: int = Field(default=5, ge=0, le=200)
    max_tokens_per_trajectory: int = Field(default=400, ge=0, le=8192)


class RunMeta(_Base):
    """Identity and reproducibility settings for the run."""

    name: str = Field(default="miniverl-run", min_length=1, max_length=120)
    mode: TrainingMode = TrainingMode.OPD
    seed: int = Field(default=1234, ge=0)
    output_dir: str = "runs"
    run_id: str | None = None
    deterministic: bool = True
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


class RunConfig(_Base):
    """Top-level miniVERL run configuration."""

    _submitted_bytes: bytes | None = PrivateAttr(default=None)
    _source_path: Path | None = PrivateAttr(default=None)

    schema_version: int = CONFIG_SCHEMA_VERSION
    run: RunMeta = Field(default_factory=RunMeta)
    models: ModelsConfig
    environment: EnvironmentConfig
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    offline_kd: OfflineKDConfig = Field(default_factory=OfflineKDConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # -- cross-field validation ----------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _reject_ambiguous_legacy_ce(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        run = value.get("run") or {}
        loss = value.get("loss") or {}
        if (
            isinstance(run, dict)
            and isinstance(loss, dict)
            and run.get("mode", TrainingMode.OPD.value) != TrainingMode.SFT.value
            and "ce_weight" in loss
            and float(loss.get("ce_weight") or 0.0) > 0.0
        ):
            raise ValueError(
                "loss.ce_weight is ambiguous in distillation modes. Rename it to "
                "loss.sampled_token_nll_weight to state that the targets are tokens "
                "sampled by the student, not oracle SFT labels."
            )
        return value

    @model_validator(mode="after")
    def _validate_combination(self) -> RunConfig:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"config schema_version {self.schema_version} is not supported by this "
                f"miniVERL build (expected {CONFIG_SCHEMA_VERSION})"
            )

        mode = self.run.mode
        if mode is TrainingMode.OPD and self.cache.reuse_across_policy_versions:
            raise ValueError(
                "cache.reuse_across_policy_versions=true contradicts run.mode=opd: "
                "reusing one teacher cache across policy versions is offline KD. "
                "Set run.mode: offline_kd, or keep the cache strictly per-cycle."
            )
        if mode is TrainingMode.OPD and not self.cache.strict_policy_version:
            raise ValueError(
                "run.mode=opd requires cache.strict_policy_version=true so teacher "
                "targets can never be consumed by a different policy version"
            )
        if mode is TrainingMode.OFFLINE_KD and not self.cache.reuse_across_policy_versions:
            raise ValueError(
                "run.mode=offline_kd needs cache.reuse_across_policy_versions=true; "
                "that flag is what makes the offline (fixed-target) semantics explicit"
            )
        if mode is not TrainingMode.OFFLINE_KD and self.offline_kd != OfflineKDConfig():
            raise ValueError("offline_kd settings apply only when run.mode=offline_kd")

        if self.loss.divergence is Divergence.JSD and not 0.0 < self.loss.jsd_beta < 1.0:
            raise ValueError(
                f"loss.jsd_beta must be strictly inside (0, 1) for divergence=jsd, got "
                f"{self.loss.jsd_beta}. At either endpoint the mixture collapses onto one "
                "input and the divergence is identically zero."
            )

        if self.loss.mode is LossMode.EXACT_FULL_VOCAB:
            if self.cache.dtype != "float32":
                raise ValueError(
                    "loss.mode=exact_full_vocab requires cache.dtype=float32; a float16 "
                    "teacher-probability cache would make the advertised exact objective lossy"
                )
            # top_k is meaningless in exact mode. Reject it only when the recipe
            # *explicitly* set it to something other than 1, so that omitting the
            # key entirely -- the remedy the message suggests -- actually works.
            explicit = "top_k" in self.loss.model_fields_set
            if explicit and self.loss.top_k != 1:
                raise ValueError(
                    "loss.mode=exact_full_vocab ignores loss.top_k. Remove top_k from "
                    "the recipe (or set it to 1) to avoid implying a truncation that "
                    "does not happen."
                )
            # Normalize so that `to_yaml()` -- which writes every field, including
            # defaults -- produces a config.resolved.yaml that loads again. The
            # standalone evaluator re-reads exactly that file. `model_copy` rather
            # than in-place assignment: pydantic reuses a caller-supplied
            # LossConfig by reference, and rewriting it would be a surprising
            # side effect of merely constructing a RunConfig.
            self.loss = self.loss.model_copy(update={"top_k": 1})

        if mode is TrainingMode.SFT and self.loss.sampled_token_nll_weight not in (0.0, 1.0):
            raise ValueError(
                "run.mode=sft trains with oracle cross-entropy only; "
                "loss.sampled_token_nll_weight must be 0.0 (implicit) or 1.0 "
                "(explicit)"
            )

        steps_per_rollout_batch = max(
            1,
            (self.train.rollouts_per_cycle + self.train.gradient_accumulation_steps - 1)
            // self.train.gradient_accumulation_steps,
        )
        if (
            mode is TrainingMode.OPD
            and self.train.opd_freshness is OPDFreshness.STRICT
            and steps_per_rollout_batch != 1
        ):
            raise ValueError(
                "train.opd_freshness=strict requires exactly one optimizer step per "
                "fresh rollout batch. Set train.gradient_accumulation_steps greater "
                "than or equal to train.rollouts_per_cycle, or explicitly select "
                "train.opd_freshness: replay (which is not genuine OPD)."
            )

        if self.train.sft_warmup_cycles > 0 and mode is TrainingMode.SFT:
            raise ValueError(
                "train.sft_warmup_cycles applies to offline_kd/opd runs; it is "
                "redundant when run.mode=sft"
            )

        if self.models.backend is ModelBackend.TOY:
            if self.models.student.quantization is not Quantization.NONE:
                raise ValueError(
                    "the toy backend does not support quantization "
                    "(models.student.quantization must be 'none')"
                )
            if self.models.teacher.quantization is not Quantization.NONE:
                raise ValueError(
                    "the toy backend does not support quantization "
                    "(models.teacher.quantization must be 'none')"
                )
            if self.models.teacher.adapter is not None:
                raise ValueError(
                    "the toy backend cannot load a PEFT teacher adapter; use models.backend: hf"
                )

        if self.rollout.max_total_tokens <= self.rollout.max_new_tokens_per_turn:
            raise ValueError("rollout.max_total_tokens must exceed rollout.max_new_tokens_per_turn")

        if self.eval.max_turns is not None and self.eval.max_turns > self.rollout.max_turns * 4:
            raise ValueError("eval.max_turns is implausibly larger than rollout.max_turns")

        return self

    # -- convenience ----------------------------------------------------

    @property
    def is_on_policy(self) -> bool:
        """``True`` only when the full strict OPD freshness contract holds."""
        steps_per_batch = max(
            1,
            (self.train.rollouts_per_cycle + self.train.gradient_accumulation_steps - 1)
            // self.train.gradient_accumulation_steps,
        )
        return (
            self.run.mode is TrainingMode.OPD
            and self.train.opd_freshness is OPDFreshness.STRICT
            and steps_per_batch == 1
            and self.cache.strict_policy_version
            and not self.cache.reuse_across_policy_versions
        )

    @property
    def effective_eval_tasks(self) -> int:
        """Number of evaluation tasks after applying the eval override."""
        if self.eval.tasks is not None:
            return self.eval.tasks
        return self.environment.eval_tasks

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load and validate a recipe from a YAML file."""
        p = Path(path)
        if not p.is_file():
            raise ConfigError(
                f"recipe not found: {p}",
                hint="run `miniverl validate <path>` with an existing YAML recipe, "
                "or copy one from the recipes/ directory",
            )
        submitted = p.read_bytes()
        try:
            text = submitted.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError(f"{p} is not valid UTF-8: {exc}") from exc
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{p} is not valid YAML: {exc}") from exc
        if raw is None:
            raise ConfigError(f"{p} is empty")
        if not isinstance(raw, dict):
            raise ConfigError(f"{p} must contain a YAML mapping at the top level")
        config = cls.model_validate(raw)
        config._submitted_bytes = submitted
        config._source_path = p.resolve()
        return config

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> RunConfig:
        """Validate a already-parsed mapping."""
        return cls.model_validate(data)

    @property
    def submitted_bytes(self) -> bytes | None:
        """Exact source bytes for a file-backed recipe, otherwise ``None``."""
        private = getattr(self, "__pydantic_private__", None)
        value = private.get("_submitted_bytes") if isinstance(private, dict) else None
        return value if isinstance(value, bytes) else None

    def resolved_for_runtime(self) -> RunConfig:
        """Return a deep copy with local paths resolved, without mutating provenance."""
        runtime = self.model_copy(deep=True)
        adapter = runtime.models.teacher.adapter
        if adapter is not None and adapter.source is AdapterSource.LOCAL:
            adapter_path = Path(adapter.path)
            if not adapter_path.is_absolute():
                private = getattr(self, "__pydantic_private__", None)
                source = private.get("_source_path") if isinstance(private, dict) else None
                base = source.parent if isinstance(source, Path) else Path.cwd()
                adapter.path = str((base / adapter_path).resolve())
        dataset_path = runtime.offline_kd.dataset_path
        if dataset_path:
            path = Path(dataset_path)
            if not path.is_absolute():
                private = getattr(self, "__pydantic_private__", None)
                source = private.get("_source_path") if isinstance(private, dict) else None
                base = source.parent if isinstance(source, Path) else Path.cwd()
                runtime.offline_kd.dataset_path = str((base / path).resolve())
        return runtime

    def to_yaml(self) -> str:
        """Serialize to canonical YAML (enums as their string values)."""
        payload = self.model_dump(mode="json")
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)

    def write_yaml(self, path: str | Path) -> Path:
        """Write :meth:`to_yaml` to ``path`` and return it."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_text(p, self.to_yaml())
        return p
