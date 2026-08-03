"""One-backbone multi-adapter policy roles for the consumer runtime."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from miniverl.errors import BackendError
from miniverl.models.base import CausalLMBackend, GenerationOutput

if TYPE_CHECKING:  # pragma: no cover - typing only
    from miniverl.models.hf import HFBackend
    from miniverl.training.batching import PaddedTrajectoryBatch

__all__ = [
    "PolicyRole",
    "AdapterRoleController",
    "SharedAdapterRoleBackend",
    "load_shared_adapter_backends",
]


class PolicyRole(str, Enum):
    """Semantically distinct policy roles on one physical model."""

    ACTOR = "actor"
    TEACHER = "teacher"
    REFERENCE = "reference"


class AdapterRoleController:
    """Failure-safe PEFT adapter activation and gradient ownership."""

    def __init__(self, model: Any, *, role_adapters: dict[PolicyRole, str]) -> None:
        if PolicyRole.ACTOR not in role_adapters:
            raise BackendError("shared_backbone requires an actor/student adapter")
        if len(set(role_adapters.values())) != len(role_adapters):
            raise BackendError("shared_backbone roles must use distinct adapter names")
        setter = getattr(model, "set_adapter", None)
        if not callable(setter):
            raise BackendError("shared_backbone model does not expose PEFT set_adapter()")
        self.model = model
        self.role_adapters = dict(role_adapters)
        self._lock = threading.RLock()
        self._actor_training = True
        self.active_role = PolicyRole.ACTOR
        actor_name = role_adapters[PolicyRole.ACTOR]
        self._student_named_parameters = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if self._belongs_to_adapter(name, actor_name)
        )
        if not self._student_named_parameters:
            raise BackendError(
                f"shared_backbone found no parameters for student adapter {actor_name!r}"
            )
        self._apply(PolicyRole.ACTOR)

    @staticmethod
    def _belongs_to_adapter(parameter_name: str, adapter_name: str) -> bool:
        return adapter_name in parameter_name.split(".")

    @property
    def student_parameter_names(self) -> tuple[str, ...]:
        """Stable checkpoint names owned by the student adapter."""
        return tuple(name for name, _ in self._student_named_parameters)

    @property
    def student_parameters(self) -> list[torch.nn.Parameter]:
        """Optimizer-visible student parameters, independent of active role."""
        return [parameter for _, parameter in self._student_named_parameters]

    def _apply(self, role: PolicyRole) -> None:
        if role not in self.role_adapters:
            raise BackendError(f"shared_backbone has no configured {role.value} adapter")
        self.model.set_adapter(self.role_adapters[role])
        actor_ids = {id(parameter) for _, parameter in self._student_named_parameters}
        actor_active = role is PolicyRole.ACTOR
        for parameter in self.model.parameters():
            parameter.requires_grad_(actor_active and id(parameter) in actor_ids)
        self.model.train(actor_active and self._actor_training)
        self.active_role = role

    @contextmanager
    def activate(self, role: PolicyRole) -> Iterator[None]:
        """Activate one role and restore the prior role even after failure."""
        with self._lock:
            previous = self.active_role
            if role is previous:
                yield
                return
            try:
                self._apply(role)
            except BaseException:
                self._apply(previous)
                raise
            try:
                yield
            finally:
                self._apply(previous)

    def set_actor_train(self, mode: bool) -> None:
        """Set the actor's desired mode without making frozen roles trainable."""
        with self._lock:
            self._actor_training = bool(mode)
            self._apply(self.active_role)

    def student_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU copy of the student adapter only."""
        return {
            name: parameter.detach().to("cpu").clone()
            for name, parameter in self._student_named_parameters
        }

    def load_student_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore a checkpoint without touching teacher/reference adapters."""
        own = dict(self._student_named_parameters)
        unknown = sorted(set(state).difference(own))
        missing = sorted(set(own).difference(state))
        if unknown or missing:
            raise BackendError(
                "shared-backbone student checkpoint names do not match "
                f"(unknown={unknown[:1]}, missing={missing[:1]})"
            )
        with torch.no_grad():
            for name, value in state.items():
                parameter = own[name]
                if value.shape != parameter.shape:
                    raise BackendError(
                        f"checkpoint parameter {name!r} has shape {tuple(value.shape)}, "
                        f"expected {tuple(parameter.shape)}"
                    )
                parameter.copy_(value.to(parameter.device, parameter.dtype))


class SharedAdapterRoleBackend(CausalLMBackend):
    """A role-scoped backend view over one :class:`HFBackend` instance."""

    def __init__(
        self,
        owner: HFBackend,
        controller: AdapterRoleController,
        role: PolicyRole,
        *,
        adapter_provenance: dict[str, Any] | None = None,
    ) -> None:
        self.owner = owner
        self.controller = controller
        self.role = role
        self.model = owner.model
        self.tokenizer = owner.tokenizer
        self.model_id = owner.model_id
        self.model_revision = owner.model_revision
        self.adapter_provenance = adapter_provenance
        trainable = (
            sum(parameter.numel() for parameter in controller.student_parameters)
            if role is PolicyRole.ACTOR
            else 0
        )
        self.capabilities = replace(owner.capabilities, num_trainable_parameters=trainable)

    @contextmanager
    def activated(self) -> Iterator[None]:
        """Expose explicit role activation for profiling and diagnostics."""
        with self.controller.activate(self.role):
            yield

    def generate(
        self,
        prefix_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        stop_sequences: Sequence[str] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
    ) -> GenerationOutput:
        with self.activated():
            return self.owner.generate(
                prefix_token_ids,
                max_new_tokens=max_new_tokens,
                stop_sequences=stop_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )

    def hidden_states_at(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        with self.activated():
            return self.owner.hidden_states_at(token_ids, positions, with_grad=with_grad)

    def hidden_states_at_batch(
        self,
        batch: PaddedTrajectoryBatch,
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        with self.activated():
            return self.owner.hidden_states_at_batch(batch, with_grad=with_grad)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        with self.activated():
            return self.owner.project(hidden)

    def set_train(self, mode: bool) -> None:
        if self.role is PolicyRole.ACTOR:
            self.controller.set_actor_train(mode)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return self.controller.student_parameters if self.role is PolicyRole.ACTOR else []

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return self.controller.student_state_dict() if self.role is PolicyRole.ACTOR else {}

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if self.role is not PolicyRole.ACTOR:
            raise BackendError(f"cannot load trainable state into frozen {self.role.value} role")
        self.controller.load_student_state_dict(state)

    def to_device(self, device: str) -> None:
        self.owner.to_device(device)

    def release(self) -> None:
        self.owner.release()

    @property
    def device(self) -> str:
        return self.owner.device


def load_shared_adapter_backends(
    config: Any,
    tokenizer: Any,
    *,
    device: str,
    local_files_only: bool = False,
    include_teacher: bool = True,
) -> tuple[
    SharedAdapterRoleBackend,
    SharedAdapterRoleBackend | None,
    SharedAdapterRoleBackend | None,
]:
    """Load one base once, then attach typed student/teacher/reference adapters."""
    from miniverl.config.models import TeacherModelConfig
    from miniverl.models.adapter_io import validate_teacher_adapter
    from miniverl.models.hf import HFBackend

    protocol_version = str(config.environment.params.get("protocol_version", "v1"))
    owner = HFBackend.load(
        config.models.student,
        device=device,
        tokenizer=tokenizer,
        trainable=True,
        local_files_only=local_files_only,
        student_adapter_name="student",
    )
    role_adapters = {PolicyRole.ACTOR: "student"}
    provenance: dict[PolicyRole, dict[str, Any] | None] = {PolicyRole.ACTOR: None}
    try:
        if include_teacher:
            teacher_adapter = config.models.teacher.adapter
            if teacher_adapter is None:  # pragma: no cover - config guard
                raise BackendError("shared_backbone requires a teacher adapter")
            validated = validate_teacher_adapter(
                teacher_adapter,
                config.models.teacher,
                tokenizer_fingerprint=tokenizer.fingerprint,
                protocol_version=protocol_version,
                local_files_only=local_files_only,
            )
            owner.model.load_adapter(
                str(validated.snapshot_dir),
                adapter_name="teacher",
                is_trainable=False,
                local_files_only=True,
            )
            role_adapters[PolicyRole.TEACHER] = "teacher"
            provenance[PolicyRole.TEACHER] = validated.provenance

        reference = config.models.reference
        if reference is not None:
            reference_spec = TeacherModelConfig(
                model_id=reference.model_id,
                revision=reference.revision,
                tokenizer_id=reference.tokenizer_id,
                tokenizer_revision=reference.tokenizer_revision,
                dtype=reference.dtype,
                quantization=reference.quantization,
                attn_implementation=reference.attn_implementation,
                trust_remote_code=reference.trust_remote_code,
                adapter=reference.adapter,
            )
            validated_reference = validate_teacher_adapter(
                reference.adapter,
                reference_spec,
                tokenizer_fingerprint=tokenizer.fingerprint,
                protocol_version=None,
                local_files_only=local_files_only,
            )
            owner.model.load_adapter(
                str(validated_reference.snapshot_dir),
                adapter_name="reference",
                is_trainable=False,
                local_files_only=True,
            )
            role_adapters[PolicyRole.REFERENCE] = "reference"
            provenance[PolicyRole.REFERENCE] = validated_reference.provenance

        controller = AdapterRoleController(owner.model, role_adapters=role_adapters)
        actor = SharedAdapterRoleBackend(owner, controller, PolicyRole.ACTOR)
        teacher = (
            SharedAdapterRoleBackend(
                owner,
                controller,
                PolicyRole.TEACHER,
                adapter_provenance=provenance[PolicyRole.TEACHER],
            )
            if PolicyRole.TEACHER in role_adapters
            else None
        )
        reference_backend = (
            SharedAdapterRoleBackend(
                owner,
                controller,
                PolicyRole.REFERENCE,
                adapter_provenance=provenance[PolicyRole.REFERENCE],
            )
            if PolicyRole.REFERENCE in role_adapters
            else None
        )
        return actor, teacher, reference_backend
    except BaseException:
        owner.release()
        raise
