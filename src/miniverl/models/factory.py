"""Build the student/teacher pair described by a :class:`RunConfig`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from miniverl.agent.transcript import TokenizerLike
from miniverl.config.models import ModelBackend, ModelsConfig, RunConfig
from miniverl.errors import ConfigError, TokenizerMismatchError
from miniverl.models.base import CausalLMBackend
from miniverl.models.tokenizers import HFTokenizerAdapter, ToyTokenizer
from miniverl.utils.lazy import have_module

__all__ = ["BackendBundle", "resolve_device", "build_tokenizer", "build_student", "build_teacher"]


@dataclass
class BackendBundle:
    """A loaded student/teacher pair sharing one tokenizer."""

    tokenizer: TokenizerLike
    student: CausalLMBackend
    teacher: CausalLMBackend | None
    device: str

    def describe(self) -> dict[str, Any]:
        """Manifest-ready description."""
        return {
            "device": self.device,
            "tokenizer_fingerprint": self.tokenizer.fingerprint,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "student": self.student.capabilities.to_dict(),
            "teacher": self.teacher.capabilities.to_dict() if self.teacher else None,
        }


def resolve_device(models: ModelsConfig) -> str:
    """Turn ``models.device: auto`` into ``cuda`` or ``cpu``."""
    if models.device != "auto":
        if models.device == "cuda" and not _cuda_available():
            raise ConfigError(
                "models.device is 'cuda' but no CUDA device is visible to torch",
                hint="run `miniverl doctor` to see what torch reports, or set models.device: auto",
            )
        return models.device
    return "cuda" if _cuda_available() else "cpu"


def _cuda_available() -> bool:
    if not have_module("torch"):
        return False
    import torch

    return bool(torch.cuda.is_available())


def build_tokenizer(config: RunConfig, *, local_files_only: bool = False) -> TokenizerLike:
    """Load the single tokenizer shared by student and teacher.

    miniVERL currently requires an identical tokenizer on both sides. When the teacher
    declares its own tokenizer id, it is loaded and fingerprint-compared rather
    than trusted, and a mismatch raises instead of degrading silently.
    """
    models = config.models
    if models.backend is ModelBackend.TOY:
        return ToyTokenizer()

    student = models.student
    teacher = models.teacher
    student_tok = HFTokenizerAdapter.load(
        student.tokenizer_id or student.model_id,
        revision=student.tokenizer_revision or student.revision,
        trust_remote_code=student.trust_remote_code,
        local_files_only=local_files_only,
    )
    teacher_id = teacher.tokenizer_id or teacher.model_id
    if teacher_id != (student.tokenizer_id or student.model_id):
        teacher_tok = HFTokenizerAdapter.load(
            teacher_id,
            revision=teacher.tokenizer_revision or teacher.revision,
            trust_remote_code=teacher.trust_remote_code,
            local_files_only=local_files_only,
        )
        if teacher_tok.fingerprint != student_tok.fingerprint:
            raise TokenizerMismatchError(
                f"the student tokenizer ({student.tokenizer_id or student.model_id}) and the "
                f"teacher tokenizer ({teacher_id}) tokenize differently",
                hint="miniVERL currently supports same-tokenizer distillation only. Pick a "
                "teacher from the same model family, e.g. Qwen/Qwen3-0.6B with "
                "Qwen/Qwen3-1.7B. Cross-tokenizer distillation is a roadmap item "
                "(docs/limitations.md).",
            )
    return student_tok


def build_student(
    config: RunConfig,
    tokenizer: TokenizerLike,
    *,
    device: str,
    local_files_only: bool = False,
) -> CausalLMBackend:
    """Load the trainable policy."""
    models = config.models
    if models.backend is ModelBackend.TOY:
        from miniverl.models.toy import ToyBackend

        assert isinstance(tokenizer, ToyTokenizer)
        toy = models.student.toy
        return ToyBackend(
            tokenizer=tokenizer,
            model_id=models.student.model_id,
            hidden_size=toy.hidden_size,
            num_layers=toy.num_layers,
            num_heads=toy.num_heads,
            intermediate_size=toy.intermediate_size,
            max_position_embeddings=toy.max_position_embeddings,
            seed=config.run.seed,
            device=device,
            trainable=True,
        )

    from miniverl.models.hf import HFBackend

    return HFBackend.load(
        models.student,
        device=device,
        tokenizer=tokenizer,
        trainable=True,
        local_files_only=local_files_only,
    )


def build_teacher(
    config: RunConfig,
    tokenizer: TokenizerLike,
    *,
    device: str,
    local_files_only: bool = False,
) -> CausalLMBackend:
    """Load the frozen scoring model."""
    models = config.models
    if models.backend is ModelBackend.TOY:
        from miniverl.models.toy import ToyBackend

        assert isinstance(tokenizer, ToyTokenizer)
        toy = models.teacher.toy
        return ToyBackend(
            tokenizer=tokenizer,
            model_id=models.teacher.model_id,
            hidden_size=toy.hidden_size,
            num_layers=toy.num_layers,
            num_heads=toy.num_heads,
            intermediate_size=toy.intermediate_size,
            max_position_embeddings=toy.max_position_embeddings,
            seed=models.teacher.toy_teacher_seed,
            device=device,
            trainable=True,  # briefly fitted on oracle traces, then frozen
        )

    from miniverl.models.hf import HFBackend

    return HFBackend.load(
        models.teacher,
        device=device,
        tokenizer=tokenizer,
        trainable=False,
        local_files_only=local_files_only,
    )
