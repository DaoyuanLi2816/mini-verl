"""One static placement capability model shared by OPD planning and runtime guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from miniverl.errors import ConfigError

Placement = Literal[
    "dual_model_resident", "shared_backbone", "swap", "requires_probe", "infeasible"
]


@dataclass(frozen=True)
class PlacementDecision:
    """Static legality and proof status for one requested local placement."""

    strategy: Placement
    reason: str
    executable_without_probe: bool
    placement_not_proven: bool
    swap_feasible: bool
    resident_feasible: bool | None
    shared_backbone_feasible: bool


def _quantized(value: str) -> bool:
    return value != "none"


def decide_placement(
    *,
    requested: str,
    student_quantization: str,
    teacher_quantization: str,
    resident_feasible: bool | None,
    shared_backbone_feasible: bool,
) -> PlacementDecision:
    """Resolve a placement without inventing feasibility for unknown model sizes."""
    swap_feasible = not (_quantized(student_quantization) or _quantized(teacher_quantization))
    if requested == "swap":
        if not swap_feasible:
            raise ConfigError(
                "swap is illegal for bitsandbytes-quantized actor or teacher parameters",
                hint="use resident phased roles for QLoRA, or unquantized LoRA for swap",
            )
        return PlacementDecision(
            "swap",
            "runtime mode was set explicitly to swap",
            True,
            False,
            True,
            resident_feasible,
            shared_backbone_feasible,
        )
    if requested == "shared_backbone":
        if not shared_backbone_feasible:
            raise ConfigError(
                "shared_backbone requires the student and teacher to use the same base model",
                hint="use auto, dual_model_resident, or legal unquantized swap",
            )
        return PlacementDecision(
            "shared_backbone",
            "runtime mode was set explicitly to shared_backbone",
            True,
            False,
            swap_feasible,
            resident_feasible,
            True,
        )
    if requested == "dual_model_resident":
        if resident_feasible is None:
            return PlacementDecision(
                "requires_probe",
                "dual_model_resident feasibility is unknown until a bounded resident probe",
                False,
                True,
                swap_feasible,
                None,
                shared_backbone_feasible,
            )
        if not resident_feasible:
            return PlacementDecision(
                "infeasible",
                "estimated resident demand exceeds configured usable VRAM",
                False,
                False,
                swap_feasible,
                False,
                shared_backbone_feasible,
            )
        return PlacementDecision(
            "dual_model_resident",
            "runtime mode was set explicitly to dual_model_resident",
            True,
            False,
            swap_feasible,
            True,
            shared_backbone_feasible,
        )
    if requested != "auto":
        raise ConfigError(f"unknown OPD runtime placement {requested!r}")
    if resident_feasible is True:
        return PlacementDecision(
            "dual_model_resident",
            "auto -> dual_model_resident from model metadata and configured headroom",
            True,
            False,
            swap_feasible,
            True,
            shared_backbone_feasible,
        )
    if resident_feasible is False and swap_feasible:
        return PlacementDecision(
            "swap",
            "auto -> swap because estimated resident demand exceeds usable VRAM",
            True,
            False,
            True,
            False,
            shared_backbone_feasible,
        )
    if resident_feasible is None and swap_feasible:
        return PlacementDecision(
            "swap",
            "auto -> swap because model sizes are unknown and unquantized roles are movable",
            True,
            False,
            True,
            None,
            shared_backbone_feasible,
        )
    return PlacementDecision(
        "requires_probe",
        "auto cannot prove resident feasibility for unknown-size quantized roles; swap is illegal",
        False,
        True,
        False,
        resident_feasible,
        shared_backbone_feasible,
    )


def assert_runtime_placement_legal(
    *, strategy: str, student_quantization: str, teacher_quantization: str
) -> None:
    """Reject a placement that static planning can already prove impossible."""
    if strategy == "swap" and (
        _quantized(student_quantization) or _quantized(teacher_quantization)
    ):
        raise ConfigError(
            "memory.strategy=swap cannot be used with a quantized model; swap is illegal "
            "for bitsandbytes-quantized actor or teacher parameters",
            hint="use QLoRA with resident/local phased roles, or unquantized LoRA with swap",
        )
