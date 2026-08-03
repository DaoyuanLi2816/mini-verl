"""Failure-safe adapter role switching on one physical backbone."""

from __future__ import annotations

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")


class _FakeMultiAdapterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.ones(2, 2))
        self.lora_A = torch.nn.ParameterDict(
            {
                "student": torch.nn.Parameter(torch.full((2, 2), 2.0)),
                "teacher": torch.nn.Parameter(torch.full((2, 2), 3.0)),
                "reference": torch.nn.Parameter(torch.full((2, 2), 4.0)),
            }
        )
        self.active_adapter = "student"
        self.fail_on: str | None = None

    def set_adapter(self, name: str) -> None:
        self.active_adapter = name
        if name == self.fail_on:
            raise RuntimeError(f"cannot activate {name}")


def test_only_student_adapter_is_optimizer_visible() -> None:
    from miniverl.models.shared import AdapterRoleController, PolicyRole

    model = _FakeMultiAdapterModel()
    controller = AdapterRoleController(
        model,
        role_adapters={
            PolicyRole.ACTOR: "student",
            PolicyRole.TEACHER: "teacher",
            PolicyRole.REFERENCE: "reference",
        },
    )

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {"lora_A.student"}
    assert controller.student_parameter_names == ("lora_A.student",)
    assert controller.active_role is PolicyRole.ACTOR


def test_teacher_and_reference_switches_are_frozen_and_restore_actor() -> None:
    from miniverl.models.shared import AdapterRoleController, PolicyRole

    model = _FakeMultiAdapterModel()
    controller = AdapterRoleController(
        model,
        role_adapters={
            PolicyRole.ACTOR: "student",
            PolicyRole.TEACHER: "teacher",
            PolicyRole.REFERENCE: "reference",
        },
    )

    with controller.activate(PolicyRole.TEACHER):
        assert model.active_adapter == "teacher"
        assert not any(parameter.requires_grad for parameter in model.parameters())
        with controller.activate(PolicyRole.REFERENCE):
            assert model.active_adapter == "reference"
            assert not any(parameter.requires_grad for parameter in model.parameters())
        assert model.active_adapter == "teacher"
    assert model.active_adapter == "student"
    assert model.lora_A["student"].requires_grad
    assert not model.lora_A["teacher"].requires_grad
    assert not model.lora_A["reference"].requires_grad


def test_role_switch_restores_actor_after_body_or_construction_failure() -> None:
    from miniverl.models.shared import AdapterRoleController, PolicyRole

    model = _FakeMultiAdapterModel()
    controller = AdapterRoleController(
        model,
        role_adapters={PolicyRole.ACTOR: "student", PolicyRole.TEACHER: "teacher"},
    )
    with (
        pytest.raises(ValueError, match="body failed"),
        controller.activate(PolicyRole.TEACHER),
    ):
        raise ValueError("body failed")
    assert controller.active_role is PolicyRole.ACTOR
    assert model.active_adapter == "student"
    assert controller.student_parameters[0].requires_grad

    model.fail_on = "teacher"
    with (
        pytest.raises(RuntimeError, match="cannot activate teacher"),
        controller.activate(PolicyRole.TEACHER),
    ):
        pass
    assert controller.active_role is PolicyRole.ACTOR
    assert model.active_adapter == "student"
    assert controller.student_parameters[0].requires_grad
