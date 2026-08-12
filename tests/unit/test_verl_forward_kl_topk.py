from __future__ import annotations

import torch

from miniverl.losses.verl_topk import verl_forward_kl_topk


def test_forward_kl_topk_matches_pinned_formula_and_gradient() -> None:
    student = torch.tensor(
        [[0.2, -0.1, 1.4, 0.7], [-1.0, 0.5, 0.2, 1.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher_ids = torch.tensor([[2, 3], [3, 1]])
    teacher_log_probs = torch.log(torch.tensor([[0.60, 0.25], [0.55, 0.30]], dtype=torch.float64))

    output = verl_forward_kl_topk(student, teacher_log_probs, teacher_ids)
    reference_student = torch.log_softmax(student, dim=-1).gather(-1, teacher_ids)
    expected = (
        (teacher_log_probs.float().exp() * (teacher_log_probs.float() - reference_student.float()))
        .sum(dim=-1)
        .clamp_min(0.0)
    )
    expected.mean().backward()
    expected_grad = student.grad.detach().clone()
    student.grad = None
    output.loss.mean().backward()

    torch.testing.assert_close(output.loss, expected)
    torch.testing.assert_close(student.grad, expected_grad)
    torch.testing.assert_close(output.teacher_mass, teacher_log_probs.exp().sum(dim=-1))
    torch.testing.assert_close(output.student_mass, reference_student.exp().sum(dim=-1))


def test_clamps_follow_pinned_order_and_overlap_uses_clamped_terms() -> None:
    student = torch.tensor([[9.0, 8.0, -20.0, -30.0]])
    teacher_ids = torch.tensor([[2, 0]])
    teacher_log_probs = torch.tensor([[-30.0, -0.2]])

    output = verl_forward_kl_topk(
        student,
        teacher_log_probs,
        teacher_ids,
        log_prob_min_clamp=-10.0,
        loss_max_clamp=0.5,
    )

    unclamped_student = torch.log_softmax(student, dim=-1).gather(-1, teacher_ids)
    expected_terms = teacher_log_probs.clamp_min(-10).exp() * (
        teacher_log_probs.clamp_min(-10) - unclamped_student.clamp_min(-10)
    )
    expected = expected_terms.sum(dim=-1).clamp_min(0).clamp(max=0.5)
    torch.testing.assert_close(output.loss, expected)
    assert output.overlap_count.tolist() == [1]
    torch.testing.assert_close(output.overlap_token_advantage, -expected_terms[:, 1])
    # Mass is measured before stability clamps in official verl v0.8.0.
    torch.testing.assert_close(output.teacher_mass, teacher_log_probs.exp().sum(dim=-1))
    torch.testing.assert_close(output.student_mass, unclamped_student.exp().sum(dim=-1))


def test_topk_objective_remains_distinct_from_tail_bucket_kl() -> None:
    student = torch.tensor([[0.1, 0.3, -0.4, 1.2]])
    teacher_ids = torch.tensor([[3, 1]])
    teacher_log_probs = torch.log(torch.tensor([[0.50, 0.20]]))

    output = verl_forward_kl_topk(student, teacher_log_probs, teacher_ids)

    assert output.loss.item() >= 0.0
    assert output.teacher_mass.item() == torch.tensor(0.7).item()
    assert output.teacher_mass.item() < 1.0
