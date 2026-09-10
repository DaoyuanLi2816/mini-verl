"""The upstream discriminative reward path preserves chats and raw logits."""

from types import SimpleNamespace

import pytest


@pytest.mark.torch
@pytest.mark.parametrize("too_long", [False, True])
def test_upstream_chat_reward_is_raw_last_logit(too_long):
    import torch

    from miniverl.config import RewardModelConfig
    from miniverl.rewards import HFSequenceClassifierRewardProvider, RewardRequest

    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]

    class Tokenizer:
        def apply_chat_template(self, chat, **kwargs):
            assert chat == [*messages, {"role": "assistant", "content": "answer"}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": False}
            return "<bos>complete-chat<eos>"

        def __call__(self, chats, **kwargs):
            assert chats == ["<bos>complete-chat<eos>"]
            assert kwargs["add_special_tokens"] is False
            assert kwargs["truncation"] is False
            return {"input_ids": torch.zeros((1, 129 if too_long else 10), dtype=torch.long)}

    class Model:
        def __call__(self, **_kwargs):
            return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))

    config = RewardModelConfig(
        model_id="org/classifier", revision="a" * 40, input_format="verl_chat", max_length=128
    )
    provider = HFSequenceClassifierRewardProvider(
        config, tokenizer=Tokenizer(), model=Model(), device="cpu"
    )
    request = RewardRequest.create(
        trajectory_id="t",
        prompt_group_id="g",
        sample_index=0,
        samples_per_prompt=1,
        row_digest="b" * 64,
        prompt_text="actor-rendered",
        response_text="answer",
        reward_model={},
        ground_truth=None,
        data_source="unit",
        raw_prompt=messages,
    )
    if too_long:
        with pytest.raises(ValueError, match="no silent truncation"):
            provider.score(request)
    else:
        assert provider.score(request).raw_reward == 2.0


def test_bound_python_reward_uses_upstream_abi(tmp_path):
    import hashlib

    from miniverl.rewards import RewardRequest
    from miniverl.rewards.binding import BoundVerlReward

    path = tmp_path / "reward.py"
    path.write_text(
        "def compute_score(data_source, solution_str, ground_truth, extra_info):\n    assert data_source == 'unit' and extra_info == {'id': 7}\n    return {'score': float(solution_str == ground_truth)}\n"
    )
    provider = BoundVerlReward(
        f"{path}:compute_score", approved_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    request = RewardRequest.create(
        trajectory_id="t",
        prompt_group_id="g",
        sample_index=0,
        samples_per_prompt=1,
        row_digest="a" * 64,
        prompt_text="prompt",
        response_text="yes",
        reward_model={},
        ground_truth="yes",
        data_source="unit",
        extra_info={"id": 7},
    )
    assert provider.score(request).raw_reward == 1.0
    assert provider.identity.deterministic is False
