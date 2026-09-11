"""Bounded systems launch arguments in upstream syntax, not native recipes."""

from __future__ import annotations


def qualification_overrides(algorithm: str) -> list[str]:
    if algorithm not in {"ppo", "grpo"}:
        raise ValueError("choose ppo or grpo")
    arguments = [
        "data.train_files=[data/rl-prompts.parquet]",
        "data.val_files=[data/rl-prompts.parquet]",
        "data.train_batch_size=2",
        "data.max_prompt_length=128",
        "data.max_response_length=24",
        "data.seed=23",
        "data.shuffle=true",
        "actor_rollout_ref.model.path=Qwen/Qwen3-0.6B",
        "actor_rollout_ref.model.lora_rank=16",
        "actor_rollout_ref.model.lora_alpha=32",
        "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]",
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.actor.ppo_epochs=2",
        "actor_rollout_ref.actor.shuffle=true",
        "actor_rollout_ref.actor.data_loader_seed=42",
        "actor_rollout_ref.actor.optim.lr=1e-5",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.entropy_coeff=0.001",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.top_k=0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=2",
        "trainer.total_training_steps=2",
        "trainer.total_epochs=1",
        "trainer.save_freq=1",
        "trainer.test_freq=1",
        "trainer.logger=[console]",
        "trainer.v1.sampler.sync_refill_failed_groups=true",
        "reward.custom_reward_function.path=local-reward.py",
    ]
    if algorithm == "ppo":
        arguments += [
            "algorithm.adv_estimator=gae",
            "algorithm.gamma=0.99",
            "algorithm.lam=0.95",
            "critic.enable=true",
            "critic.model.path=Qwen/Qwen3-0.6B",
            "critic.model.lora_rank=16",
            "critic.model.lora_alpha=32",
            "critic.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]",
            "critic.ppo_mini_batch_size=2",
            "critic.ppo_epochs=2",
            "critic.shuffle=true",
            "critic.data_loader_seed=113",
            "critic.optim.lr=2e-5",
            "critic.optim.lr_warmup_steps=0",
        ]
    else:
        arguments += [
            "algorithm.adv_estimator=grpo",
            "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
            "algorithm.filter_groups.enable=true",
            "algorithm.filter_groups.metric=score",
        ]
    return arguments
