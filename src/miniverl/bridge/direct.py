"""Resolved verl v0.9 direct execution, with explicit semantic and placement states.

The packaged schema bounds the input surface, but is NOT an acceptance list.
Every leaf also needs a consumed-field rule, an inactive-branch proof, or a
physical-lowering rule. Unknown semantics fail closed. No config-driven imports.
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT, UPSTREAM_VERL_TAG
from miniverl.bridge.rl_v09 import _RULES_V3, _get
from miniverl.errors import ConfigError

DIRECT_PROFILE = "verl-rl-v0.9-single-gpu-v4"


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueLoader, node: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or "." in key:
            raise ValueError("resolved config keys must be strings without dots")
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


def _put(value: dict[str, Any], field: str, item: Any) -> None:
    parts = field.split(".")
    for part in parts[:-1]:
        value = value.setdefault(part, {})
    value[parts[-1]] = item


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _schema() -> dict[str, Any]:
    return json.loads(
        files("miniverl.resources").joinpath("verl_v09_direct_schema.json").read_text()
    )


def direct_rules_digest() -> str:
    """Bind the input schema and implemented v4 field policy, not older profiles."""
    return _digest(
        {
            "schema": _schema(),
            "consumed": sorted(_CONSUMED),
            "physical": sorted(_PHYSICAL_SUFFIXES),
            "invariants": _INVARIANTS,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )


# Experiment fields that the v4 IR builder consumes (not merely acknowledges).
_CONSUMED = {field for field in _RULES_V3 if not field.startswith("miniverl.")} | {
    "miniverl.actor.revision",
    "miniverl.actor.tokenizer_revision",
    "miniverl.critic.revision",
    "miniverl.reward.provider",
    "miniverl.reward.revision",
    "miniverl.reward.tokenizer_revision",
    "miniverl.rollout.backend",
    "actor_rollout_ref.actor.loss_scale_factor",
    "actor_rollout_ref.actor.grad_clip",
    "actor_rollout_ref.actor.optim.betas",
    "actor_rollout_ref.actor.optim.lr_scheduler_type",
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio",
    "actor_rollout_ref.model.trust_remote_code",
    "actor_rollout_ref.model.tokenizer_path",
    "actor_rollout_ref.rollout.do_sample",
    "actor_rollout_ref.rollout.val_kwargs.do_sample",
    "actor_rollout_ref.rollout.val_kwargs.temperature",
    "actor_rollout_ref.rollout.val_kwargs.top_p",
    "actor_rollout_ref.rollout.val_kwargs.top_k",
    "actor_rollout_ref.rollout.val_kwargs.n",
    "critic.grad_clip",
    "critic.optim.betas",
    "critic.optim.lr_scheduler_type",
    "critic.optim.lr_warmup_steps_ratio",
    "data.filter_overlong_prompts",
    "data.validation_shuffle",
    "trainer.val_before_train",
    "trainer.default_local_dir",
    "reward.reward_model.model_path",
    "reward.reward_model.enable",
    "reward.custom_reward_function.path",
    "reward.custom_reward_function.name",
    "reward.reward_manager.name",
}

# Physical controls never choose a different objective, sample count, or mask.
_PHYSICAL_SUFFIXES = {
    "strategy",
    "param_offload",
    "optimizer_offload",
    "forward_prefetch",
    "fsdp_size",
    "reshard_after_forward",
    "use_orig_params",
    "use_torch_compile",
    "ulysses_sequence_parallel_size",
    "use_remove_padding",
    "use_dynamic_bsz",
    "log_prob_use_dynamic_bsz",
    "use_fused_kernels",
    "entropy_checkpointing",
    "entropy_from_logits_chunk_size",
    "entropy_from_logits_with_chunking",
    "pad_to_length",
    "pad_to_length_bucket",
    "ppo_micro_batch_size",
    "ppo_micro_batch_size_per_gpu",
    "ppo_max_token_len_per_gpu",
    "log_prob_micro_batch_size",
    "log_prob_micro_batch_size_per_gpu",
    "log_prob_max_token_len_per_gpu",
    "forward_micro_batch_size",
    "forward_micro_batch_size_per_gpu",
    "forward_max_token_len_per_gpu",
    "enable_activation_offload",
    "enable_gradient_checkpointing",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "data_parallel_size",
    "expert_parallel_size",
    "gpu_memory_utilization",
    "max_num_batched_tokens",
    "max_num_seqs",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "enforce_eager",
    "free_cache_engine",
    "cudagraph_capture_sizes",
    "disable_log_stats",
    "skip_tokenizer_init",
    "layered_summon",
    "multi_stage_wake_up",
    "n_gpus_per_node",
    "nnodes",
    "num_workers",
    "use_shm",
    "dataloader_num_workers",
    "filter_overlong_prompts_workers",
    "balance_batch",
    "hybrid_engine",
    "load_format",
    "scheduling_policy",
    "max_model_len",
    "log_prob_calculate",
}

# These invariants are checked by value. A different active value is NOT ignored.
_INVARIANTS: dict[str, Any] = {
    "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
    "actor_rollout_ref.actor.shuffle": False,
    "critic.shuffle": False,
    "critic.loss_agg_mode": "token-mean",
    "critic.loss_scale_factor": None,
    "actor_rollout_ref.rollout.ignore_eos": False,
    "actor_rollout_ref.rollout.over_sample_rate": 0,
    "actor_rollout_ref.rollout.quantization": None,
    "actor_rollout_ref.rollout.quantization_config_file": None,
    "actor_rollout_ref.rollout.engine_kwargs": {},
    "actor_rollout_ref.rollout.mode": "async",
    "algorithm.use_pf_ppo": False,
    "algorithm.kl_ctrl.type": "fixed",
    "algorithm.rollout_correction.bypass_mode": False,
    "algorithm.rollout_correction.rollout_is": None,
    "algorithm.rollout_correction.rollout_rs": None,
    "algorithm.rollout_correction.loss_type": "ppo_clip",
    "algorithm.rollout_correction.rollout_is_batch_normalize": False,
    "data.custom_cls.path": None,
    "data.custom_cls.name": None,
    "data.tokenizer": None,
    "data.apply_chat_template_kwargs": {},
    "data.mm_processor_kwargs": {},
    "data.function_tool_path": None,
    "data.tool_config_path": None,
    "data.train_max_samples": -1,
    "data.val_max_samples": -1,
    "data.gen_batch_size": None,
    "data.reward_fn_key": "data_source",
    "data.trust_remote_code": False,
    "trainer.critic_warmup": 0,
    "trainer.val_only": False,
    "trainer.resume_from_path": None,
    "trainer.v1.trainer_mode": "sync",
    "trainer.v1.sampler.sync_refill_failed_groups": False,
    "trainer.v1.sampler.custom_sampler.path": None,
    "trainer.v1.sampler.custom_sampler.name": None,
    "trainer.v1.sampler.sampler_kwargs": {},
    "model_engine": "dp",
    "reward.reward_manager.source": "register",
    "reward.reward_manager.module.path": None,
    "actor_rollout_ref.actor.freeze_vision_tower": False,
    "actor_rollout_ref.actor.use_prefix_grouper": False,
    "actor_rollout_ref.model.custom_chat_template": None,
    "actor_rollout_ref.model.exclude_modules": None,
    "actor_rollout_ref.model.external_lib": None,
    "actor_rollout_ref.model.hf_config_path": None,
    "actor_rollout_ref.model.lora_adapter_path": None,
    "actor_rollout_ref.model.override_config": {},
    "actor_rollout_ref.model.use_liger": False,
    "actor_rollout_ref.rollout.agent.agent_loop_config_path": None,
    "actor_rollout_ref.rollout.agent.custom_async_server.name": None,
    "actor_rollout_ref.rollout.agent.custom_async_server.path": None,
    "actor_rollout_ref.rollout.enable_rollout_routing_replay": False,
    "actor_rollout_ref.rollout.logprobs_mode": "processed_logprobs",
    "actor_rollout_ref.rollout.engine_kwargs.vllm": {},
    "actor_rollout_ref.rollout.engine_kwargs.sglang": {},
    "actor_rollout_ref.rollout.engine_kwargs.trtllm": {},
    "trainer.device": "cuda",
    "trainer.v1.sync": {},
}
for _key in (
    "custom_chat_template",
    "exclude_modules",
    "external_lib",
    "hf_config_path",
    "lora_adapter_path",
    "override_config",
    "use_liger",
):
    _INVARIANTS[f"critic.model.{_key}"] = _INVARIANTS[f"actor_rollout_ref.model.{_key}"]
_CONSUMED.update(
    f"critic.model.{key}"
    for key in ("lora_alpha", "lora_rank", "target_modules", "tokenizer_path", "trust_remote_code")
)


def classify(source: dict[str, Any]) -> list[dict[str, Any]]:
    flat = flatten(source)
    known = _schema()["fields"]
    estimator = _get(source, "algorithm.adv_estimator", None)
    ref_active = _get(source, "algorithm.use_kl_in_reward", False) or _get(
        source, "actor_rollout_ref.actor.use_kl_loss", False
    )
    inactive: dict[str, str] = {}
    # Disabled feature subtrees are inactive even if they contain upstream ???
    # sentinels. Never instantiate their _target_ or resolve their credentials.
    for field, value in flat.items():
        if field.endswith((".enable", ".enabled")) and value is False:
            inactive[field.rsplit(".", 1)[0] + "."] = field
    if estimator != "gae":
        inactive["critic."] = "algorithm.adv_estimator != gae"
    if not ref_active:
        inactive["actor_rollout_ref.ref."] = "both reference KL coefficients disabled"
    if _get(source, "actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla") == "vanilla":
        inactive["actor_rollout_ref.actor.policy_loss."] = "vanilla objective"
    if not _get(source, "algorithm.use_pf_ppo", False):
        inactive["algorithm.pf_ppo."] = "algorithm.use_pf_ppo=false"
    if _get(source, "algorithm.kl_ctrl.type", "fixed") == "fixed":
        inactive["algorithm.kl_ctrl.horizon"] = "fixed KL controller"
        inactive["algorithm.kl_ctrl.target_kl"] = "fixed KL controller"
    # Megatron's nested LoRA configuration is distinct from FSDP PEFT fields.
    for role in ("actor_rollout_ref", "critic"):
        if _get(
            source,
            f"{role}.actor.strategy" if role == "actor_rollout_ref" else "critic.strategy",
            "fsdp",
        ) in {"fsdp", "fsdp2"}:
            inactive[f"{role}.model.lora."] = "nested LoRA is a Megatron-only branch"
    rows = []
    for field, value in sorted(flat.items()):
        kind, reason = "unsupported", "unimplemented active experiment field"
        if field not in known and field not in _CONSUMED:
            reason = "unknown field outside pinned upstream schema"
        elif field.startswith("critic.") and estimator != "gae":
            kind, reason = "informational_only", "no critic role for this estimator"
        elif field in _INVARIANTS:
            if type(value) is type(_INVARIANTS[field]) and value == _INVARIANTS[field]:
                kind, reason = "semantically_conformant", "validated supported invariant"
            else:
                reason = f"requires {_INVARIANTS[field]!r}; requested {value!r} changes semantics"
        elif field in inactive.values():
            kind, reason = "semantically_conformant", "feature disabled; subtree is inactive"
        elif gate := next(
            (
                gate
                for prefix, gate in inactive.items()
                if field.startswith(prefix) and field != gate
            ),
            None,
        ):
            kind, reason = "informational_only", f"inactive branch: {gate}"
        elif field in _CONSUMED:
            kind, reason = "exact", "consumed by the local experiment IR"
        elif field.endswith("._target_"):
            if value in known[field]:
                kind, reason = (
                    "locally_lowered",
                    "pinned typed config marker; no Python object imported",
                )
        elif field.rsplit(".", 1)[-1] in _PHYSICAL_SUFFIXES:
            kind, reason = (
                "locally_lowered",
                "physical placement/packing; logical samples and updates retained",
            )
        elif (
            ".router_replay." in field
            and _get(source, field.rsplit(".", 1)[0] + ".mode", None) == "disabled"
        ):
            kind, reason = "informational_only", "router replay disabled"
        elif ".fsdp_config." in field or ".fsdp." in field:
            leaf = field.rsplit(".", 1)[-1]
            allowed = {
                "dtype": "bfloat16",
                "model_dtype": "fp32",
                "forward_only": field.startswith("actor_rollout_ref.ref."),
                "full_determinism": False,
                "offload_policy": False,
                "seed": 42,
                "min_num_params": 0,
            }
            if leaf in allowed and value == allowed[leaf]:
                kind, reason = (
                    "locally_lowered",
                    "validated distributed engine default; local role precision/placement recorded",
                )
        elif field in {
            "actor_rollout_ref.actor.rollout_n",
            "actor_rollout_ref.ref.rollout_n",
            "critic.rollout_n",
        }:
            if value == _get(source, "actor_rollout_ref.rollout.n", 1):
                kind, reason = "derived", "same rollout.n alias"
        elif field in {
            "actor_rollout_ref.rollout.prompt_length",
            "actor_rollout_ref.rollout.response_length",
        }:
            target = (
                "data.max_prompt_length"
                if field.endswith("prompt_length")
                else "data.max_response_length"
            )
            if value == _get(source, target, None):
                kind, reason = "derived", f"validated alias of {target}"
        elif field in {"actor_rollout_ref.actor.tau_neg", "actor_rollout_ref.actor.tau_pos"}:
            kind, reason = (
                "informational_only",
                "SAPO temperatures are inactive for vanilla policy loss",
            )
        elif field in {
            "algorithm.rollout_correction.rollout_is_threshold",
            "algorithm.rollout_correction.rollout_rs_threshold",
        }:
            if (
                _get(source, "algorithm.rollout_correction.rollout_is", None) is None
                and _get(source, "algorithm.rollout_correction.rollout_rs", None) is None
            ):
                kind, reason = (
                    "informational_only",
                    "importance correction and rejection sampling disabled",
                )
        elif field.startswith("actor_rollout_ref.rollout.checkpoint_engine."):
            kind, reason = (
                "locally_lowered",
                "same-process policy handoff replaces distributed weight transport",
            )
        elif field in {
            "actor_rollout_ref.nccl_timeout",
            "actor_rollout_ref.rollout.moe_load_balance_metrics_interval",
            "actor_rollout_ref.rollout.calculate_log_probs",
            "actor_rollout_ref.rollout.full_determinism",
            "actor_rollout_ref.rollout.dtype",
        }:
            kind, reason = (
                "locally_lowered",
                "local dtype/determinism and actor log-probability path recorded",
            )
        elif (
            field.startswith("reward.sandbox_fusion.")
            and _get(source, "reward.sandbox_fusion.url", None) is None
        ):
            kind, reason = "informational_only", "no external sandbox configured"
        elif field.startswith("reward.reward_model.rollout."):
            # The common Skywork path uses vLLM's /classify endpoint with
            # use_activation=false, NOT autoregressive reward generation.
            if value == known[field][0] or field.endswith(
                (".prompt_length", ".response_length", ".name")
            ):
                kind, reason = (
                    "locally_lowered",
                    "discriminative RM endpoint lowered to raw HF classifier logits",
                )
        elif field == "reward.reward_model.enable_resource_pool":
            kind, reason = "locally_lowered", "reward inference scheduled locally"
        elif (
            any(
                field.startswith(prefix)
                for prefix in (
                    "ray_kwargs.",
                    "global_profiler.",
                    "actor_rollout_ref.rollout.trace.",
                )
            )
            or ".profiler." in field
            or ".checkpoint." in field
        ):
            kind, reason = (
                "locally_lowered",
                "local journal/checkpoint/logging replaces distributed plumbing",
            )
        elif (
            field.startswith(("reward_model.", "custom_reward_function.", "sandbox_fusion."))
            and value is None
        ):
            kind, reason = "informational_only", "unset deprecated upstream alias"
        elif field in {
            "actor_rollout_ref.model.fused_kernel_options.impl_backend",
            "critic.model.fused_kernel_options.impl_backend",
            "actor_rollout_ref.actor.calculate_entropy",
            "actor_rollout_ref.actor.calculate_sum_pi_squared",
            "actor_rollout_ref.actor.data_loader_seed",
            "critic.data_loader_seed",
            "actor_rollout_ref.rollout.seed",
            "data.audio_key",
            "data.image_key",
            "data.video_key",
            "data.image_patch_size",
            "data.return_full_prompt",
            "data.return_multi_modal_inputs",
            "data.return_raw_chat",
            "data.return_raw_input_ids",
            "data.val_batch_size",
            "trainer.default_hdfs_dir",
            "trainer.del_local_ckpt_after_load",
            "trainer.esi_redundant_time",
            "trainer.log_val_generations",
            "trainer.max_actor_ckpt_to_keep",
            "trainer.max_critic_ckpt_to_keep",
            "trainer.ray_wait_register_center_timeout",
            "trainer.rollout_data_dir",
            "trainer.validation_data_dir",
            "trainer.resume_mode",
            "trainer.use_v1",
            "reward.reward_manager.module.name",
            "actor_rollout_ref.rollout.agent.default_agent_loop",
            "actor_rollout_ref.rollout.agent.num_workers",
        }:
            kind, reason = (
                "locally_lowered",
                "local serialization/seed/monitoring policy recorded independently",
            )
        elif (
            field.startswith(
                (
                    "trainer.v1.colocate_async.",
                    "trainer.v1.separate_async.",
                    "trainer.v1.sampler.max_off_policy",
                )
            )
            and _get(source, "trainer.v1.trainer_mode", "sync") == "sync"
        ):
            kind, reason = (
                "informational_only",
                "synchronous local rollout has no asynchronous stale queue",
            )
        elif ".optim." in field and field.rsplit(".", 1)[-1] in {
            "optimizer",
            "optimizer_impl",
            "override_optimizer_config",
            "total_training_steps",
            "warmup_style",
            "min_lr_ratio",
            "num_cycles",
            "zero_indexed_step",
            "clip_grad",
        }:
            allowed = {
                "optimizer": "AdamW",
                "optimizer_impl": "torch.optim",
                "override_optimizer_config": None,
                "total_training_steps": -1,
                "warmup_style": None,
                "min_lr_ratio": 0.0,
                "num_cycles": 0.5,
                "zero_indexed_step": True,
                "clip_grad": 1.0,
            }
            if value == allowed[field.rsplit(".", 1)[-1]]:
                kind, reason = (
                    "semantically_conformant",
                    "validated AdamW/constant-schedule default",
                )
        rows.append(
            {"field": field, "source_value": value, "classification": kind, "reason": reason}
        )
    return rows


def build_native(source: dict[str, Any], *, cycles: int) -> dict[str, Any]:
    """Reuse the historical IR skeleton, then apply only explicit v4 semantics."""
    from miniverl.bridge.rl_v09 import VERL_RL_V09_PRODUCT_PROFILE, _number, _recipe
    from miniverl.config import RunConfig

    working = copy.deepcopy(source)

    def get(field: str, default: Any = None) -> Any:
        return _get(source, field, default)

    if get("reward.reward_manager.name", "naive") != "naive":
        raise ConfigError(
            "direct v4 requires the naive reward manager; other managers may alter selection or reward semantics"
        )
    if get("actor_rollout_ref.rollout.do_sample", True) is not True:
        raise ConfigError("direct RL requires stochastic do_sample=true")
    if get(
        "actor_rollout_ref.rollout.logprobs_mode", "processed_logprobs"
    ) == "processed_logprobs" and (
        float(get("actor_rollout_ref.rollout.top_p", 1.0)) != 1.0
        or get("actor_rollout_ref.rollout.top_k", -1) not in {-1, 0}
    ):
        raise ConfigError(
            "processed rollout probabilities with top-p/top-k truncation need a matching importance-correction implementation"
        )
    if get("actor_rollout_ref.actor.optim.lr_scheduler_type", "constant") not in {
        "constant",
        "cosine",
    }:
        raise ConfigError("unsupported actor learning-rate schedule")
    if get("algorithm.adv_estimator") == "gae" and get(
        "critic.optim.lr_scheduler_type", "constant"
    ) not in {"constant", "cosine"}:
        raise ConfigError("unsupported critic learning-rate schedule")
    if get("trainer.total_epochs", 1) < 1:
        raise ConfigError("trainer.total_epochs must be positive")
    if get("critic.enable") is False and get("algorithm.adv_estimator") == "gae":
        raise ConfigError("GAE requires a critic; critic.enable=false contradicts the estimator")
    if get("critic.enable") is True and get("algorithm.adv_estimator") != "gae":
        raise ConfigError("an explicitly enabled critic for a non-GAE estimator is unsupported")
    if get("algorithm.adv_estimator") == "gae":
        from miniverl.bridge.rl_v09 import _integer

        critic_minibatch = _integer(
            get(
                "critic.ppo_mini_batch_size",
                get("actor_rollout_ref.actor.ppo_mini_batch_size", get("data.train_batch_size")),
            ),
            "critic.ppo_mini_batch_size",
        )
        if critic_minibatch < 1 or get("data.train_batch_size") % critic_minibatch:
            raise ConfigError(
                "data.train_batch_size must be divisible by the positive critic prompt minibatch"
            )
    kl_reward = get("algorithm.use_kl_in_reward", False)
    kl_actor = get("actor_rollout_ref.actor.use_kl_loss", False)
    rank = get("actor_rollout_ref.model.lora_rank", 0)
    provider = get("miniverl.reward.provider", "python_api")
    # The v3 builder is a private skeleton, never the selected profile. Remove
    # v3-only adapter requirements before constructing the explicit v4 roles.
    replacements = {
        "algorithm.use_kl_in_reward": False,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "miniverl.reward.provider": "exact_answer",
        "miniverl.actor.lora_enabled": rank > 0,
        "actor_rollout_ref.model.lora_rank": max(rank, 1),
        "actor_rollout_ref.model.target_modules": (
            [get("actor_rollout_ref.model.target_modules")]
            if isinstance(get("actor_rollout_ref.model.target_modules"), str)
            else get("actor_rollout_ref.model.target_modules", ["q_proj", "v_proj"])
        ),
        "miniverl.critic.lora_enabled": rank > 0,
        "critic.enable": get("algorithm.adv_estimator") == "gae",
        "critic.model.path": get("actor_rollout_ref.model.path"),
        "critic.ppo_mini_batch_size": get(
            "actor_rollout_ref.actor.ppo_mini_batch_size", get("data.train_batch_size")
        ),
        "trainer.total_training_steps": cycles,
        "data.seed": get("data.seed") if get("data.seed") is not None else 42,
        "actor_rollout_ref.rollout.top_k": max(get("actor_rollout_ref.rollout.top_k", -1), 0),
    }
    for role in ("actor_rollout_ref.actor", "critic"):
        replacements[f"{role}.optim.lr_warmup_steps"] = max(
            get(f"{role}.optim.lr_warmup_steps", 0), 0
        )
    for key, value in replacements.items():
        _put(working, key, value)
    recipe, lowering = _recipe(working, profile=VERL_RL_V09_PRODUCT_PROFILE)
    recipe["run"]["profile_identity"] = {
        "profile_name": DIRECT_PROFILE,
        "field_rules_sha256": direct_rules_digest(),
        "upstream_tag": UPSTREAM_VERL_TAG,
        "upstream_commit": UPSTREAM_VERL_COMMIT,
        "local_lowering": lowering,
    }
    recipe["run"]["tags"] = [get("trainer.project_name", "verl"), DIRECT_PROFILE]
    recipe["run"]["output_dir"] = get("trainer.default_local_dir", "runs")
    student = recipe["models"]["student"]
    student["trust_remote_code"] = get("actor_rollout_ref.model.trust_remote_code", False)
    student["tokenizer_id"] = get("actor_rollout_ref.model.tokenizer_path")
    if get("algorithm.adv_estimator") == "gae":
        critic_model = copy.deepcopy(student)
        critic_model["model_id"] = get("critic.model.path", student["model_id"])
        critic_model["revision"] = get("miniverl.critic.revision", student["revision"])
        critic_model["tokenizer_revision"] = critic_model["revision"]
        critic_model["tokenizer_id"] = get("critic.model.tokenizer_path")
        critic_model["gradient_checkpointing"] = get(
            "critic.model.enable_gradient_checkpointing", False
        )
        critic_model["trust_remote_code"] = get("critic.model.trust_remote_code", False)
        critic_rank = get("critic.model.lora_rank", 0)
        critic_model["lora"] = {
            "enabled": critic_rank > 0,
            "r": max(critic_rank, 1),
            "alpha": get("critic.model.lora_alpha", 16),
            "target_modules": [get("critic.model.target_modules", "all-linear")]
            if isinstance(get("critic.model.target_modules", "all-linear"), str)
            else get("critic.model.target_modules"),
        }
        recipe["critic"]["model"] = critic_model
        recipe["critic"]["gradient_accumulation_steps"] = get(
            "critic.ppo_mini_batch_size",
            get("actor_rollout_ref.actor.ppo_mini_batch_size", get("data.train_batch_size")),
        ) * get("actor_rollout_ref.rollout.n", 1)
    recipe["loss"]["aggregation"] = get("actor_rollout_ref.actor.loss_agg_mode", "token-mean")
    recipe["loss"]["loss_scale_factor"] = get("actor_rollout_ref.actor.loss_scale_factor")
    if kl_reward or kl_actor:
        recipe["models"]["reference"] = {
            "model_id": student["model_id"],
            "revision": student["revision"],
            "tokenizer_id": student["tokenizer_id"],
            "tokenizer_revision": student["tokenizer_revision"],
            "frozen_base": True,
            "trust_remote_code": student["trust_remote_code"],
        }
        recipe["algorithm"]["kl_coef"] = (
            _number(get("algorithm.kl_ctrl.kl_coef", 0.001), "kl_coef") if kl_reward else 0.0
        )
        recipe["algorithm"]["actor_kl_coef"] = (
            _number(get("actor_rollout_ref.actor.kl_loss_coef", 0.001), "actor_kl_coef")
            if kl_actor
            else 0.0
        )
    for role, target in (("actor_rollout_ref.actor", "train"), ("critic", "critic")):
        if target not in recipe:
            continue
        betas = get(f"{role}.optim.betas", [0.9, 0.999])
        recipe[target].update(
            {
                "adam_beta1": betas[0],
                "adam_beta2": betas[1],
                "max_grad_norm": get(f"{role}.grad_clip", 1.0),
                "lr_schedule": get(f"{role}.optim.lr_scheduler_type", "constant"),
                "lr_step_unit": "rollout_iteration",
                "zero_indexed_warmup": True,
            }
        )
        warmup = get(f"{role}.optim.lr_warmup_steps", -1)
        recipe[target]["warmup_steps"] = (
            int(cycles * float(get(f"{role}.optim.lr_warmup_steps_ratio", 0.0)))
            if warmup < 0
            else warmup
        )
    recipe["source"]["filter_overlong_prompts"] = get("data.filter_overlong_prompts", False)
    recipe["source"]["drop_last"] = True
    recipe["source"]["unsupported_input_columns"] = [
        get("data.image_key", "images"),
        get("data.video_key", "videos"),
        get("data.audio_key", "audios"),
    ]
    recipe["source"]["validation_shuffle"] = get("data.validation_shuffle", False)
    recipe["source"]["use_task_rewards"] = provider in {"exact_answer", "target_length"}
    recipe["source"]["reward_metadata_mode"] = (
        "builtin" if recipe["source"]["use_task_rewards"] else "upstream"
    )
    recipe["reward"]["provider"] = provider
    if get("reward.reward_model.enable", False):
        recipe["reward"] = {
            "enabled": True,
            "provider": "hf_sequence_classifier",
            "model": {
                "model_id": get("reward.reward_model.model_path"),
                "revision": get("miniverl.reward.revision"),
                "tokenizer_revision": get(
                    "miniverl.reward.tokenizer_revision", get("miniverl.reward.revision")
                ),
                "positive_class_index": 0,
                "input_format": "verl_chat",
                "max_length": get("reward.reward_model.rollout.prompt_length", 8192),
            },
        }
    val = get("actor_rollout_ref.rollout.val_kwargs", {})
    enabled = bool(get("data.val_files")) and (
        get("trainer.val_before_train", True) or get("trainer.test_freq", -1) > 0
    )
    recipe["eval"] = {
        "enabled": enabled,
        "baseline_enabled": get("trainer.val_before_train", True),
        "temperature": val.get("temperature", 0.0) if val.get("do_sample", False) else 0.0,
        "top_p": val.get("top_p", 1.0),
        "top_k": max(val.get("top_k", -1), 0),
        "samples_per_prompt": val.get("n", 1),
    }
    return RunConfig.model_validate(recipe).model_dump(mode="json")


_BINDINGS = {
    "data.train_files": "data.train_files",
    "data.val_files": "data.val_files",
    "actor.model": "actor_rollout_ref.model.path",
    "actor.revision": "miniverl.actor.revision",
    "actor.tokenizer_revision": "miniverl.actor.tokenizer_revision",
    "critic.model": "critic.model.path",
    "critic.revision": "miniverl.critic.revision",
    "reward.model": "reward.reward_model.model_path",
    "reward.revision": "miniverl.reward.revision",
    "reward.tokenizer_revision": "miniverl.reward.tokenizer_revision",
    "reward.provider": "miniverl.reward.provider",
    "reward.function": None,
    "output": "trainer.default_local_dir",
    "rollout.backend": "miniverl.rollout.backend",
}


def apply_bindings(
    source: dict[str, Any], bindings: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from miniverl.utils.privacy import _CREDENTIAL_VALUE

    effective = copy.deepcopy(source)
    ledger = []
    used = set()
    for assignment in bindings:
        key, separator, text = assignment.partition("=")
        if not separator or key not in _BINDINGS or key in used:
            raise ConfigError(
                f"unknown or repeated binding {key!r}; supported: {', '.join(_BINDINGS)}"
            )
        used.add(key)
        value = yaml.safe_load(text)
        if _CREDENTIAL_VALUE.search(json.dumps(value, default=str)):
            raise ConfigError("credentials belong in the process environment, not runtime bindings")
        if not isinstance(value, (str, list)) or not value:
            raise ConfigError(f"binding {key} requires a non-empty path, identity, or path list")
        if isinstance(value, list) and (
            not key.startswith("data.")
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ConfigError("only data bindings accept lists of non-empty paths")
        if key == "reward.provider" and value not in {"exact_answer", "target_length"}:
            raise ConfigError(
                "reward.provider binding supports exact_answer or target_length; use reward.function for trusted Python"
            )
        if key == "rollout.backend" and value not in {"hf_cached", "hf_reference"}:
            raise ConfigError("direct v4 local backend must be hf_cached or hf_reference")
        target = _BINDINGS[key]
        old = _get(source, target, None) if target else None
        if target:
            _put(effective, target, value)
        ledger.append({"binding": key, "target": target, "original": old, "bound": value})
    if len(used & {"reward.function", "reward.provider"}) > 1 or (
        used & {"reward.function", "reward.provider"}
        and _get(source, "reward.reward_model.enable", False)
    ):
        raise ConfigError(
            "choose one reward definition: Python function, builtin provider, or trained model"
        )
    # A same-base critic remains same-base after an explicit actor snapshot bind.
    if (
        "actor.model" in used
        and "critic.model" not in used
        and _get(source, "critic.model.path", None)
        == _get(source, "actor_rollout_ref.model.path", None)
    ):
        _put(effective, "critic.model.path", _get(effective, "actor_rollout_ref.model.path"))
        ledger.append(
            {
                "binding": "critic.model",
                "target": "critic.model.path",
                "original": _get(source, "critic.model.path"),
                "bound": _get(effective, "critic.model.path"),
                "derived": "same-base role identity",
            }
        )
    return effective, ledger


def compile_direct(source: str | Path, *, bindings: list[str] | None = None) -> dict[str, Any]:
    path = Path(source)
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ConfigError("resolved config exceeds the 4 MiB input bound")
        content = path.read_bytes()
        if any(
            isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken))
            for token in yaml.scan(content)
        ):
            raise ConfigError("resolve YAML anchors and aliases before direct compilation")
        payload = yaml.load(content, Loader=_UniqueLoader)
    except (OSError, ValueError, yaml.YAMLError, RecursionError) as exc:
        raise ConfigError(f"cannot read resolved verl YAML: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("algorithm"), dict):
        raise ConfigError(
            "direct RL input requires a resolved verl mapping with algorithm.adv_estimator"
        )
    from miniverl.utils.privacy import _CREDENTIAL_VALUE, _is_sensitive_key

    for field, value in flatten(payload).items():
        if "${" in json.dumps(value, default=str):
            raise ConfigError(
                f"unresolved interpolation at {field}; resolve Hydra config before running"
            )
        if (
            _is_sensitive_key(field.rsplit(".", 1)[-1])
            and field not in _schema()["fields"]
            and value is not None
        ) or _CREDENTIAL_VALUE.search(json.dumps(value, default=str)):
            raise ConfigError(
                "config contains credential material; use the process environment, not YAML or --bind"
            )
    # JSON finite check also rejects YAML objects/dates before report publication.
    try:
        _digest(payload)
    except (ValueError, TypeError) as exc:
        raise ConfigError("config must contain finite JSON-compatible values") from exc
    rows = classify(payload)
    rejections = [row for row in rows if row["classification"] == "unsupported"]
    effective, ledger = apply_bindings(payload, bindings or [])
    # Validate mathematical and native invariants even if deployment inputs are
    # absent. Placeholder iteration/revision values never become runnable output.
    probe = copy.deepcopy(effective)
    if (
        _get(probe, "reward.reward_model.enable", False)
        and _get(probe, "miniverl.reward.revision", None) is None
    ):
        _put(probe, "miniverl.reward.revision", "0" * 40)
    try:
        native_probe = build_native(
            probe, cycles=_get(probe, "trainer.total_training_steps", None) or 1
        )
    except (ConfigError, ValueError, TypeError, KeyError) as exc:
        native_probe = None
        rejections.append(
            {"field": "experiment", "reason": str(exc), "classification": "unsupported"}
        )
    required = []
    for field in ("data.train_files", "data.val_files"):
        value = _get(effective, field, [])
        paths = [value] if isinstance(value, str) else value
        if not isinstance(paths, list) or any(
            not isinstance(item, str) or not Path(item).is_file() for item in paths
        ):
            required.append({"field": field, "reason": "bind existing local Parquet inputs"})
    if _get(effective, "trainer.total_training_steps", None) is None:
        required.append(
            {"field": "dataset", "reason": "derive epoch schedule from local, filtered dataset"}
        )
    if (
        not _get(effective, "reward.reward_model.enable", False)
        and not _get(effective, "miniverl.reward.provider", None)
        and not any(row["binding"] == "reward.function" for row in ledger)
    ):
        required.append(
            {
                "field": "reward.function",
                "reason": "bind the trusted upstream reward implementation",
            }
        )
    return {
        "profile": DIRECT_PROFILE,
        "upstream_tag": UPSTREAM_VERL_TAG,
        "upstream_commit": UPSTREAM_VERL_COMMIT,
        "source_config_sha256": hashlib.sha256(content).hexdigest(),
        "source_leaf_count": len(flatten(payload)),
        "field_rules_sha256": direct_rules_digest(),
        "semantic_status": "rejected" if rejections else "accepted",
        "execution_status": "unsupported_semantics"
        if rejections
        else "requires_bindings"
        if required
        else "preflight_required",
        "fields": rows,
        "rejections": rejections,
        "required_bindings": required,
        "runtime_bindings": ledger,
        "effective_source": effective,
        "ir_schema_validation_passed": native_probe is not None,
    }
