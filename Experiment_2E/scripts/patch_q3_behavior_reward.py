#!/usr/bin/env python3
"""Install the audited Qwen3-1.7B behavior-reward hook into veRL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ENGINE = Path("verl/workers/engine/fsdp/transformer_impl.py")
TRAINER = Path("verl/trainer/ppo/v1/trainer_base.py")
MAIN_SYNC = Path("verl/trainer/main_ppo_sync.py")
RAY_TRAINER = Path("verl/trainer/ppo/ray_trainer.py")
DAPO_TRAINER = Path("recipe/dapo/dapo_ray_trainer.py")
ENGINE_MARKER = "# Experiment 2E Q3 behavior reward: compact H2/H10 values."
TRAINER_MARKER = "# Experiment 2E Q3 behavior reward: add centered behavior bonus."
MAIN_SYNC_MARKER = "# Experiment 2E Q3 behavior reward: DAPO sync trainer hook."
RAY_TRAINER_MARKER = "# Experiment 2E Q3 behavior reward: ray trainer hook."
DAPO_TRAINER_MARKER = "# Experiment 2E Q3 behavior reward: DAPO advantage hook."
BACKUP_SUFFIX = ".experiment2e_q3_behavior.orig"


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label} anchor count={count}")
    return source.replace(old, new, 1)


def patch_engine(source: str) -> str:
    source = source.replace("\r\n", "\n")
    if ENGINE_MARKER in source:
        return source
    old = (
        '        capture_hidden_probe = bool(\n'
        '            forward_only\n'
        '            and tu.get_non_tensor_data(data=micro_batch, key="capture_hidden_probe", default=False)\n'
        '            and not tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)\n'
        '            and tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)\n'
        '            and not self.use_ulysses_sp\n'
        '        )\n'
    )
    new = old + (
        '        # Experiment 2E Q3 behavior reward: compact H2/H10 values.\n'
        '        capture_behavior = bool(\n'
        '            forward_only\n'
        '            and tu.get_non_tensor_data(data=micro_batch, key="experiment2e_behavior_reward", default=False)\n'
        '            and not tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)\n'
        '            and tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)\n'
        '            and not self.use_ulysses_sp\n'
        '        )\n'
        '        capture_hidden_any = capture_hidden_probe or capture_behavior\n'
    )
    source = replace_once(source, old, new, "engine capture")
    source = replace_once(source, "output_hidden_states=capture_hidden_probe,", "output_hidden_states=capture_hidden_any,", "engine output")
    source = replace_once(source, "raw_output.hidden_states if capture_hidden_probe else None", "raw_output.hidden_states if capture_hidden_any else None", "engine pending")
    old = (
        '                pending_hs = getattr(self, "_pending_hidden_states", None)\n'
        '                if pending_hs is not None:\n'
        '                    try:\n'
        '                        from verl.hidden_probe import engine_extract as _hp\n'
    )
    new = (
        '                pending_hs = getattr(self, "_pending_hidden_states", None)\n'
        '                if pending_hs is not None and tu.get_non_tensor_data(data=micro_batch, key="experiment2e_behavior_reward", default=False):\n'
        '                    from experiment_2e.online_behavior import reduce_behavior_batch\n'
        '                    behavior_values, behavior_coverage = reduce_behavior_batch(pending_hs, input_ids, micro_batch["loss_mask"])\n'
        '                    model_output["behavior_values"] = torch.from_numpy(behavior_values).to(log_probs.device)\n'
        '                    model_output["behavior_coverage"] = torch.from_numpy(behavior_coverage).to(log_probs.device)\n'
        '                if pending_hs is not None and tu.get_non_tensor_data(data=micro_batch, key="capture_hidden_probe", default=False):\n'
        '                    try:\n'
        '                        from verl.hidden_probe import engine_extract as _hp\n'
    )
    return replace_once(source, old, new, "engine hidden reduction")


def patch_trainer(source: str) -> str:
    source = source.replace("\r\n", "\n")
    if TRAINER_MARKER in source:
        method = '    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:\n'
        prefix, suffix = source.split(method, 1)
        anchor = '        """Compute the advantage of the batch."""\n'
        flag = '        behavior_reward_enabled = (\n            os.getenv("EXPERIMENT_2E_BEHAVIOR_REWARD") == "1"\n'
        if flag not in suffix[:400]:
            suffix = replace_once(
                suffix,
                anchor,
                anchor
                + '        behavior_reward_enabled = (\n'
                + '            os.getenv("EXPERIMENT_2E_BEHAVIOR_REWARD") == "1"\n'
                + '            and not batch.extra_info.get("experiment2e_disable_behavior_reward", False)\n'
                + '        )\n',
                "trainer advantage behavior flag",
            )
        source = prefix + method + suffix
        unconditional = (
            '            "behavior_values",\n'
            '            "behavior_coverage",\n'
            '        ]\n'
            '        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)\n'
        )
        conditional = (
            '        ]\n'
            '        if behavior_reward_enabled:\n'
            '            fields.extend(["behavior_values", "behavior_coverage"])\n'
            '        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)\n'
        )
        if source.count(unconditional) == 1:
            source = source.replace(unconditional, conditional, 1)
        return source
    old = (
        '        online_hidden_enabled = (\n'
        '            os.getenv("EXPERIMENT_2E_ONLINE_HIDDEN") == "1"\n'
        '            and not batch.extra_info.get("experiment2e_disable_online_hidden", False)\n'
        '        )\n'
    )
    new = old + (
        '        behavior_reward_enabled = (\n'
        '            os.getenv("EXPERIMENT_2E_BEHAVIOR_REWARD") == "1"\n'
        '            and not batch.extra_info.get("experiment2e_disable_behavior_reward", False)\n'
        '        )\n'
    )
    source = replace_once(source, old, new, "trainer behavior flag")
    source = replace_once(source, '        batch.extra_info["experiment2e_online_hidden"] = online_hidden_enabled\n', '        batch.extra_info["experiment2e_online_hidden"] = online_hidden_enabled\n        batch.extra_info["experiment2e_behavior_reward"] = behavior_reward_enabled\n', "trainer metadata")
    old = '        if online_hidden_enabled:\n            fields.extend(["experiment2e_online_values", "experiment2e_online_coverage", "rm_scores"])\n'
    new = old + '        if behavior_reward_enabled:\n            fields.extend(["behavior_values", "behavior_coverage"])\n'
    source = replace_once(source, old, new, "trainer fields")
    source = replace_once(source, '        write_back_fields = ["old_log_probs", "entropy"]\n', '        write_back_fields = ["old_log_probs", "entropy"]\n        if behavior_reward_enabled:\n            write_back_fields.extend(["behavior_values", "behavior_coverage"])\n', "trainer write fields")
    source = replace_once(source, '            "extra_fields",\n        ]\n', '            "extra_fields",\n            "behavior_values",\n            "behavior_coverage",\n        ]\n', "trainer advantage fields")
    old = '        data = DataProto(batch=data.to_padded_tensor())\n        if os.getenv("EXPERIMENT_2E_ARM", "A").upper() == "B":\n'
    new = (
        '        data = DataProto(batch=data.to_padded_tensor())\n'
        '        # Experiment 2E Q3 behavior reward: add centered behavior bonus.\n'
        '        if behavior_reward_enabled:\n'
        '            from experiment_2e.online_behavior import compute_behavior_advantage\n'
        '            behavior_bonus, behavior_metrics = compute_behavior_advantage(\n'
        '                data.batch["behavior_values"].float(), data.batch["behavior_coverage"].bool(),\n'
        '                data.batch["uid"].tolist(), lambda_=float(os.getenv("EXPERIMENT_2E_BEHAVIOR_LAMBDA", "0.2")),\n'
        '                expected_group_size=int(self.config.actor_rollout_ref.rollout.n),\n'
        '            )\n'
        '            metrics.update(behavior_metrics)\n'
        '            lengths = data.batch["response_mask"].sum(dim=1).to(dtype=torch.long)\n'
        '            original_scores = data.batch["rm_scores"].clone()\n'
        '            padded_scores = original_scores.clone()\n'
        '            for row_index, length in enumerate(lengths.tolist()):\n'
        '                if length > 0:\n'
        '                    padded_scores[row_index].zero_()\n'
        '                    padded_scores[row_index, length - 1] = original_scores[row_index, length - 1] + float(behavior_bonus[row_index])\n'
        '            data.batch["rm_scores"] = padded_scores\n'
        '        if os.getenv("EXPERIMENT_2E_ARM", "A").upper() == "B":\n'
    )
    return replace_once(source, old, new, "trainer reward insertion")


def patch_main_sync(source: str) -> str:
    """Patch the DAPO synchronous trainer, which overrides trainer_base methods."""
    source = source.replace("\r\n", "\n")
    if MAIN_SYNC_MARKER in source:
        return source

    old = (
        '        """Compute the old log prob of the batch."""\n'
        '        # Operating Mode Selection:\n'
    )
    new = (
        '        """Compute the old log prob of the batch."""\n'
        '        # Experiment 2E Q3 behavior reward: DAPO sync trainer hook.\n'
        '        behavior_reward_enabled = (\n'
        '            os.getenv("EXPERIMENT_2E_BEHAVIOR_REWARD") == "1"\n'
        '            and not batch.extra_info.get("experiment2e_disable_behavior_reward", False)\n'
        '        )\n'
        '        batch.extra_info["experiment2e_behavior_reward"] = behavior_reward_enabled\n'
        '        # Operating Mode Selection:\n'
    )
    source = replace_once(source, old, new, "main sync old-log-prob flag")
    source = replace_once(
        source,
        '        fields = ["entropy", "log_probs", "response_mask"]\n',
        '        fields = ["entropy", "log_probs", "response_mask"]\n'
        '        if behavior_reward_enabled:\n'
        '            fields.extend(["behavior_values", "behavior_coverage"])\n',
        "main sync old-log-prob fields",
    )
    source = replace_once(
        source,
        '        batch = tq.kv_batch_put(\n'
        '            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")\n'
        '        )\n',
        '        write_back_fields = ["old_log_probs", "entropy"]\n'
        '        if behavior_reward_enabled:\n'
        '            write_back_fields.extend(["behavior_values", "behavior_coverage"])\n'
        '        batch = tq.kv_batch_put(\n'
        '            keys=batch.keys, partition_id=batch.partition_id, fields=data.select(*write_back_fields)\n'
        '        )\n',
        "main sync old-log-prob writeback",
    )

    old = (
        '    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:\n'
        '        """Compute the advantage of the batch."""\n'
        '        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]\n'
    )
    new = (
        '    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:\n'
        '        """Compute the advantage of the batch."""\n'
        '        behavior_reward_enabled = (\n'
        '            os.getenv("EXPERIMENT_2E_BEHAVIOR_REWARD") == "1"\n'
        '            and not batch.extra_info.get("experiment2e_disable_behavior_reward", False)\n'
        '        )\n'
        '        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]\n'
        '        if behavior_reward_enabled:\n'
        '            fields.extend(["behavior_values", "behavior_coverage"])\n'
    )
    source = replace_once(source, old, new, "main sync advantage flag")
    old = (
        '        data = DataProto(batch=data.to_padded_tensor())\n'
        '        data.batch["token_level_scores"] = data.batch["rm_scores"]\n'
    )
    new = (
        '        data = DataProto(batch=data.to_padded_tensor())\n'
        '        # Experiment 2E Q3 behavior reward: add centered behavior bonus.\n'
        '        if behavior_reward_enabled:\n'
        '            from experiment_2e.online_behavior import compute_behavior_advantage\n'
        '            behavior_bonus, behavior_metrics = compute_behavior_advantage(\n'
        '                data.batch["behavior_values"].float(), data.batch["behavior_coverage"].bool(),\n'
        '                data.batch["uid"].tolist(), lambda_=float(os.getenv("EXPERIMENT_2E_BEHAVIOR_LAMBDA", "0.2")),\n'
        '                expected_group_size=int(self.config.actor_rollout_ref.rollout.n),\n'
        '            )\n'
        '            metrics.update(behavior_metrics)\n'
        '            lengths = data.batch["response_mask"].sum(dim=1).to(dtype=torch.long)\n'
        '            original_scores = data.batch["rm_scores"].clone()\n'
        '            padded_scores = original_scores.clone()\n'
        '            for row_index, length in enumerate(lengths.tolist()):\n'
        '                if length > 0:\n'
        '                    padded_scores[row_index].zero_()\n'
        '                    padded_scores[row_index, length - 1] = original_scores[row_index, length - 1] + float(behavior_bonus[row_index])\n'
        '            data.batch["rm_scores"] = padded_scores\n'
        '        data.batch["token_level_scores"] = data.batch["rm_scores"]\n'
    )
    return replace_once(source, old, new, "main sync reward insertion")


def patch_ray_trainer(source: str) -> str:
    """Pass the behavior capture flag through the actual RayPPOTrainer actor call."""
    source = source.replace("\r\n", "\n")
    if RAY_TRAINER_MARKER in source:
        return source
    old = (
        '    def _compute_old_log_prob(self, batch: DataProto):\n'
        '        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free\n'
        '        # step 1: convert dataproto to tensordict.\n'
        '        batch_td = batch.to_tensordict()\n'
    )
    new = (
        '    def _compute_old_log_prob(self, batch: DataProto):\n'
        '        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free\n'
        '        # step 1: convert dataproto to tensordict.\n'
        '        batch_td = batch.to_tensordict()\n'
        '        # Experiment 2E Q3 behavior reward: ray trainer hook.\n'
        '        behavior_reward_enabled = (\n'
        '            os.getenv("EXPERIMENT_2E_BEHAVIOR_REWARD") == "1"\n'
        '            and not batch.meta_info.get("experiment2e_disable_behavior_reward", False)\n'
        '        )\n'
    )
    source = replace_once(source, old, new, "ray trainer behavior flag")
    source = replace_once(
        source,
        '            compute_loss=False,\n'
        '            capture_hidden_probe=capture_hidden_probe,\n'
        '        )\n',
        '            compute_loss=False,\n'
        '            capture_hidden_probe=capture_hidden_probe,\n'
        '            experiment2e_behavior_reward=behavior_reward_enabled,\n'
        '        )\n',
        "ray trainer behavior metadata",
    )
    source = replace_once(
        source,
        '        hidden_probe = tu.get(output, "hidden_probe", default=None) if capture_hidden_probe else None\n',
        '        hidden_probe = tu.get(output, "hidden_probe", default=None) if capture_hidden_probe else None\n'
        '        behavior_values = tu.get(output, "behavior_values", default=None) if behavior_reward_enabled else None\n'
        '        behavior_coverage = tu.get(output, "behavior_coverage", default=None) if behavior_reward_enabled else None\n',
        "ray trainer behavior output",
    )
    source = replace_once(
        source,
        '        if routed_experts is not None:\n'
        '            result["routed_experts"] = routed_experts\n',
        '        if routed_experts is not None:\n'
        '            result["routed_experts"] = routed_experts\n'
        '        if behavior_values is not None:\n'
        '            result["behavior_values"] = behavior_values.float()\n'
        '        if behavior_coverage is not None:\n'
        '            result["behavior_coverage"] = behavior_coverage.bool()\n',
        "ray trainer behavior result",
    )
    return source


def patch_dapo_trainer(source: str) -> str:
    """Add the behavior bonus immediately before DAPO calls compute_advantage."""
    source = source.replace("\r\n", "\n")
    if DAPO_TRAINER_MARKER in source:
        return source
    anchor = (
        '                    with marked_timer("adv", timing_raw, "brown"):\n'
        '                        # compute advantages, executed on the driver process\n'
        '                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)\n'
    )
    replacement = (
        '                    # Experiment 2E Q3 behavior reward: DAPO advantage hook.\n'
        '                    if os.getenv("EXPERIMENT_2E_BEHAVIOR_REWARD") == "1" and "behavior_values" in batch.batch:\n'
        '                        from experiment_2e.online_behavior import compute_behavior_advantage\n'
        '                        behavior_bonus, behavior_metrics = compute_behavior_advantage(\n'
        '                            batch.batch["behavior_values"].float(), batch.batch["behavior_coverage"].bool(),\n'
        '                            batch.non_tensor_batch["uid"].tolist(),\n'
        '                            lambda_=float(os.getenv("EXPERIMENT_2E_BEHAVIOR_LAMBDA", "0.2")),\n'
        '                            expected_group_size=int(self.config.actor_rollout_ref.rollout.n),\n'
        '                        )\n'
        '                        metrics.update(behavior_metrics)\n'
        '                        lengths = batch.batch["response_mask"].sum(dim=1).to(dtype=torch.long)\n'
        '                        terminal_scores = batch.batch["rm_scores"].clone()\n'
        '                        for row_index, length in enumerate(lengths.tolist()):\n'
        '                            if length > 0:\n'
        '                                terminal_scores[row_index, length - 1] += float(behavior_bonus[row_index])\n'
        '                        batch.batch["rm_scores"] = terminal_scores\n'
        '                        batch.batch.pop("behavior_values", None)\n'
        '                        batch.batch.pop("behavior_coverage", None)\n'
        '                    with marked_timer("adv", timing_raw, "brown"):\n'
        '                        # compute advantages, executed on the driver process\n'
        '                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)\n'
    )
    return replace_once(source, anchor, replacement, "dapo behavior reward insertion")


def apply_patch(root: Path, dry_run: bool) -> dict[str, object]:
    results = []
    for relative, function in (
        (ENGINE, patch_engine),
        (TRAINER, patch_trainer),
        (MAIN_SYNC, patch_main_sync),
        (RAY_TRAINER, patch_ray_trainer),
        (DAPO_TRAINER, patch_dapo_trainer),
    ):
        target = root / relative
        before = target.read_bytes()
        after = function(before.decode("utf-8")).encode("utf-8")
        item: dict[str, object] = {"target": str(target), "before_sha256": _sha(before), "after_sha256": _sha(after), "changed": before != after}
        if before != after and not dry_run:
            backup = Path(str(target) + BACKUP_SUFFIX)
            if backup.exists() and backup.read_bytes() != before and TRAINER_MARKER not in before.decode("utf-8"):
                raise RuntimeError(f"backup mismatch: {backup}")
            if not backup.exists():
                _atomic_write(backup, before)
            _atomic_write(target, after)
            item["backup"] = str(backup)
        results.append(item)
    return {"dry_run": dry_run, "files": results}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = apply_patch(args.verl_root, args.dry_run)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
