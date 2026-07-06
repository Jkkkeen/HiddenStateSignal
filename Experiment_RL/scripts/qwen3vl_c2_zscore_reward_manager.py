#!/usr/bin/env python3
"""C2 group-zscored option-gain reward manager for VERL smoke runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def group_zscore_bonus(values: list[float], clip_value: float = 2.0, eps: float = 1e-6) -> list[float]:
    arr = np.asarray([_finite_float(value) for value in values], dtype=np.float64)
    if arr.size == 0:
        return []
    std = float(arr.std(ddof=0))
    if std < eps:
        return [0.0 for _ in values]
    z = (arr - float(arr.mean())) / std
    z = np.clip(z, -abs(float(clip_value)), abs(float(clip_value)))
    return [float(value) for value in z]


def _group_key(row: dict[str, Any]) -> str:
    for key in ("question_id", "uid", "source_index", "index"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(row.get("extra_info", {}).get("question_id", "unknown"))


def _prompt_uid_from_tq_key(key: Any) -> str:
    parts = str(key).rsplit("_", 2)
    return parts[0] if len(parts) == 3 else str(key)


def merge_group_zscore_rewards(
    rows: list[dict[str, Any]],
    bonus_lambda: float = 0.2,
    zscore_clip: float = 2.0,
    eps: float = 1e-6,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    merged = [dict(row) for row in rows]
    for index, row in enumerate(merged):
        groups[_group_key(row)].append(index)

    for indices in groups.values():
        raw_values = [_finite_float(merged[i].get("option_logit_gain_raw")) for i in indices]
        bonuses = group_zscore_bonus(raw_values, clip_value=zscore_clip, eps=eps)
        std_b = float(np.asarray(bonuses, dtype=np.float64).std(ddof=0)) if bonuses else 0.0
        std_raw = float(np.asarray(raw_values, dtype=np.float64).std(ddof=0)) if raw_values else 0.0
        for i, bonus in zip(indices, bonuses, strict=True):
            answer_reward = _finite_float(merged[i].get("answer_reward"))
            merged[i]["option_logit_gain_zscore"] = float(bonus)
            merged[i]["option_bonus"] = float(float(bonus_lambda) * bonus)
            merged[i]["score"] = float(answer_reward + float(bonus_lambda) * bonus)
            merged[i]["within_group_std_raw_gain"] = std_raw
            merged[i]["within_group_std_b_gain"] = std_b
            merged[i]["c2_group_size"] = float(len(indices))
    return merged


def compute_group_diagnostics(rows: list[dict[str, Any]], eps: float = 1e-6) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    total = len(groups)
    nonzero_b = 0
    changed_rank = 0
    all_same_answer = 0
    all_same_with_b_rank = 0
    corr_values: list[float] = []

    for group in groups.values():
        answers = np.asarray([_finite_float(row.get("answer_reward")) for row in group], dtype=np.float64)
        b_gain = np.asarray([_finite_float(row.get("option_logit_gain_zscore")) for row in group], dtype=np.float64)
        b_std = float(b_gain.std(ddof=0)) if b_gain.size else 0.0
        if b_std > eps:
            nonzero_b += 1
        answer_std = float(answers.std(ddof=0)) if answers.size else 0.0
        if answer_std <= eps:
            all_same_answer += 1
            if b_std > eps:
                all_same_with_b_rank += 1
        else:
            if b_std > eps:
                centered_answers = answers - float(answers.mean())
                centered_b = b_gain - float(b_gain.mean())
                denom = float(np.sqrt(np.sum(centered_answers**2) * np.sum(centered_b**2)))
                corr = float(np.sum(centered_answers * centered_b) / denom) if denom > eps else 0.0
                if np.isfinite(corr):
                    corr_values.append(corr)
        answer_order = np.argsort(answers, kind="stable")
        total_order = np.argsort(
            np.asarray([_finite_float(row.get("score")) for row in group], dtype=np.float64),
            kind="stable",
        )
        if len(group) > 1 and not np.array_equal(answer_order, total_order):
            changed_rank += 1

    return {
        "c2_groups_total": float(total),
        "c2_groups_with_nonzero_b_gain_std": float(nonzero_b),
        "c2_frac_groups_with_nonzero_b_gain_std": float(nonzero_b / total) if total else 0.0,
        "c2_groups_where_b_gain_changes_ranking": float(changed_rank),
        "c2_frac_groups_where_b_gain_changes_ranking": float(changed_rank / total) if total else 0.0,
        "c2_all_same_answer_groups": float(all_same_answer),
        "c2_all_same_answer_groups_with_b_gain_ranking": float(all_same_with_b_rank),
        "c2_frac_all_same_answer_groups_with_b_gain_ranking": float(all_same_with_b_rank / all_same_answer)
        if all_same_answer
        else 0.0,
        "c2_within_group_corr_answer_b_gain_mean": float(np.mean(corr_values)) if corr_values else 0.0,
    }


def _is_c2_probe_key(key: str) -> bool:
    return (
        key.startswith("option_logit_gain")
        or key.startswith("option_margin_")
        or key.startswith("option_probe_")
        or key.startswith("within_group_std_")
        or key == "c2_group_size"
    )


def merge_v1_extra_fields_preserving_c2_probe(prior_extra_fields: Any, reward_extra_fields: Any) -> np.ndarray:
    """Merge reward output with previously computed C2 vLLM probe extras.

    In the v1 trainer, the vLLM option probe writes into TransferQueue before
    colocated answer reward is computed. The answer reward output can contain
    deferred option fields set to zero; those must not overwrite the real probe
    values from the earlier trainer-side probe.
    """

    prior_list = list(prior_extra_fields.tolist() if hasattr(prior_extra_fields, "tolist") else prior_extra_fields)
    reward_list = list(reward_extra_fields.tolist() if hasattr(reward_extra_fields, "tolist") else reward_extra_fields)
    if len(prior_list) != len(reward_list):
        raise ValueError(f"extra_fields length mismatch: {len(prior_list)} != {len(reward_list)}")

    merged: list[dict[str, Any]] = []
    for prior_item, reward_item in zip(prior_list, reward_list, strict=True):
        prior_item = dict(prior_item or {})
        reward_item = dict(reward_item or {})
        prior_info = dict(prior_item.get("reward_extra_info", {}))
        reward_info = dict(reward_item.get("reward_extra_info", {}))

        merged_item = dict(prior_item)
        merged_item.update(reward_item)
        merged_info = dict(prior_info)
        merged_info.update(reward_info)
        for key, value in prior_info.items():
            if _is_c2_probe_key(str(key)):
                merged_info[key] = value
        merged_item["reward_extra_info"] = merged_info
        merged.append(merged_item)

    out = np.empty(len(merged), dtype=object)
    out[:] = merged
    return out


def build_v1_batch_postprocess_updates(
    batch_keys: list[Any],
    extra_fields: Any,
    bonus_lambda: float = 0.2,
    zscore_clip: float = 2.0,
) -> tuple[list[float], np.ndarray, dict[str, float]]:
    extra_list = list(extra_fields.tolist() if hasattr(extra_fields, "tolist") else extra_fields)
    rows: list[dict[str, Any]] = []
    for key, extra_field in zip(batch_keys, extra_list, strict=True):
        extra_field = dict(extra_field or {})
        reward_extra = dict(extra_field.get("reward_extra_info", {}))
        question_id = reward_extra.get("question_id") or _prompt_uid_from_tq_key(key)
        rows.append(
            {
                "question_id": question_id,
                "uid": reward_extra.get("uid"),
                "source_index": reward_extra.get("source_index"),
                "index": reward_extra.get("index"),
                "answer_reward": _finite_float(reward_extra.get("answer_reward", reward_extra.get("acc"))),
                "option_logit_gain_raw": _finite_float(reward_extra.get("option_logit_gain_raw")),
            }
        )

    merged = merge_group_zscore_rewards(rows, bonus_lambda=bonus_lambda, zscore_clip=zscore_clip)
    diagnostics = compute_group_diagnostics(merged)
    scores: list[float] = []
    updated_extra: list[dict[str, Any]] = []
    for extra_field, row in zip(extra_list, merged, strict=True):
        extra_field = dict(extra_field or {})
        reward_extra = dict(extra_field.get("reward_extra_info", {}))
        reward_extra.update(
            {
                "score": float(row["score"]),
                "option_logit_gain_zscore": float(row["option_logit_gain_zscore"]),
                "option_bonus": float(row["option_bonus"]),
                "within_group_std_raw_gain": float(row["within_group_std_raw_gain"]),
                "within_group_std_b_gain": float(row["within_group_std_b_gain"]),
                "c2_group_size": float(row["c2_group_size"]),
            }
        )
        reward_extra.update(diagnostics)
        extra_field["reward_extra_info"] = reward_extra
        scores.append(float(row["score"]))
        updated_extra.append(extra_field)

    updated_array = np.empty(len(updated_extra), dtype=object)
    updated_array[:] = updated_extra
    return scores, updated_array, diagnostics


try:
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
except Exception:  # pragma: no cover - local unit tests do not have VERL.

    class NaiveRewardManager:  # type: ignore[no-redef]
        @classmethod
        def init_class(cls, config: Any, tokenizer: Any) -> None:
            return None

        async def run_single(self, data: Any) -> dict[str, Any]:
            raise RuntimeError("VERL NaiveRewardManager is unavailable")

        @classmethod
        def assemble_rm_scores(cls, data: Any, scores: list[float]) -> torch.Tensor:
            return torch.tensor(scores, dtype=torch.float32)


class Qwen3VLC2ZscoreRewardManager(NaiveRewardManager):
    """VERL reward manager that applies group-zscored option-gain bonuses."""

    @classmethod
    def init_class(cls, config: Any, tokenizer: Any) -> None:
        super().init_class(config, tokenizer)
        import os

        reward_kwargs = dict(config.reward.custom_reward_function.get("reward_kwargs", {}))
        cls.bonus_lambda = float(reward_kwargs.get("bonus_lambda", os.environ.get("OPTION_GAIN_LAMBDA", 0.2)))
        cls.zscore_clip = float(reward_kwargs.get("zscore_clip", os.environ.get("OPTION_GAIN_ZSCORE_CLIP", 2.0)))

    async def run_single(self, data: Any) -> dict[str, Any]:
        result = await super().run_single(data)
        extra = dict(result.get("reward_extra_info", {}))
        # The scalar sent to assemble_rm_scores is raw answer reward. The final
        # zscored score is assembled once all rollouts from the batch are visible.
        result["reward_score"] = float(extra.get("answer_reward", extra.get("acc", result["reward_score"])))
        return result

    @classmethod
    def _merged_rows_from_outputs(cls, data: Any, outputs_flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
        extra_infos = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])
        rows: list[dict[str, Any]] = []
        for i, output in enumerate(outputs_flat):
            extra = extra_infos[i] if isinstance(extra_infos[i], dict) else {}
            reward_extra = dict(output.get("reward_extra_info", {}))
            rows.append(
                {
                    "question_id": extra.get("question_id", extra.get("source_index", i)),
                    "uid": data.non_tensor_batch.get("uid", [None] * len(data))[i]
                    if "uid" in data.non_tensor_batch
                    else None,
                    "answer_reward": _finite_float(reward_extra.get("answer_reward", output.get("reward_score"))),
                    "option_logit_gain_raw": _finite_float(reward_extra.get("option_logit_gain_raw")),
                }
            )
        return merge_group_zscore_rewards(
            rows,
            bonus_lambda=getattr(cls, "bonus_lambda", 0.2),
            zscore_clip=getattr(cls, "zscore_clip", 2.0),
        )

    @classmethod
    def postprocess_batch(cls, data: Any, outputs_flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = cls._merged_rows_from_outputs(data, outputs_flat)
        diagnostics = compute_group_diagnostics(merged)
        for output, row in zip(outputs_flat, merged, strict=True):
            output["reward_score"] = float(row["score"])
            extra = dict(output.get("reward_extra_info", {}))
            extra.update(
                {
                    "score": float(row["score"]),
                    "option_logit_gain_zscore": float(row["option_logit_gain_zscore"]),
                    "option_bonus": float(row["option_bonus"]),
                    "within_group_std_raw_gain": float(row["within_group_std_raw_gain"]),
                    "within_group_std_b_gain": float(row["within_group_std_b_gain"]),
                    "c2_group_size": float(row["c2_group_size"]),
                }
            )
            extra.update(diagnostics)
            output["reward_extra_info"] = extra
        print("[qwen3vl_c2_zscore_reward_manager] diagnostics", diagnostics, flush=True)
        return outputs_flat

    @classmethod
    def postprocess_v1_batch(cls, batch_keys: list[Any], extra_fields: Any) -> tuple[list[float], np.ndarray, dict[str, float]]:
        return build_v1_batch_postprocess_updates(
            batch_keys=batch_keys,
            extra_fields=extra_fields,
            bonus_lambda=getattr(cls, "bonus_lambda", 0.2),
            zscore_clip=getattr(cls, "zscore_clip", 2.0),
        )

    @classmethod
    def merge_v1_extra_fields(cls, prior_extra_fields: Any, reward_extra_fields: Any) -> np.ndarray:
        return merge_v1_extra_fields_preserving_c2_probe(prior_extra_fields, reward_extra_fields)

    @classmethod
    def assemble_rm_scores(cls, data: Any, scores: list[float]) -> torch.Tensor:
        return super().assemble_rm_scores(data, scores)
