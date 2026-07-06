import math
import importlib.util

import numpy as np

from scripts.qwen3vl_c2_zscore_reward_manager import (
    build_v1_batch_postprocess_updates,
    compute_group_diagnostics,
    group_zscore_bonus,
    merge_v1_extra_fields_preserving_c2_probe,
    merge_group_zscore_rewards,
)


def test_group_zscore_bonus_returns_zero_for_constant_group():
    bonuses = group_zscore_bonus([0.12, 0.12], clip_value=2.0)

    assert bonuses == [0.0, 0.0]


def test_group_zscore_bonus_normalizes_and_clips_group():
    bonuses = group_zscore_bonus([0.0, 2.0], clip_value=2.0)

    assert bonuses == [-1.0, 1.0]


def test_merge_group_zscore_rewards_uses_question_id_groups():
    rows = [
        {"question_id": "q1", "answer_reward": 1.0, "option_logit_gain_raw": -0.2},
        {"question_id": "q1", "answer_reward": 1.0, "option_logit_gain_raw": 0.2},
        {"question_id": "q2", "answer_reward": 0.0, "option_logit_gain_raw": 0.4},
        {"question_id": "q2", "answer_reward": 0.0, "option_logit_gain_raw": -0.4},
    ]

    merged = merge_group_zscore_rewards(rows, bonus_lambda=0.2, zscore_clip=2.0)

    assert [round(row["option_logit_gain_zscore"], 4) for row in merged] == [
        -1.0,
        1.0,
        1.0,
        -1.0,
    ]
    assert [round(row["score"], 4) for row in merged] == [0.8, 1.2, 0.2, -0.2]
    assert all(row["within_group_std_b_gain"] > 0 for row in merged)


def test_compute_group_diagnostics_counts_all_correct_and_all_wrong_ranking():
    rows = [
        {"question_id": "correct", "answer_reward": 1.0, "option_logit_gain_zscore": -1.0},
        {"question_id": "correct", "answer_reward": 1.0, "option_logit_gain_zscore": 1.0},
        {"question_id": "wrong", "answer_reward": 0.0, "option_logit_gain_zscore": -1.0},
        {"question_id": "wrong", "answer_reward": 0.0, "option_logit_gain_zscore": 1.0},
        {"question_id": "mixed", "answer_reward": 0.0, "option_logit_gain_zscore": 0.5},
        {"question_id": "mixed", "answer_reward": 1.0, "option_logit_gain_zscore": -0.5},
    ]

    diag = compute_group_diagnostics(rows)

    assert diag["c2_groups_total"] == 3
    assert diag["c2_groups_with_nonzero_b_gain_std"] == 3
    assert math.isclose(diag["c2_frac_groups_with_nonzero_b_gain_std"], 1.0)
    assert diag["c2_all_same_answer_groups"] == 2
    assert diag["c2_all_same_answer_groups_with_b_gain_ranking"] == 2
    assert math.isclose(diag["c2_frac_all_same_answer_groups_with_b_gain_ranking"], 1.0)


def test_reward_manager_importlib_load_then_normal_import_does_not_double_register():
    path = (
        "C:/Users/LENOVO/Desktop/A-G实验/AI-HiddenState/Experiment_RL/"
        "scripts/qwen3vl_c2_zscore_reward_manager.py"
    )
    spec = importlib.util.spec_from_file_location("custom_module_c2_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import scripts.qwen3vl_c2_zscore_reward_manager as normal_module

    assert module.Qwen3VLC2ZscoreRewardManager.__name__ == "Qwen3VLC2ZscoreRewardManager"
    assert normal_module.group_zscore_bonus([-0.2, 0.2]) == [-1.0, 1.0]


def test_build_v1_batch_postprocess_updates_groups_by_prompt_uid():
    extra_fields = np.array(
        [
            {
                "reward_extra_info": {
                    "answer_reward": 1.0,
                    "option_logit_gain_raw": -0.2,
                    "option_probe_failed": 0.0,
                }
            },
            {
                "reward_extra_info": {
                    "answer_reward": 1.0,
                    "option_logit_gain_raw": 0.2,
                    "option_probe_failed": 0.0,
                }
            },
        ],
        dtype=object,
    )

    scores, updated_extra_fields, diagnostics = build_v1_batch_postprocess_updates(
        batch_keys=["prompt-123_0_0", "prompt-123_1_0"],
        extra_fields=extra_fields,
        bonus_lambda=0.2,
        zscore_clip=2.0,
    )

    assert [round(score, 4) for score in scores] == [0.8, 1.2]
    infos = [item["reward_extra_info"] for item in updated_extra_fields]
    assert [info["option_logit_gain_zscore"] for info in infos] == [-1.0, 1.0]
    assert diagnostics["c2_all_same_answer_groups"] == 1.0
    assert diagnostics["c2_frac_all_same_answer_groups_with_b_gain_ranking"] == 1.0


def test_build_v1_batch_postprocess_updates_prefers_reward_question_id():
    extra_fields = np.array(
        [
            {
                "reward_extra_info": {
                    "question_id": "mathverse-q42",
                    "answer_reward": 1.0,
                    "option_logit_gain_raw": -0.2,
                    "option_probe_failed": 0.0,
                }
            },
            {
                "reward_extra_info": {
                    "question_id": "mathverse-q42",
                    "answer_reward": 1.0,
                    "option_logit_gain_raw": 0.2,
                    "option_probe_failed": 0.0,
                }
            },
        ],
        dtype=object,
    )

    scores, updated_extra_fields, diagnostics = build_v1_batch_postprocess_updates(
        batch_keys=["opaque-transfer-key-a", "opaque-transfer-key-b"],
        extra_fields=extra_fields,
        bonus_lambda=0.2,
        zscore_clip=2.0,
    )

    assert [round(score, 4) for score in scores] == [0.8, 1.2]
    infos = [item["reward_extra_info"] for item in updated_extra_fields]
    assert [info["option_logit_gain_zscore"] for info in infos] == [-1.0, 1.0]
    assert diagnostics["c2_groups_total"] == 1.0
    assert diagnostics["c2_frac_all_same_answer_groups_with_b_gain_ranking"] == 1.0


def test_merge_v1_extra_fields_preserves_c2_probe_over_deferred_reward():
    prior = np.array(
        [
            {
                "reward_extra_info": {
                    "question_id": "q1",
                    "option_logit_gain_raw": 0.3,
                    "option_probe_count": 4.0,
                    "option_probe_deferred": 0.0,
                    "option_margin_0": -0.1,
                    "option_margin_1": 0.2,
                }
            }
        ],
        dtype=object,
    )
    reward = np.array(
        [
            {
                "reward_extra_info": {
                    "question_id": "q1",
                    "answer_reward": 1.0,
                    "acc": 1.0,
                    "option_logit_gain_raw": 0.0,
                    "option_probe_count": 0.0,
                    "option_probe_deferred": 1.0,
                    "option_margin_0": np.nan,
                    "option_margin_1": np.nan,
                }
            }
        ],
        dtype=object,
    )

    merged = merge_v1_extra_fields_preserving_c2_probe(prior, reward)

    info = merged[0]["reward_extra_info"]
    assert info["answer_reward"] == 1.0
    assert info["option_logit_gain_raw"] == 0.3
    assert info["option_probe_count"] == 4.0
    assert info["option_probe_deferred"] == 0.0
    assert info["option_margin_0"] == -0.1
