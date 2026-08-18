import numpy as np
import pandas as pd

from vertical.depth import analyze_condition_delta, summarize_depth_change


def policy_frame():
    rows = []
    depths = [0.0, 1 / 3, 2 / 3, 1.0]
    for condition, model, values in (
        ("base", "MiMo-Base", [0.0, 0.0, 0.0, 0.0]),
        ("sft", "MiMo-SFT", [0.0, 1.0, 2.0, 1.0]),
    ):
        for layer, (depth, value) in enumerate(zip(depths, values, strict=True)):
            rows.append(
                {
                    "model_family": "mimo",
                    "model_name": model,
                    "condition": condition,
                    "checkpoint": None,
                    "global_step": None,
                    "training_progress": None,
                    "stage": 0,
                    "representation": "last_s32",
                    "layer_index": layer,
                    "relative_depth": depth,
                    "profile_mean": value,
                    "metric": "v1_raw_update_norm",
                }
            )
    return pd.DataFrame(rows)


def test_depth_summary_uses_absolute_change_mass():
    result = summarize_depth_change(
        policy_frame(),
        metric="v1_raw_update_norm",
        base_condition="base",
    )
    sft = result.loc[result["condition"] == "sft"].iloc[0]

    assert sft["depth_mass"] == 4.0
    assert sft["peak_depth"] == 2 / 3
    assert sft["depth_center"] == 2 / 3
    assert sft["early_mass"] == 0.25
    assert sft["middle_mass"] == 0.5
    assert sft["late_mass"] == 0.25


def test_condition_delta_aligns_by_relative_layer_keys():
    result = analyze_condition_delta(
        policy_frame(),
        base_condition="base",
        target_condition="sft",
    )

    assert result.sort_values("layer_index")["condition_delta"].tolist() == [0.0, 1.0, 2.0, 1.0]
    assert result["paired_layers"].unique().tolist() == [4]


def test_missing_base_condition_fails_closed():
    frame = policy_frame().loc[lambda data: data["condition"] == "sft"]

    try:
        summarize_depth_change(frame, metric="v1_raw_update_norm")
    except ValueError as exc:
        assert "Base" in str(exc)
    else:
        raise AssertionError("missing Base profile should fail")

