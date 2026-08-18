from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_q3_vertical_dynamics_report.py"
SPEC = importlib.util.spec_from_file_location("q3_vertical_report", SCRIPT)
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_pass_at_k_uses_roll8_combinatorics():
    assert report.pass_at_k_from_count(0, 8, 4) == 0.0
    assert report.pass_at_k_from_count(8, 8, 4) == 1.0
    assert report.pass_at_k_from_count(1, 8, 1) == 0.125
    assert report.pass_at_k_from_count(1, 8, 4) == 0.5


def test_pass_at_k_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        report.pass_at_k_from_count(9, 8, 4)
    with pytest.raises(ValueError):
        report.pass_at_k_from_count(1, 8, 0)


def test_paired_bootstrap_is_deterministic():
    values = np.arange(24, dtype=float).reshape(3, 8)
    first = report.paired_bootstrap(values, draws=50, seed=20260818)
    second = report.paired_bootstrap(values, draws=50, seed=20260818)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


def test_validate_result_root_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_status"):
        report.validate_result_root(tmp_path)


def test_profile_dynamics_has_zero_base_and_ordered_updates():
    values = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    cumulative, adjacent, cosine, level = report._profile_dynamics(values)
    assert cumulative[0] == pytest.approx(0.0)
    assert cumulative[2] > cumulative[1] > cumulative[0]
    assert adjacent[1] > adjacent[2] > 0
    assert cosine[2] == pytest.approx(1.0)
    assert level.tolist() == pytest.approx([1.5, 3.0, 4.5])


def _behavior_questions() -> pd.DataFrame:
    rows = []
    for step in report.EXPECTED_STEPS:
        for question in range(report.QUESTIONS):
            base = (question + step // 25) % 9
            rows.append({
                "global_step": step,
                "question_index": question,
                "pass_at_1": float(base % 2),
                "pass_at_4": float(base % 3) / 2,
                "pass_at_8": float(base % 4) / 3,
            })
    return pd.DataFrame(rows)


def _fake_data():
    steps = list(report.EXPECTED_STEPS)
    behavior = pd.DataFrame({
        "global_step": steps,
        "pass_at_1": np.linspace(0.3965, 0.6333, len(steps)),
        "pass_at_1_low": np.linspace(0.35, 0.60, len(steps)),
        "pass_at_1_high": np.linspace(0.44, 0.67, len(steps)),
        "pass_at_4": np.linspace(0.55, 0.66, len(steps)),
        "pass_at_4_low": np.linspace(0.50, 0.62, len(steps)),
        "pass_at_4_high": np.linspace(0.60, 0.70, len(steps)),
        "pass_at_8": np.linspace(0.70, 0.78, len(steps)),
        "pass_at_8_low": np.linspace(0.65, 0.74, len(steps)),
        "pass_at_8_high": np.linspace(0.75, 0.82, len(steps)),
        "delta_pass_at_1": [0.0] + [0.02] * (len(steps) - 1),
        "delta_pass_at_1_low": [0.0] + [-0.01] * (len(steps) - 1),
        "delta_pass_at_1_high": [0.0] + [0.04] * (len(steps) - 1),
    })
    questions = _behavior_questions()
    effects = report.paired_behavior_effects(questions)
    vertical_rows = []
    for family, variant, label in (
        ("V1", "raw", "V1 raw update norm"),
        ("V3", "demean", "V3 demeaned state angle"),
        ("V4", "median", "V4 turning angle"),
        ("V4C", "weighted", "V4C weighted turning"),
        ("V8", "ER", "V8 layer-update ER"),
    ):
        for step in steps:
            vertical_rows.append({
                "metric_family": family,
                "metric_variant": variant,
                "metric_label": label,
                "representation": "mean_w128_s32",
                "global_step": step,
                "level_mean": 1.0 + step / 1000,
                "cumulative_relative_l1": step / 2500,
                "adjacent_relative_l1": 0.001,
                "update_cosine": 0.9,
            })
    vertical = pd.DataFrame(vertical_rows)
    auc = pd.DataFrame({
        "global_step": steps,
        "median": np.linspace(0.51, 0.53, len(steps)),
        "p90": np.linspace(0.55, 0.58, len(steps)),
        "maximum": np.linspace(0.62, 0.67, len(steps)),
    })
    auc_late = pd.DataFrame({
        "global_step": [175, 200, 225, 250, 175, 200, 225, 250],
        "metric": ["v3_demean_state_angle"] * 4 + ["v1_relative_update_norm"] * 4,
        "question_equal_auc": [0.64, 0.65, 0.66, 0.65, 0.35, 0.36, 0.34, 0.35],
        "separability": [0.64, 0.65, 0.66, 0.65, 0.65, 0.64, 0.66, 0.65],
        "n_mixed_questions": [120] * 8,
        "n_pairs": [400] * 8,
    })
    horizontal = pd.DataFrame({
        "family": ["H2", "H5"] * 10,
        "block_start": list(np.repeat([1, 26, 51, 76, 101, 126, 151, 176, 201, 226], 2)),
        "block_end": list(np.repeat([25, 50, 75, 100, 125, 150, 175, 200, 225, 250], 2)),
        "correct_mean": [0.6] * 20,
        "wrong_mean": [0.5] * 20,
        "auroc": [0.58] * 20,
        "auc_questions": [16] * 20,
        "cohort": ["online training probe"] * 20,
    })
    images = {key: "data:image/png;base64,AA==" for key in (
        "ability_passk", "ability_delta_pass1", "vertical_cumulative", "vertical_adjacent",
        "v1_heatmap", "v3_heatmap", "v4_heatmap", "v3_profiles", "auc_distribution",
        "v3_auc", "v1_relative_auc", "horizontal_h2_h5",
    )}
    data = {
        "behavior_summary": behavior,
        "behavior_questions": questions,
        "behavior_effects": effects,
        "vertical_dynamics": vertical,
        "auc_summary": auc,
        "auc_late": auc_late,
        "horizontal": horizontal,
        "provenance": {"analysis_input_hashes": ["fixture"]},
        "audit": {"passed": True},
    }
    return data, images


def test_render_html_contains_required_sections():
    data, images = _fake_data()
    rendered = report.render_html(data, images)
    for section in report.REQUIRED_SECTIONS:
        assert f'id="{section}"' in rendered
    assert rendered.count("data:image/png;base64,") == len(images)
    assert "39.65%" in rendered
    assert "63.33%" in rendered
    assert "step175–200" in rendered
    assert "H2" in rendered and "reward shaping" in rendered
    assert "behavior_rows.join" not in rendered


def test_encode_png_returns_embeddable_data_uri():
    encoded = report.encode_png(b"\x89PNG\r\n\x1a\nfixture")
    assert encoded.startswith("data:image/png;base64,")


def test_render_summary_figures_produces_nonempty_pngs():
    data, _ = _fake_data()
    figures = report.render_summary_figures(data)
    assert set(figures) == {
        "ability_passk",
        "ability_delta_pass1",
        "vertical_cumulative",
        "vertical_adjacent",
        "auc_distribution",
        "horizontal_h2_h5",
    }
    assert all(payload.startswith(b"\x89PNG") for payload in figures.values())
    assert all(len(payload) > 10_000 for payload in figures.values())
