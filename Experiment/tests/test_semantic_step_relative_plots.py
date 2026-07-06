import numpy as np
import pandas as pd

from pathlib import Path

from scripts.plot_semantic_step_relative import (
    add_relative_bins,
    binary_auc,
    correct_wrong_gap_by_bin,
    main,
    progress_label,
    within_question_auc_by_bin,
)


def test_add_relative_bins_clips_and_labels_deciles():
    df = pd.DataFrame({"relative_step_pos": [-0.1, 0.0, 0.05, 0.95, 1.0, 1.2]})

    out = add_relative_bins(df, n_bins=10)

    assert out["rel_bin"].tolist() == [0, 0, 0, 9, 9, 9]
    assert out["rel_bin_mid"].tolist() == [0.05, 0.05, 0.05, 0.95, 0.95, 0.95]


def test_correct_wrong_gap_by_bin_uses_question_level_pairs():
    df = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q1", "q1", "q2", "q2", "q2", "q2"],
            "rollout_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "layer": [24] * 8,
            "is_correct": [True, False, True, False, True, False, True, False],
            "rel_bin": [0, 0, 1, 1, 0, 0, 1, 1],
            "rel_bin_mid": [0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75],
            "score": [3.0, 1.0, 2.0, 4.0, 5.0, 2.0, 6.0, 1.0],
        }
    )

    out = correct_wrong_gap_by_bin(df, "score")

    got = {
        (row.layer, row.rel_bin): row.gap_mean
        for row in out.itertuples(index=False)
    }
    assert got[(24, 0)] == 2.5
    assert got[(24, 1)] == 1.5


def test_within_question_auc_by_bin_averages_mixed_question_aucs():
    df = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q1", "q1", "q2", "q2", "q2", "q2"],
            "rollout_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "layer": [36] * 8,
            "is_correct": [True, True, False, False, True, False, True, False],
            "rel_bin": [0] * 8,
            "rel_bin_mid": [0.5] * 8,
            "score": [4.0, 3.0, 2.0, 1.0, 1.0, 4.0, 2.0, 3.0],
        }
    )

    out = within_question_auc_by_bin(df, "score")

    assert len(out) == 1
    assert out.iloc[0]["questions"] == 2
    assert np.isclose(out.iloc[0]["auc_mean"], 0.5)
    assert np.isclose(binary_auc(np.array([True, False]), np.array([2.0, 1.0])), 1.0)


def test_progress_label_names_legacy_thinking_outputs():
    assert progress_label("think") == "relative thinking progress"
    assert progress_label("thinking") == "relative thinking progress"
    assert progress_label("answer") == "relative answer progress"


def test_main_writes_margin_and_gain_curve_figures(tmp_path, monkeypatch):
    rows = []
    for layer in [24, 36]:
        for qid in ["q1", "q2"]:
            for rid, is_correct in [(0, True), (1, False)]:
                for step_idx, rel_pos in enumerate([0.0, 0.5, 1.0]):
                    rows.append(
                        {
                            "question_id": qid,
                            "rollout_id": rid,
                            "layer": layer,
                            "segment": "answer",
                            "is_correct": is_correct,
                            "relative_step_pos": rel_pos,
                            "align_correct_state": 0.1 + rel_pos,
                            "align_correct_step": 0.2 + rel_pos,
                            "state_margin": (0.4 if is_correct else 0.1) + rel_pos,
                            "step_correct_gain": (0.05 if is_correct else -0.02) + 0.01 * step_idx,
                            "turn_to_correct": (0.2 if is_correct else 0.1) + rel_pos,
                        }
                    )
    pd.DataFrame(rows).to_parquet(tmp_path / "semantic_step_basin_steps.parquet", index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_semantic_step_relative.py",
            "--input-dir",
            str(tmp_path),
            "--bins",
            "3",
            "--layers",
            "24,36",
            "--bootstrap",
            "0",
        ],
    )

    main()

    assert (tmp_path / "figures" / "G8_state_margin_curve.png").is_file()
    assert (tmp_path / "figures" / "G9_step_correct_gain_curve.png").is_file()
    report = (tmp_path / "SEMANTIC_STEP_RELATIVE_ANALYSIS.md").read_text(encoding="utf-8")
    assert "G8_state_margin_curve.png" in report
    assert "G9_step_correct_gain_curve.png" in report
