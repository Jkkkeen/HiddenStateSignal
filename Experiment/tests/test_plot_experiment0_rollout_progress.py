from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_experiment0_rollout_progress import (
    FROZEN_SIGNALS,
    main,
    question_weighted_summary,
    rollout_curve,
    select_signal,
)


def _synthetic_all_four_signals() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    representation_layers = sorted(
        {(spec.representation, spec.layer) for spec in FROZEN_SIGNALS}
    )
    for spec_index, (representation, layer) in enumerate(representation_layers):
        for question_id in ("q1", "q2"):
            for rollout_id, is_correct in ((0, True), (1, False)):
                for progress_bin in (0, 1):
                    row: dict[str, object] = {
                        "question_id": question_id,
                        "rollout_id": rollout_id,
                        "is_correct": is_correct,
                        "representation": representation,
                        "layer": layer,
                        "progress_bin": progress_bin,
                        "think_length": 4096,
                    }
                    for feature_index, signal in enumerate(FROZEN_SIGNALS):
                        row[signal.feature] = float(
                            10 * spec_index
                            + feature_index
                            + progress_bin
                            + (1 if is_correct else -1)
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def test_select_signal_freezes_representation_layer_and_feature() -> None:
    frame = _synthetic_all_four_signals()
    selected = select_signal(frame, FROZEN_SIGNALS[0])

    assert selected["representation"].eq("mean_w128_s64").all()
    assert selected["layer"].eq(24).all()
    assert selected["value"].notna().all()
    assert set(selected["progress_bin"]) == {0, 1}


def test_summary_weights_questions_not_rollout_count() -> None:
    selected = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q1", "q2", "q1", "q2"],
            "rollout_id": [0, 1, 2, 0, 3, 1],
            "is_correct": [True, True, True, True, False, False],
            "progress_bin": [0, 0, 0, 0, 0, 0],
            "value": [0.0, 0.0, 0.0, 10.0, 2.0, 4.0],
        }
    )

    summary = question_weighted_summary(
        selected,
        FROZEN_SIGNALS[0],
        bootstrap=100,
        seed=7,
    )

    correct = summary[summary["is_correct"]].iloc[0]
    assert correct["mean"] == pytest.approx(5.0)
    assert correct["n_questions"] == 2
    assert correct["n_rollouts"] == 4


def test_summary_is_deterministic_and_does_not_create_missing_bins() -> None:
    selected = pd.DataFrame(
        {
            "question_id": ["q1", "q2", "q1", "q2"],
            "rollout_id": [0, 0, 0, 0],
            "is_correct": [True, True, True, True],
            "progress_bin": [0, 0, 2, 2],
            "value": [1.0, 3.0, 2.0, 4.0],
        }
    )

    first = question_weighted_summary(selected, FROZEN_SIGNALS[0], 200, 19)
    second = question_weighted_summary(selected, FROZEN_SIGNALS[0], 200, 19)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["progress_bin"]) == {0, 2}
    assert 1 not in set(first["progress_bin"])


def test_rollout_curve_preserves_missing_bin_as_gap() -> None:
    rollout = pd.DataFrame({"progress_bin": [0, 2], "value": [1.0, 3.0]})

    x, y = rollout_curve(rollout)

    np.testing.assert_array_equal(x, np.arange(10))
    assert y[0] == 1.0
    assert np.isnan(y[1])
    assert y[2] == 3.0


def test_cli_writes_png_csv_report_and_idempotent_parent_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "features.parquet"
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    _synthetic_all_four_signals().to_parquet(input_path, index=False)
    parent = output_dir / "LONG_EXPERIMENT_0_RESULTS.md"
    parent.write_text("# Existing report\n", encoding="utf-8")
    argv = [
        "plot_experiment0_rollout_progress.py",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--bootstrap",
        "20",
        "--seed",
        "11",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()
    main()

    figure = output_dir / "figures/E0_F8_rollout_progress_spaghetti.png"
    summary = output_dir / "long_experiment_0_spaghetti_summary.csv"
    report = output_dir / "E0_F8_ROLLOUT_PROGRESS_REPORT.md"
    assert figure.stat().st_size > 0
    assert summary.stat().st_size > 0
    assert report.stat().st_size > 0
    text = parent.read_text(encoding="utf-8")
    assert text.count("## Rollout-Progress Spaghetti Addendum") == 1
