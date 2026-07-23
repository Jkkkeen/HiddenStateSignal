from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_correct_trajectory_pairwise_geometry import main


def _synthetic_geometry() -> pd.DataFrame:
    rows = []
    for question_index in range(3):
        question_id = f"q{question_index}"
        labels = {rollout_id: rollout_id < 3 for rollout_id in range(6)}
        for layer in (24, 36):
            for progress_bin in (0, 1):
                for query_id, query_correct in labels.items():
                    for reference_id, reference_correct in labels.items():
                        if query_id == reference_id:
                            continue
                        cosine = (
                            0.95
                            if query_correct and reference_correct
                            else 0.50
                            if not query_correct and not reference_correct
                            else 0.0
                        )
                        rows.append(
                            {
                                "question_id": question_id,
                                "rollout_id": query_id,
                                "is_correct": query_correct,
                                "representation": "mean_w128_s64",
                                "span_id": progress_bin + 1,
                                "span_start": progress_bin * 64,
                                "span_end": progress_bin * 64 + 128,
                                "relative_progress": 0.25 + 0.5 * progress_bin,
                                "progress_bin": progress_bin,
                                "layer": layer,
                                "displacement_norm": 1.0 + 0.1 * query_correct,
                                "span_turn_cos": 0.0,
                                "think_length": 4096,
                                "reference_rollout_id": reference_id,
                                "reference_is_correct": reference_correct,
                                "reference_span_id": progress_bin + 1,
                                "reference_progress": 0.25 + 0.5 * progress_bin,
                                "reference_norm": 1.0 + 0.1 * reference_correct,
                                "cosine_similarity": cosine,
                            }
                        )
    return pd.DataFrame(rows)


def test_cli_writes_declared_outputs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "geometry.parquet"
    output_dir = tmp_path / "results"
    _synthetic_geometry().to_parquet(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_correct_trajectory_pairwise_geometry.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap",
            "20",
            "--permutations",
            "5",
            "--seed",
            "11",
        ],
    )

    main()

    expected = [
        "directed_pair_metrics.parquet",
        "undirected_rollout_pairs.parquet",
        "question_pair_type_means.csv",
        "question_contrasts.csv",
        "directed_four_cell_interactions.csv",
        "contrast_summary.csv",
        "permutation_null.parquet",
        "analysis_meta.json",
        "CORRECT_TRAJECTORY_PAIRWISE_GEOMETRY_REPORT.md",
        "figures/G1_direction_angle_progress.png",
        "figures/G2_relative_vector_progress.png",
        "figures/G3_amplitude_progress.png",
        "figures/G4_whole_trajectory_contrasts.png",
    ]
    for relative in expected:
        path = output_dir / relative
        assert path.exists(), relative
        assert path.stat().st_size > 0, relative

    meta = json.loads((output_dir / "analysis_meta.json").read_text(encoding="utf-8"))
    assert meta["questions"] == 3
    assert meta["layers"] == [24, 36]
    assert meta["progress_bins"] == [0, 1]
    assert meta["h200_accessed"] is False
