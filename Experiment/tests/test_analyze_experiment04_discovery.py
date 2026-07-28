from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_experiment04_discovery as analysis  # noqa: E402
from analyze_experiment04_discovery import (  # noqa: E402
    canonical_family_table,
    cluster_components,
    label_permutation_cluster_test,
    pairwise_auc,
    question_equal_curve,
    select_cluster_anchor,
    select_primary_clusters,
)


def test_pairwise_auc_handles_ties() -> None:
    assert pairwise_auc(np.asarray([2.0, 1.0]), np.asarray([1.0, 0.0])) == 0.875


def test_question_equal_curve_does_not_weight_extra_rollouts() -> None:
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "is_correct": True, "x": 0, "score": 10.0},
            {"question_id": "q1", "is_correct": False, "x": 0, "score": 0.0},
            {"question_id": "q2", "is_correct": True, "x": 0, "score": 2.0},
            {"question_id": "q2", "is_correct": True, "x": 0, "score": 2.0},
            {"question_id": "q2", "is_correct": True, "x": 0, "score": 2.0},
            {"question_id": "q2", "is_correct": False, "x": 0, "score": 0.0},
        ]
    )
    curve = question_equal_curve(frame, "score", "x", bootstrap=200, seed=7)

    correct = curve.loc[curve["is_correct"], "mean"].item()
    assert np.isclose(correct, 6.0)
    assert curve["questions"].min() == 2


def test_cluster_components_use_four_neighbor_connectivity() -> None:
    statistic = np.asarray(
        [
            [3.0, 3.0, 0.0],
            [0.0, 3.0, -3.0],
            [0.0, 0.0, -3.0],
        ]
    )
    clusters = cluster_components(statistic, threshold=2.0)

    masses = sorted(round(cluster["mass"], 6) for cluster in clusters)
    assert masses == [6.0, 9.0]


def test_label_permutation_cluster_is_deterministic() -> None:
    rows = []
    for question_id in ("q1", "q2", "q3"):
        for rollout_id, label in enumerate((True, True, False, False)):
            for layer in (0, 1):
                for progress_bin in (0, 1):
                    rows.append(
                        {
                            "question_id": question_id,
                            "rollout_id": rollout_id,
                            "is_correct": label,
                            "representation": "mean",
                            "layer": layer,
                            "progress_bin": progress_bin,
                            "score": float(label) + 0.1 * rollout_id,
                        }
                    )
    frame = pd.DataFrame(rows)
    first = label_permutation_cluster_test(
        frame, "score", "mean", "progress_bin", permutations=20, seed=3
    )
    second = label_permutation_cluster_test(
        frame, "score", "mean", "progress_bin", permutations=20, seed=3
    )

    assert np.array_equal(first[1], second[1])
    assert first[2:] == ([0, 1], [0, 1])


def test_token_features_are_tested_once_with_canonical_family() -> None:
    rows = []
    for direction in ("horizontal", "vertical"):
        for representation in ("last", "last25", "mean"):
            rows.append(
                {
                    "direction": direction,
                    "representation": representation,
                    "feature": "token_raw_entropy_mean",
                    "abs_t": 3.0,
                }
            )
    rows.append(
        {
            "direction": "vertical",
            "representation": "last25",
            "feature": "state_angle_demean",
            "abs_t": 4.0,
        }
    )

    families = canonical_family_table(pd.DataFrame(rows))

    token = families[families["feature"] == "token_raw_entropy_mean"].iloc[0]
    assert len(families) == 2
    assert token["direction"] == "horizontal"
    assert token["representation"] == "mean"
    assert token["duplicate_family_count"] == 6
    assert token["hypothesis_key"] == "token::token_raw_entropy_mean"


def test_primary_figures_use_significant_clusters_not_single_cell_rank() -> None:
    clusters = pd.DataFrame(
        [
            {
                "hypothesis_key": "geometry::isolated",
                "p_value": 0.20,
                "mass": 8.0,
                "anchor_abs_t": 8.0,
            },
            {
                "hypothesis_key": "geometry::cluster",
                "p_value": "0.01",
                "mass": 6.0,
                "anchor_abs_t": 3.0,
            },
            {
                "hypothesis_key": "geometry::cluster",
                "p_value": 0.03,
                "mass": 7.0,
                "anchor_abs_t": 4.0,
            },
            {
                "hypothesis_key": "geometry::missing",
                "p_value": None,
                "mass": 100.0,
                "anchor_abs_t": 10.0,
            },
            {
                "hypothesis_key": "geometry::invalid",
                "p_value": "not-tested",
                "mass": 100.0,
                "anchor_abs_t": 10.0,
            },
        ]
    )

    selected = select_primary_clusters(clusters, top_figures=12)

    assert selected["hypothesis_key"].tolist() == ["geometry::cluster"]
    assert selected["p_value"].tolist() == [0.01]


def test_primary_figures_allow_no_supported_significant_clusters() -> None:
    clusters = pd.DataFrame(
        [
            {
                "hypothesis_key": "geometry::nonsignificant",
                "p_value": 0.20,
                "mass": 8.0,
                "primary_supported": True,
            },
            {
                "hypothesis_key": "geometry::unsupported",
                "p_value": 0.01,
                "mass": 6.0,
                "primary_supported": False,
            },
        ]
    )

    selected = select_primary_clusters(clusters, top_figures=12)

    assert selected.empty
    assert {"hypothesis_key", "p_value", "mass"}.issubset(selected.columns)


def test_cluster_anchor_is_selected_only_from_cluster_cells() -> None:
    effects = pd.DataFrame(
        [
            {"layer": 4, "progress_bin": 2, "t_stat": 3.0},
            {"layer": 5, "progress_bin": 2, "t_stat": -4.0},
            {"layer": 20, "progress_bin": 10, "t_stat": 9.0},
        ]
    )

    anchor = select_cluster_anchor(effects, [(4, 2), (5, 2)])

    assert (int(anchor["layer"]), int(anchor["progress_bin"])) == (5, 2)


def test_main_writes_cluster_primary_and_single_cell_appendix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "scalars"
    output_dir = tmp_path / "results"
    input_dir.mkdir()
    metric_rows = []
    summary_rows = []
    summary_features = (
        "step_norm_mean",
        "step_norm_median",
        "path_length",
        "net_displacement",
        "straightness",
        "log_detour",
        "direction_consistency",
        "turn_angle_std",
        "turn_angle_late_std",
    )
    for question_index in range(20):
        for rollout_id, is_correct in enumerate((True, True, False, False)):
            for layer in (0, 1):
                for progress_bin, relative_progress in ((0, 0.1), (1, 0.6)):
                    metric_rows.append(
                        {
                            "question_id": f"q{question_index:02d}",
                            "rollout_id": rollout_id,
                            "is_correct": is_correct,
                            "think_length": 4096,
                            "response_length": 4200,
                            "representation": "mean",
                            "layer": layer,
                            "chunk_id": progress_bin,
                            "chunk_start": progress_bin * 256,
                            "chunk_end": (progress_bin + 1) * 256,
                            "relative_progress": relative_progress,
                            "end_aligned_chunk": False,
                            "is_partial": False,
                            "state_angle_demean": (
                                float(is_correct) * (1.0 + 0.01 * question_index)
                                + 0.001 * rollout_id
                                + 0.01 * layer
                                + 0.02 * progress_bin
                            ),
                        }
                    )
            for direction in ("horizontal", "vertical"):
                summary_row = {
                    "question_id": f"q{question_index:02d}",
                    "rollout_id": rollout_id,
                    "is_correct": is_correct,
                    "direction": direction,
                    "representation": "mean",
                    "layer": 0,
                    "chunk_id": 0,
                    "is_partial": False,
                }
                for feature in summary_features:
                    summary_row[feature] = float(is_correct) + 0.01 * question_index
                summary_rows.append(summary_row)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(input_dir / "horizontal_metrics.parquet", index=False)
    metrics.to_parquet(input_dir / "vertical_metrics.parquet", index=False)
    pd.DataFrame(summary_rows).to_parquet(
        input_dir / "summary_metrics.parquet", index=False
    )

    monkeypatch.setattr(
        analysis,
        "parse_args",
        lambda: SimpleNamespace(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            bootstrap=2,
            permutations=2,
            progress_bins=2,
            seed=7,
            top_figures=2,
            cluster_top_families=0,
        ),
    )
    monkeypatch.setattr(
        analysis,
        "label_permutation_cluster_test",
        lambda *args, **kwargs: (
            [{"sign": 1, "mass": 6.0, "p_value": "0.01", "cells": [(0, 0), (0, 1)]}],
            np.zeros(2),
            [0, 1],
            [0, 1],
        ),
    )

    analysis.main()

    primary = sorted((output_dir / "figures").glob("F*.png"))
    appendix = sorted(
        (output_dir / "figures" / "exploratory_single_cell").glob("A*.png")
    )
    assert len(primary) == 4
    assert len(appendix) == 4
    assert not list((output_dir / "figures").glob("A*.png"))
    assert all(path.stat().st_size > 0 for path in primary + appendix)

    report = (output_dir / "DISCOVERY_REPORT.md").read_text(encoding="utf-8")
    assert "## Primary Cluster-Significant Findings" in report
    assert "cluster p=0.01" in report
    assert "## Exploratory Single-Cell Appendix" in report
    assert "Uncorrected maximum-|t| ranking" in report

    clusters = pd.read_csv(output_dir / "cluster_permutation.csv")
    assert clusters["p_value"].le(0.05).all()
    assert set(clusters["cells"]) == {"[[0, 0], [0, 1]]"}
