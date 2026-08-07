from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .analysis import TEST_KEYS


class SwanLabClient(Protocol):
    def log(self, data: dict[str, object], step: int) -> None: ...

    def Image(self, path: str) -> object: ...


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fragment(value: object) -> str:
    text = str(value).replace("/", "-").replace(" ", "-")
    return text.replace(".", "p")


def _checkpoint_steps(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["checkpoint"]): int(row["global_step"]) for row in payload["checkpoints"]}


def _metric_prefix(row: pd.Series) -> str:
    stage = int(row["stage"]) + 1
    return "/".join(
        (
            "hidden",
            _fragment(row["family_id"]),
            _fragment(row["representation"]),
            f"L{_fragment(row['anchor_layer'])}",
            _fragment(row["metric"]),
            f"B{stage}",
        )
    )


def build_step_payloads(
    standardized_path: Path,
    outcome_path: Path,
    checkpoint_manifest: Path,
) -> dict[int, dict[str, float]]:
    """Build per-checkpoint SwanLab scalars from frozen analysis outputs."""
    standardized = pd.read_parquet(standardized_path)
    outcomes = pd.read_csv(outcome_path)
    required = set(TEST_KEYS) | {
        "checkpoint",
        "training_progress",
        "stage",
        "is_correct",
        "z_value",
        "standardization_ok",
    }
    missing = required - set(standardized.columns)
    if missing:
        raise ValueError(f"primary_standardized is missing columns: {sorted(missing)}")
    standardized = standardized.loc[
        standardized["standardization_ok"].astype(bool)
        & pd.to_numeric(standardized["z_value"], errors="coerce").notna()
    ].copy()
    group_keys = TEST_KEYS + ["checkpoint", "training_progress", "stage"]
    if standardized.empty:
        return {}
    grouped = (
        standardized.groupby(group_keys + ["is_correct"], dropna=False)["z_value"]
        .agg(["mean", "size"])
        .unstack("is_correct")
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    grouped = grouped.rename(
        columns={
            "mean_False": "wrong_mean",
            "mean_True": "correct_mean",
            "size_False": "n_wrong",
            "size_True": "n_correct",
        }
    )
    for column in ("wrong_mean", "correct_mean", "n_wrong", "n_correct"):
        if column not in grouped:
            grouped[column] = math.nan
    grouped["gap"] = grouped["correct_mean"] - grouped["wrong_mean"]

    outcome_columns = group_keys + [
        "question_equal_auc",
        "pair_weighted_auc",
        "n_mixed_questions",
        "n_pairs",
    ]
    available = [column for column in outcome_columns if column in outcomes.columns]
    if set(group_keys).issubset(outcomes.columns):
        grouped = grouped.merge(
            outcomes[available], on=group_keys, how="left", validate="one_to_one"
        )
    steps = _checkpoint_steps(checkpoint_manifest)
    payloads: dict[int, dict[str, float]] = defaultdict(dict)
    for _, row in grouped.iterrows():
        checkpoint = str(row["checkpoint"])
        if checkpoint not in steps:
            raise ValueError(f"checkpoint {checkpoint!r} is absent from {checkpoint_manifest}")
        prefix = _metric_prefix(row)
        for source_name, log_name in (
            ("correct_mean", "correct_mean"),
            ("wrong_mean", "wrong_mean"),
            ("gap", "gap"),
            ("n_correct", "n_correct"),
            ("n_wrong", "n_wrong"),
            ("question_equal_auc", "question_equal_auroc"),
            ("pair_weighted_auc", "pair_weighted_auroc"),
            ("n_mixed_questions", "n_mixed_questions"),
            ("n_pairs", "n_pairs"),
        ):
            value = _finite(row.get(source_name))
            if value is not None:
                payloads[steps[checkpoint]][f"{prefix}/{log_name}"] = value
    return dict(sorted(payloads.items()))


def upload_analysis_bundle(
    *,
    client: SwanLabClient,
    analysis_dir: Path,
    figures_dir: Path,
    checkpoint_manifest: Path,
) -> dict[str, int]:
    payloads = build_step_payloads(
        analysis_dir / "primary_standardized.parquet",
        analysis_dir / "outcome_effects.csv",
        checkpoint_manifest,
    )
    for step, payload in payloads.items():
        client.log(payload, step=step)
    figure_paths = sorted(figures_dir.glob("*.png"))
    final_step = max(payloads, default=max(_checkpoint_steps(checkpoint_manifest).values()))
    for figure in figure_paths:
        client.log({f"figures/primary/{figure.stem}": client.Image(str(figure))}, step=final_step)
    return {
        "n_steps": len(payloads),
        "n_scalar_payloads": len(payloads),
        "n_figures": len(figure_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload frozen Experiment 2E hidden analysis to SwanLab.")
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-id", default=os.environ.get("SWANLAB_RUN_ID"))
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_id:
        parser.error("--run-id or SWANLAB_RUN_ID is required to resume the training run")
    api_key = os.environ.get("SWANLAB_API_KEY")
    if not api_key:
        parser.error("SWANLAB_API_KEY must be set by the external credential file")

    import swanlab

    swanlab.login(api_key)
    swanlab.init(
        project=args.project,
        name=args.experiment_name,
        id=args.run_id,
        resume="allow",
        config={"FRAMEWORK": "Experiment_2E", "kind": "frozen_hidden_analysis"},
        log_dir=str(args.log_dir),
        mode="online",
    )
    try:
        result = upload_analysis_bundle(
            client=swanlab,
            analysis_dir=args.analysis_dir,
            figures_dir=args.figures_dir,
            checkpoint_manifest=args.checkpoint_manifest,
        )
    finally:
        swanlab.finish()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
