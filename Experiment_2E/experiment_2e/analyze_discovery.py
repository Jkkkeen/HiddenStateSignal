from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import (
    TEST_KEYS,
    apply_base_standardizers,
    benjamini_hochberg,
    fit_base_standardizers,
    fit_training_stage_interaction,
    grouped_oof_length_increment,
    select_primary_metrics,
    spearman_by_stage,
    summarize_outcome_auc,
    summarize_question_equal,
)
from .common import sha256_file, write_json_atomic
from .formal_manifest import CHECKPOINT_PROGRESS


def load_metric_outputs(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_paths = sorted(metrics_dir.glob("horizontal_metrics_*.parquet")) + sorted(
        metrics_dir.glob("vertical_metrics_*.parquet")
    )
    control_paths = sorted(metrics_dir.glob("controls_*.parquet"))
    if not metric_paths or not control_paths:
        raise FileNotFoundError("checkpoint metric and control parquet files are required")
    metrics = pd.concat((pd.read_parquet(path) for path in metric_paths), ignore_index=True)
    controls = pd.concat((pd.read_parquet(path) for path in control_paths), ignore_index=True)
    return metrics, controls


def primary_selection_audit(primary: pd.DataFrame, *, allow_incomplete: bool) -> dict[str, Any]:
    tests = primary[TEST_KEYS].drop_duplicates()
    audit = {
        "n_primary_tests": int(len(tests)),
        "expected_primary_tests": 37,
        "n_rows": int(len(primary)),
        "n_rollouts": int(primary["rollout_id"].nunique()),
        "n_questions": int(primary["question_id"].nunique()),
        "checkpoints": sorted(primary["checkpoint"].unique().tolist()),
        "families": sorted(primary["family_id"].unique().tolist()),
    }
    if not allow_incomplete:
        expected_checkpoints = set(CHECKPOINT_PROGRESS)
        if set(audit["checkpoints"]) != expected_checkpoints:
            raise ValueError(f"formal analysis requires checkpoints={sorted(expected_checkpoints)}")
        if audit["n_primary_tests"] != 37:
            raise ValueError(f"formal analysis requires 37 primary tests, found {audit['n_primary_tests']}")
    return audit


def _test_label(values: dict[str, Any]) -> str:
    anchor = str(values["anchor_layer"]).replace(".", "p")
    return "_".join(
        str(values[key]).replace("/", "-").replace(" ", "-")
        for key in ("family_id", "representation")
    ) + f"_L{anchor}_{values['aggregation_mode']}"


def _plot_test(summary: pd.DataFrame, values: dict[str, Any], output_dir: Path) -> list[str]:
    selection = np.ones(len(summary), dtype=bool)
    for key, value in values.items():
        selection &= summary[key].astype(str).to_numpy() == str(value)
    subset = summary.loc[selection].sort_values(["stage", "training_progress"])
    if subset.empty:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    label = _test_label(values)
    paths = []

    pivot = subset.pivot(index="stage", columns="checkpoint", values="question_equal_mean")
    checkpoint_order = [name for name in CHECKPOINT_PROGRESS if name in pivot.columns]
    pivot = pivot.reindex(index=range(4), columns=checkpoint_order)
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    axis.set_xticks(range(len(checkpoint_order)), checkpoint_order)
    axis.set_yticks(range(4), ["B1", "B2", "B3", "B4"])
    axis.set_xlabel("Training checkpoint")
    axis.set_ylabel("Response stage")
    axis.set_title(f"{values['family_id']} base-standardized trajectory")
    figure.colorbar(image, ax=axis, label="Question-equal mean z")
    figure.tight_layout()
    heatmap = output_dir / f"{label}_heatmap.png"
    figure.savefig(heatmap, dpi=160)
    plt.close(figure)
    paths.append(heatmap.name)

    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for stage, group in subset.groupby("stage"):
        group = group.sort_values("training_progress")
        x = group["training_progress"].to_numpy(dtype=float)
        y = group["question_equal_mean"].to_numpy(dtype=float)
        low = group["ci_low"].to_numpy(dtype=float)
        high = group["ci_high"].to_numpy(dtype=float)
        axis.plot(x, y, marker="o", label=f"B{int(stage) + 1}")
        axis.fill_between(x, low, high, alpha=0.15)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Training progress")
    axis.set_ylabel("Question-equal mean z")
    axis.set_title(f"{values['family_id']} response-stage curves")
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    curves = output_dir / f"{label}_stage_curves.png"
    figure.savefig(curves, dpi=160)
    plt.close(figure)
    paths.append(curves.name)
    return paths


def _write_report(output_dir: Path, audit: dict[str, Any], figure_names: list[str]) -> None:
    links = "\n".join(
        f'<li><a href="figures/primary/{html.escape(name)}">{html.escape(name)}</a></li>'
        for name in figure_names
    )
    body = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Experiment 2E discovery report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;line-height:1.5}}
code,pre{{background:#f3f4f6;padding:3px 6px}} table{{border-collapse:collapse}} td,th{{padding:6px;border:1px solid #ddd}}</style>
<body><h1>Experiment 2E discovery report</h1>
<p>This report is generated from frozen scalar outputs. It does not select a reward candidate automatically.</p>
<h2>Audit</h2><pre>{html.escape(json.dumps(audit, indent=2, ensure_ascii=False))}</pre>
<h2>Primary figures</h2><ul>{links}</ul>
<h2>Machine-readable results</h2>
<ul><li>primary_effects.csv</li><li>outcome_effects.csv</li><li>oof_length_entropy_increment.csv</li>
<li>primary_summary.csv</li><li>coverage.csv</li><li>spearman_by_stage.csv</li></ul></body></html>"""
    (output_dir / "REPORT.html").write_text(body, encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    metrics, controls = load_metric_outputs(args.metrics_dir)
    primary = select_primary_metrics(metrics)
    selection_audit = primary_selection_audit(primary, allow_incomplete=args.allow_incomplete)
    control_audit = {
        "n_control_rows": int(len(controls)),
        "n_control_rollouts": int(controls["rollout_id"].nunique()),
        "control_checkpoints": sorted(controls["checkpoint"].unique().tolist()),
    }
    if not args.allow_incomplete:
        if control_audit["n_control_rollouts"] != 12288:
            raise ValueError(
                "formal analysis requires controls for 12,288 unique rollouts, "
                f"found {control_audit['n_control_rollouts']}"
            )
        if control_audit["n_control_rows"] != 4 * 12288:
            raise ValueError("formal analysis requires four stage-control rows per rollout")
    calibrators = fit_base_standardizers(primary)
    standardized = apply_base_standardizers(primary, calibrators)
    analysis_dir = args.output_dir / "analysis"
    figures_dir = args.output_dir / "figures" / "primary"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    calibrators.to_csv(analysis_dir / "base_scalar_calibrators.csv", index=False)
    standardized.to_parquet(analysis_dir / "primary_standardized.parquet", index=False)

    summary = summarize_question_equal(
        standardized.loc[standardized["standardization_ok"]],
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    summary.to_csv(analysis_dir / "primary_summary.csv", index=False)
    spearman = spearman_by_stage(standardized.loc[standardized["standardization_ok"]])
    spearman.to_csv(analysis_dir / "spearman_by_stage.csv", index=False)

    model_rows = []
    if not args.skip_models:
        for keys, group in standardized.groupby(TEST_KEYS, dropna=False, sort=False):
            result = fit_training_stage_interaction(group.loc[group["standardization_ok"]])
            model_rows.append({**dict(zip(TEST_KEYS, keys, strict=True)), **result})
    effects = pd.DataFrame(model_rows)
    if not effects.empty:
        effects["bh_q"] = benjamini_hochberg(effects["interaction_p"])
        effects["passes_q_0_10"] = effects["bh_q"] <= 0.10
    effects.to_csv(analysis_dir / "primary_effects.csv", index=False)

    outcomes = summarize_outcome_auc(
        standardized.loc[standardized["standardization_ok"]],
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    if not outcomes.empty:
        outcomes["bh_q"] = benjamini_hochberg(outcomes["question_equal_p"])
        outcomes["passes_q_0_10"] = outcomes["bh_q"] <= 0.10
    outcomes.to_csv(analysis_dir / "outcome_effects.csv", index=False)

    oof_rows = []
    if not args.skip_oof:
        oof_keys = TEST_KEYS + ["checkpoint", "training_progress", "stage"]
        for keys, group in standardized.groupby(oof_keys, dropna=False, sort=False):
            result = grouped_oof_length_increment(group.loc[group["standardization_ok"]], seed=args.seed)
            oof_rows.append({**dict(zip(oof_keys, keys, strict=True)), **result})
    pd.DataFrame(oof_rows).to_csv(analysis_dir / "oof_length_entropy_increment.csv", index=False)

    coverage_keys = ["axis", "family_id", "representation", "anchor_layer", "stage", "checkpoint"]
    coverage = (
        metrics.groupby(coverage_keys, dropna=False)
        .agg(
            n_rows=("value", "size"),
            n_covered=("coverage", "sum"),
            n_rollouts=("rollout_id", "nunique"),
            n_questions=("question_id", "nunique"),
        )
        .reset_index()
    )
    coverage["coverage_rate"] = coverage["n_covered"] / coverage["n_rows"]
    coverage.to_csv(analysis_dir / "coverage.csv", index=False)

    figure_names = []
    for _, row in primary[TEST_KEYS].drop_duplicates().iterrows():
        figure_names.extend(_plot_test(summary, row.to_dict(), figures_dir))
    audit = {
        **selection_audit,
        **control_audit,
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "models_skipped": args.skip_models,
        "oof_skipped": args.skip_oof,
        "n_figures": len(figure_names),
        "input_sha256": {path.name: sha256_file(path) for path in sorted(args.metrics_dir.glob("*.parquet"))},
    }
    write_json_atomic(args.output_dir / "AUDIT.json", audit)
    write_json_atomic(
        analysis_dir / "candidate_decisions.json",
        {
            "status": "requires_scientific_review",
            "automatic_reward_promotion": False,
            "rule": "Apply the eight preregistered criteria in RL_discovery_plan.md section 11.",
        },
    )
    _write_report(args.output_dir, audit, figure_names)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Experiment 2E discovery analysis.")
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-oof", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
