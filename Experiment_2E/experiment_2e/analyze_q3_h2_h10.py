from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _pass_at_k(path: Path, *, question_limit: int | None = None) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[bool]] = {}
    for row in rows:
        groups.setdefault(str(row["input"]), []).append(bool(row.get("acc", row.get("score", 0) > 0)))
    if question_limit is not None:
        groups = dict(list(groups.items())[:question_limit])
    output: dict[str, float] = {}
    for k in (1, 4, 8):
        estimates = []
        for values in groups.values():
            n = len(values)
            successes = int(sum(values))
            failures = n - successes
            if failures < k:
                estimates.append(1.0)
            else:
                estimates.append(1.0 - math.comb(failures, k) / math.comb(n, k))
        output[f"pass@{k}"] = float(np.mean(estimates)) if estimates else np.nan
    output["n_questions"] = float(len(groups))
    output["mean_response_length"] = float(np.mean([len(row.get("output", "")) for row in rows]))
    return output


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method="spearman"))


def run(args: argparse.Namespace) -> dict[str, object]:
    steps = [int(step) for step in args.steps]
    frames = []
    pass_rows = []
    for step in steps:
        metric_path = args.metrics_dir / f"h2_h10_step{step:03d}.parquet"
        generation_path = args.generation_dir / f"{step}.jsonl"
        if not metric_path.is_file():
            raise FileNotFoundError(metric_path)
        if not generation_path.is_file():
            raise FileNotFoundError(generation_path)
        frame = pd.read_parquet(metric_path)
        frames.append(frame)
        pass_rows.append({"global_step": step, **_pass_at_k(generation_path, question_limit=len(frame["question_id"].unique()))})
    data = pd.concat(frames, ignore_index=True)
    pass_frame = pd.DataFrame(pass_rows).sort_values("global_step")
    n_questions = int(data["question_id"].nunique())

    group_keys = ["global_step", "representation", "stage"]
    scalar_columns = [
        "h2_local",
        "h2_cumulative",
        "delta_h2",
        "forward_ratio",
        "backward_ratio",
        "orthogonal_ratio",
        "h10_p",
        "h10_q",
        "h10_forward_reward",
        "h10_exploration_reward",
        "h10_effective_exploration",
        "response_length",
    ]
    available = [column for column in scalar_columns if column in data.columns]
    summary = (
        data.groupby(group_keys, dropna=False)[available]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    summary.to_csv(args.output_dir / "h2_h10_stage_summary.csv", index=False)
    pass_frame.to_csv(args.output_dir / "pass_at_k.csv", index=False)

    overall = (
        data.groupby(["global_step", "representation"], dropna=False)[available]
        .mean(numeric_only=True)
        .reset_index()
        .merge(pass_frame, on="global_step", how="left", validate="many_to_one")
    )
    overall.to_csv(args.output_dir / "h2_h10_checkpoint_summary.csv", index=False)

    behavior_rows = []
    for (representation, stage), group in data.groupby(["representation", "stage"], dropna=False):
        checkpoint = (
            group.groupby("global_step", as_index=False)[available]
            .mean(numeric_only=True)
            .merge(pass_frame, on="global_step", how="left", validate="one_to_one")
            .sort_values("global_step")
        )
        for metric in ["h2_cumulative", "delta_h2", "h10_p", "h10_forward_reward", "h10_exploration_reward", "h10_effective_exploration"]:
            if metric not in checkpoint:
                continue
            for target in ["pass@1", "pass@4", "pass@8", "mean_response_length"]:
                behavior_rows.append(
                    {
                        "representation": representation,
                        "stage": int(stage),
                        "metric": metric,
                        "target": target,
                        "spearman_checkpoint_corr": _safe_corr(checkpoint[metric], checkpoint[target]),
                        "n_checkpoints": int(checkpoint[metric].notna().sum()),
                    }
                )
    correlations = pd.DataFrame(behavior_rows)
    correlations.to_csv(args.output_dir / "h2_h10_checkpoint_correlations.csv", index=False)

    report = [
        "# Qwen3-1.7B H2/H10 Offline Report",
        "",
        "This report is descriptive. The hidden bonus was not used during training.",
        "",
        f"- checkpoints: {steps}",
        f"- H2/H10 rows: {len(data)}",
        f"- unique rollouts: {data['rollout_id'].nunique()}",
        f"- representations: {sorted(data['representation'].unique().tolist())}",
        "",
        "## Pass@k",
        "",
        pass_frame.to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        "The bonus is evaluated as a behavior preference, not as a correct/wrong classifier.",
        "Checkpoint correlations are descriptive and are not causal evidence.",
    ]
    (args.output_dir / "H2_H10_OFFLINE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    audit = {
        "passed": bool(
            len(data) > 0
            and data["rollout_id"].nunique() == len(steps) * n_questions * 8
            and len(pass_frame) == len(steps)
        ),
        "steps": steps,
        "rows": int(len(data)),
        "n_questions": n_questions,
        "unique_rollouts": int(data["rollout_id"].nunique()),
        "output_files": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    (args.output_dir / "analysis_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise RuntimeError(f"analysis audit failed: {audit}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
