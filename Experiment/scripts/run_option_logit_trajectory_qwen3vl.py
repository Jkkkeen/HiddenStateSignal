#!/usr/bin/env python3
"""Run fixed-grid option-logit trajectory probes for Qwen3-VL rollouts.

The expensive stage stores raw A/B/C/D logits for each rollout prefix. The
cheap stage derives margin/gain features and within-question AUROC from those
saved logits.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from response_segment_steps import answer_after_think
from run_long_path_smoke_qwen3vl import (
    build_prompt_messages,
    encode_messages,
    load_jsonl,
    select_rows,
    to_device,
)
from run_option_logit_probe_debug_qwen3vl import (
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    DEFAULT_PROBE_SUFFIX,
    build_probe_messages,
    label_token_ids,
    logit_metrics,
    parse_csv_floats,
    parse_labels,
    probe_char_positions,
)
from run_path_dynamics import aucs_by_question, bootstrap_ci
from run_semantic_step_basin_qwen3vl import normalize_label, think_text_and_char_span


ID_COLUMNS = {
    "question_id",
    "rollout_id",
    "source_line",
    "is_correct",
    "answer",
    "pred_answer",
}

PRIMARY_FEATURES = {
    "think_early_to_final_gain_max": "pos",
    "think_early_to_late_gain_max": "pos",
    "think_late_margin_drop_max": "neg",
    "think_relative_final_margin_max": "pos",
    "think_relative_late_margin_max": "pos",
    "think_entropy_delta": "neg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run option-logit trajectory probes.")
    parser.add_argument("--input", required=True, help="Labeled long rollout JSONL.")
    parser.add_argument("--output-dir", default="option_logit_trajectory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--option-labels", default=DEFAULT_LABELS)
    parser.add_argument(
        "--think-fracs",
        default="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,1.00",
    )
    parser.add_argument("--answer-fracs", default="0.00,0.25,0.50,0.75")
    parser.add_argument("--limit-rollouts", type=int, default=-1)
    parser.add_argument("--clean-only", action="store_true", default=True)
    parser.add_argument("--probe-suffix", default=DEFAULT_PROBE_SUFFIX)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def entropy_from_logit_columns(group: pd.DataFrame, labels: list[str]) -> pd.Series:
    cols = [f"logit_{label}" for label in labels]
    values = group[cols].to_numpy(dtype=np.float64)
    values = values - np.nanmax(values, axis=1, keepdims=True)
    probs = np.exp(values)
    probs = probs / np.nansum(probs, axis=1, keepdims=True)
    entropy = -np.nansum(probs * np.log(probs + 1e-12), axis=1)
    return pd.Series(entropy, index=group.index)


def add_prompt_probe(rows: list[dict[str, Any]], row: dict[str, Any], labels: list[str]) -> None:
    # Marker row used only when loading precomputed probes without a model. The
    # live forward path creates prompt probes through the same model call as all
    # other probes, so this helper intentionally remains unused there.
    raise NotImplementedError


def first_or_nan(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.iloc[0]) if not values.empty else float("nan")


def last_or_nan(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def mean_or_nan(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.mean()) if not values.empty else float("nan")


def ci_or_nan(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if n_boot <= 0:
        return float("nan"), float("nan")
    return bootstrap_ci(values, n_boot, seed)


def summarize_margin(group: pd.DataFrame, col: str, suffix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    prompt = group[group["segment"] == "prompt"].sort_values("frac")
    think = group[group["segment"] == "think"].sort_values("frac")
    answer = group[group["segment"] == "answer"].sort_values("frac")

    prompt_only = first_or_nan(prompt[col])
    early = first_or_nan(think[col])
    final = last_or_nan(think[col])
    late = think[think["frac"] >= 0.75]
    late_mean = mean_or_nan(late[col])
    early_mean = mean_or_nan(think[think["frac"] <= 0.25][col])
    answer_early = first_or_nan(answer[col])
    answer_late = last_or_nan(answer[col])

    values = think[col].dropna()
    diffs = values.diff().dropna()

    out[f"prompt_only_margin_{suffix}"] = prompt_only
    out[f"think_early_margin_{suffix}"] = early
    out[f"think_early_margin_mean_{suffix}"] = early_mean
    out[f"think_final_margin_{suffix}"] = final
    out[f"think_late_margin_mean_{suffix}"] = late_mean
    out[f"think_early_to_final_gain_{suffix}"] = final - early
    out[f"think_early_to_late_gain_{suffix}"] = late_mean - early
    out[f"think_late_margin_drop_{suffix}"] = early - late_mean
    out[f"think_relative_final_margin_{suffix}"] = final - prompt_only
    out[f"think_relative_late_margin_{suffix}"] = late_mean - prompt_only
    out[f"think_mean_step_gain_{suffix}"] = mean_or_nan(diffs)
    out[f"think_positive_gain_rate_{suffix}"] = float((diffs > 0).mean()) if not diffs.empty else float("nan")
    out[f"think_max_margin_{suffix}"] = float(values.max()) if not values.empty else float("nan")
    out[f"think_min_margin_{suffix}"] = float(values.min()) if not values.empty else float("nan")
    out[f"think_margin_std_{suffix}"] = float(values.std(ddof=0)) if values.size > 1 else float("nan")
    out[f"think_late_flip_rate_{suffix}"] = float((late[col] < 0).mean()) if not late.empty else float("nan")
    out[f"answer_early_margin_{suffix}"] = answer_early
    out[f"answer_late_margin_{suffix}"] = answer_late
    out[f"answer_gain_{suffix}"] = answer_late - answer_early
    return out


def build_rollout_features(probes: pd.DataFrame) -> pd.DataFrame:
    if probes.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    keys = ["question_id", "rollout_id"]
    for (qid, rid), group in probes.groupby(keys, sort=False):
        group = group.copy()
        if "probe_idx" not in group.columns:
            group["probe_idx"] = np.arange(len(group))
        group = group.sort_values(["segment", "frac", "probe_idx"], kind="stable")
        first = group.iloc[0]
        row: dict[str, Any] = {
            "question_id": qid,
            "rollout_id": int(rid),
            "source_line": int(first.get("source_line", -1)),
            "is_correct": bool(first.get("is_correct", False)),
            "answer": first.get("answer"),
            "pred_answer": first.get("pred_answer"),
        }
        for margin_col, suffix in [
            ("correct_margin_max", "max"),
            ("correct_margin_mean", "mean"),
        ]:
            if margin_col in group.columns:
                row.update(summarize_margin(group, margin_col, suffix))

        think = group[group["segment"] == "think"].sort_values("frac")
        if not think.empty:
            entropy = think["abcd_entropy"].dropna()
            row["think_early_entropy"] = first_or_nan(think["abcd_entropy"])
            row["think_final_entropy"] = last_or_nan(think["abcd_entropy"])
            row["think_entropy_delta"] = row["think_final_entropy"] - row["think_early_entropy"]
            row["think_entropy_mean"] = float(entropy.mean()) if not entropy.empty else float("nan")
            tops = think["top_option"].astype(str).to_list()
            row["think_top_option_switch_count"] = int(sum(a != b for a, b in zip(tops, tops[1:])))
            row["think_is_top_correct_rate"] = float(think["is_top_correct"].mean()) if "is_top_correct" in think else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_option_logit_features(features: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    feature_cols = [
        col
        for col in features.columns
        if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(features[col])
    ]
    rows = []
    for feature in feature_cols:
        aucs = aucs_by_question(features, feature)
        if aucs.size == 0:
            continue
        neg = 1.0 - aucs
        low, high = ci_or_nan(aucs, n_boot, seed)
        nlow, nhigh = ci_or_nan(neg, n_boot, seed + 17)
        direction = PRIMARY_FEATURES.get(feature, "")
        committed = float(np.mean(neg)) if direction == "neg" else float(np.mean(aucs))
        rows.append(
            {
                "feature": feature,
                "is_primary": feature in PRIMARY_FEATURES,
                "direction": direction,
                "n_questions": int(aucs.size),
                "mean_auc_pos": float(np.mean(aucs)),
                "ci_low_pos": low,
                "ci_high_pos": high,
                "mean_auc_neg": float(np.mean(neg)),
                "ci_low_neg": nlow,
                "ci_high_neg": nhigh,
                "committed_auc": committed,
                "best_auc": float(max(np.mean(aucs), np.mean(neg))),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["is_primary", "best_auc"], ascending=[False, False]).reset_index(drop=True)
    return out


def plot_top_auc(eval_df: pd.DataFrame, output_path: Path, top_n: int = 16) -> None:
    import matplotlib.pyplot as plt

    if eval_df.empty:
        return
    view = eval_df.sort_values("best_auc", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(view))))
    ax.barh(view["feature"], view["best_auc"], color="#3A78C2")
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("within-question AUROC (best direction)")
    ax.set_title("Option-logit trajectory features")
    ax.set_xlim(0.45, max(0.75, float(view["best_auc"].max()) + 0.03))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_margin_curve(probes: pd.DataFrame, output_path: Path, margin_col: str) -> None:
    import matplotlib.pyplot as plt

    think = probes[probes["segment"] == "think"].copy()
    if think.empty or margin_col not in think:
        return
    grouped = think.groupby(["frac", "is_correct"])[margin_col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for is_correct, sub in grouped.groupby("is_correct"):
        label = "correct" if bool(is_correct) else "incorrect"
        ax.plot(sub["frac"], sub[margin_col], marker="o", linewidth=2, label=label)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("thinking progress")
    ax.set_ylabel(margin_col)
    ax.set_title(f"Mean {margin_col} over thinking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    probes: pd.DataFrame,
    features: pd.DataFrame,
    eval_df: pd.DataFrame,
    skipped: list[dict[str, Any]],
    figures: list[Path],
) -> Path:
    report = output_dir / "OPTION_LOGIT_TRAJECTORY_RESULTS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Option Logit Trajectory Results\n\n")
        f.write("This experiment stores A/B/C/D logits at fixed reasoning prefixes and evaluates derived margin trajectories.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Probe rows: {len(probes)}\n")
        f.write(f"- Rollout feature rows: {len(features)}\n")
        f.write(f"- Questions: {features['question_id'].nunique() if not features.empty else 0}\n")
        f.write(f"- Rollouts: {len(features)}\n")
        f.write(f"- Skipped rollouts: {len(skipped)}\n")
        if not probes.empty:
            top_counts = probes["top_option"].value_counts().to_dict()
            f.write(f"- Top option counts: {top_counts}\n")
        f.write("\n## Primary AUROC\n\n")
        f.write("| feature | direction | questions | AUROC(+feature) | 95% CI | AUROC(-feature) | 95% CI | best |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        if not eval_df.empty:
            top = eval_df.sort_values(["is_primary", "best_auc"], ascending=[False, False]).head(24)
            for _, row in top.iterrows():
                f.write(
                    f"| {row['feature']} | {row['direction']} | {int(row['n_questions'])} | "
                    f"{row['mean_auc_pos']:.4f} | [{row['ci_low_pos']:.4f}, {row['ci_high_pos']:.4f}] | "
                    f"{row['mean_auc_neg']:.4f} | [{row['ci_low_neg']:.4f}, {row['ci_high_neg']:.4f}] | "
                    f"{row['best_auc']:.4f} |\n"
                )
        f.write("\n## Figures\n\n")
        for fig in figures:
            f.write(f"- `figures/{fig.name}`\n")
        f.write("\n## Interpretation Notes\n\n")
        f.write("- `top_option` is diagnostic only because raw label logits can have strong option priors.\n")
        f.write("- Main signals are continuous correct-option margin gain/drop over thinking.\n")
        f.write("- `prompt_only_margin` controls for questions where the prompt already makes the answer easy.\n")
    return report


def forward_probe_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    labels = parse_labels(args.option_labels)
    think_fracs = parse_csv_floats(args.think_fracs)
    answer_fracs = parse_csv_floats(args.answer_fracs)

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_jsonl(Path(args.input)), args.clean_only, args.limit_rollouts)
    if not rows:
        raise ValueError("No rows selected.")

    print("Option logit trajectory")
    print(f"input: {args.input}")
    print(f"selected rollouts: {len(rows)}")
    print(f"output_dir: {args.output_dir}")
    print(f"model: {args.model}")
    print(f"labels: {labels}")
    print(f"think_fracs: {think_fracs}")
    print(f"answer_fracs: {answer_fracs}")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    tokenizer = getattr(processor, "tokenizer", processor)
    label_ids = label_token_ids(tokenizer, labels)
    print(f"label_token_ids: {label_ids}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    model.eval()
    device = next(model.parameters()).device

    out_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="option-logit trajectory"):
        qid = str(row.get("question_id", ""))
        rid = int(row.get("rollout_id", -1))
        correct = normalize_label(row.get("answer"))
        chosen = normalize_label(row.get("pred_answer"))
        try:
            response = str(row.get("response", ""))
            think_text, _, _, think_status = think_text_and_char_span(response)
            answer_text, _, _, answer_status = answer_after_think(response)
            probes = probe_char_positions(think_text, answer_text, think_fracs, answer_fracs)
            prompt_probe = [("prompt", 0.0, 0, "")]
            probe_items = prompt_probe + [
                (probe.segment, probe.frac, probe.char_end, probe.text_prefix) for probe in probes
            ]
            for probe_idx, (segment, frac, char_end, prefix_text) in enumerate(probe_items):
                messages = build_probe_messages(row, prefix_text, args.probe_suffix)
                batch = encode_messages(processor, messages, add_generation_prompt=False)
                batch = to_device(batch, device)
                with torch.inference_mode():
                    outputs = model(**batch, use_cache=False)
                last_logits = outputs.logits[0, -1].float().detach().cpu().numpy()
                logits = {label: float(last_logits[token_id]) for label, token_id in label_ids.items()}
                metrics = logit_metrics(logits, labels, correct, chosen)
                out_rows.append(
                    {
                        "question_id": qid,
                        "rollout_id": rid,
                        "source_line": int(row.get("_source_line", row.get("source_line", -1))),
                        "is_correct": bool(row.get("is_correct", False)),
                        "answer": correct,
                        "pred_answer": chosen,
                        "probe_idx": int(probe_idx),
                        "segment": segment,
                        "frac": float(frac),
                        "char_end": int(char_end),
                        "think_chars": int(len(think_text)),
                        "answer_chars": int(len(answer_text)),
                        "think_status": think_status,
                        "answer_status": answer_status,
                        **metrics,
                    }
                )
                del outputs, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as exc:
            skipped.append({"question_id": qid, "rollout_id": rid, "reason": str(exc)})
            print(f"[skip] question={qid} rollout={rid} reason={exc}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(out_rows), skipped, label_ids


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    probes, skipped, label_ids = forward_probe_rows(args)
    features = build_rollout_features(probes)
    eval_df = evaluate_option_logit_features(features, args.bootstrap, args.seed)

    probes_path = output_dir / "option_logit_probes.parquet"
    features_path = output_dir / "option_logit_features.parquet"
    eval_path = output_dir / "option_logit_eval.csv"
    skipped_path = output_dir / "option_logit_skipped.jsonl"

    probes.to_parquet(probes_path, index=False)
    features.to_parquet(features_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    with skipped_path.open("w", encoding="utf-8") as f:
        for item in skipped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    figures: list[Path] = []
    fig = figures_dir / "O1_option_logit_top_auc.png"
    plot_top_auc(eval_df, fig)
    if fig.exists():
        figures.append(fig)
    fig = figures_dir / "O2_correct_margin_max_curve.png"
    plot_margin_curve(probes, fig, "correct_margin_max")
    if fig.exists():
        figures.append(fig)
    fig = figures_dir / "O3_correct_margin_mean_curve.png"
    plot_margin_curve(probes, fig, "correct_margin_mean")
    if fig.exists():
        figures.append(fig)

    report = write_report(output_dir, probes, features, eval_df, skipped, figures)

    print(f"saved probes: {probes_path} ({len(probes)} rows)")
    print(f"saved features: {features_path} ({len(features)} rows)")
    print(f"saved eval: {eval_path} ({len(eval_df)} rows)")
    print(f"saved skipped: {skipped_path} ({len(skipped)} rows)")
    print(f"saved report: {report}")
    print(f"label token ids: {label_ids}")
    if not eval_df.empty:
        print(eval_df.sort_values("best_auc", ascending=False).head(20))


if __name__ == "__main__":
    main()
