#!/usr/bin/env python3
"""Probe option logits after trimming the final conclusion from thinking text."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_long_path_smoke_qwen3vl import encode_messages, load_jsonl, select_rows, to_device
from run_option_logit_probe_debug_qwen3vl import (
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    DEFAULT_PROBE_SUFFIX,
    build_probe_messages,
    label_token_ids,
    logit_metrics,
    parse_labels,
)
from run_option_logit_trajectory_qwen3vl import ci_or_nan
from run_path_dynamics import aucs_by_question
from run_semantic_step_basin_qwen3vl import normalize_label, think_text_and_char_span


TRIGGER_RE = re.compile(
    r"(?i)\b("
    r"therefore|thus|hence|so[, ]+the answer|so[, ]+answer|"
    r"final answer|answer is|the answer is|correct answer|option [ABCD]|"
    r"answer should be|answer would be|answer must be"
    r")\b"
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
ID_COLUMNS = {"question_id", "rollout_id", "source_line", "is_correct", "answer", "pred_answer", "trim_status"}
PRIMARY_FEATURES = {
    "trimmed_margin_max": "pos",
    "trimmed_margin_mean": "pos",
    "trimmed_relative_margin_max": "pos",
    "trimmed_relative_margin_mean": "pos",
    "original_minus_trimmed_final_margin_max": "neg",
    "original_minus_trimmed_final_margin_mean": "neg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trimmed-conclusion option-logit probe.")
    parser.add_argument("--input", required=True, help="Labeled long rollout JSONL.")
    parser.add_argument("--output-dir", default="option_logit_trimmed_conclusion")
    parser.add_argument("--original-features", default="", help="Existing option_logit_features.parquet.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--option-labels", default=DEFAULT_LABELS)
    parser.add_argument("--limit-rollouts", type=int, default=-1)
    parser.add_argument("--clean-only", action="store_true", default=True)
    parser.add_argument("--probe-suffix", default=DEFAULT_PROBE_SUFFIX)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-keep-chars", type=int, default=40)
    parser.add_argument("--tail-window-chars", type=int, default=1800)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sentence_spans(text: str) -> list[tuple[int, int]]:
    text = str(text or "")
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def previous_sentence_start(text: str, pos: int) -> int:
    spans = sentence_spans(text)
    for start, end in spans:
        if start <= pos < end:
            return start
    starts = [start for start, _ in spans if start < pos]
    return starts[-1] if starts else 0


def trim_final_conclusion(
    text: str,
    min_keep_chars: int = 40,
    tail_window_chars: int = 1800,
) -> tuple[str, str]:
    text = str(text or "").strip()
    if len(text) <= min_keep_chars:
        return text, "too_short"

    tail_start = max(0, len(text) - tail_window_chars)
    tail = text[tail_start:]
    matches = list(TRIGGER_RE.finditer(tail))
    if matches:
        trigger_pos = tail_start + matches[-1].start()
        cut = previous_sentence_start(text, trigger_pos)
        if cut >= min_keep_chars:
            return text[:cut].strip(), "trigger"

    spans = sentence_spans(text)
    if len(spans) >= 2:
        cut = spans[-1][0]
        if cut >= min_keep_chars:
            return text[:cut].strip(), "last_sentence"
    return text, "no_trim"


def evaluate_trimmed_features(features: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
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


def build_trimmed_features(trimmed: pd.DataFrame, original: pd.DataFrame | None = None) -> pd.DataFrame:
    if trimmed.empty:
        return pd.DataFrame()
    cols = ["question_id", "rollout_id", "source_line", "is_correct", "answer", "pred_answer", "trim_status"]
    keep = [col for col in cols if col in trimmed.columns]
    features = trimmed[keep].copy()
    features["question_id"] = features["question_id"].astype(str)
    features["rollout_id"] = features["rollout_id"].astype(int)
    features["trimmed_margin_max"] = trimmed["correct_margin_max"].astype(float)
    features["trimmed_margin_mean"] = trimmed["correct_margin_mean"].astype(float)
    features["trimmed_entropy"] = trimmed["abcd_entropy"].astype(float)
    if "is_top_correct" in trimmed:
        features["trimmed_is_top_correct"] = trimmed["is_top_correct"].astype(float)

    if original is not None and not original.empty:
        original = original.copy()
        original["question_id"] = original["question_id"].astype(str)
        original["rollout_id"] = original["rollout_id"].astype(int)
        merge_cols = [
            "question_id",
            "rollout_id",
            "prompt_only_margin_max",
            "prompt_only_margin_mean",
            "think_final_margin_max",
            "think_final_margin_mean",
            "think_relative_final_margin_max",
            "think_relative_final_margin_mean",
        ]
        merge_cols = [col for col in merge_cols if col in original.columns]
        features = features.merge(original[merge_cols], on=["question_id", "rollout_id"], how="left")
        if "prompt_only_margin_max" in features:
            features["trimmed_relative_margin_max"] = features["trimmed_margin_max"] - features["prompt_only_margin_max"]
        if "prompt_only_margin_mean" in features:
            features["trimmed_relative_margin_mean"] = features["trimmed_margin_mean"] - features["prompt_only_margin_mean"]
        if "think_final_margin_max" in features:
            features["original_minus_trimmed_final_margin_max"] = (
                features["think_final_margin_max"] - features["trimmed_margin_max"]
            )
        if "think_final_margin_mean" in features:
            features["original_minus_trimmed_final_margin_mean"] = (
                features["think_final_margin_mean"] - features["trimmed_margin_mean"]
            )
    return features


def plot_top_auc(eval_df: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    import matplotlib.pyplot as plt

    if eval_df.empty:
        return
    view = eval_df.sort_values("best_auc", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(view))))
    ax.barh(view["feature"], view["best_auc"], color="#276FBF")
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("within-question AUROC (best direction)")
    ax.set_title("Trimmed-conclusion option-logit features")
    ax.set_xlim(0.45, max(0.75, float(view["best_auc"].max()) + 0.03))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_original_vs_trimmed(features: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if features.empty or "think_final_margin_max" not in features:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = features["is_correct"].map({True: "#227C56", False: "#B84343"})
    ax.scatter(features["think_final_margin_max"], features["trimmed_margin_max"], s=12, alpha=0.55, c=colors)
    lo = float(np.nanmin([features["think_final_margin_max"].min(), features["trimmed_margin_max"].min()]))
    hi = float(np.nanmax([features["think_final_margin_max"].max(), features["trimmed_margin_max"].max()]))
    ax.plot([lo, hi], [lo, hi], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("original think final margin max")
    ax.set_ylabel("trimmed conclusion margin max")
    ax.set_title("Original vs trimmed final probe")
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
    report = output_dir / "TRIMMED_CONCLUSION_LOGIT_RESULTS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Trimmed-Conclusion Option Logit Results\n\n")
        f.write("This experiment removes the final conclusion sentence from thinking and probes A/B/C/D logits once.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Probe rows: {len(probes)}\n")
        f.write(f"- Feature rows: {len(features)}\n")
        f.write(f"- Questions: {features['question_id'].nunique() if not features.empty else 0}\n")
        f.write(f"- Rollouts: {len(features)}\n")
        f.write(f"- Skipped rollouts: {len(skipped)}\n")
        if not features.empty and "trim_status" in features:
            f.write(f"- Trim status: {features['trim_status'].value_counts().to_dict()}\n")
        if not probes.empty and "top_option" in probes:
            f.write(f"- Top option counts: {probes['top_option'].value_counts().to_dict()}\n")
        f.write("\n## Primary AUROC\n\n")
        f.write("| feature | direction | questions | AUROC(+feature) | 95% CI | AUROC(-feature) | 95% CI | best |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        if not eval_df.empty:
            top = eval_df.sort_values(["is_primary", "best_auc"], ascending=[False, False]).head(20)
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
        f.write("- If trimmed margins remain strong, the option-logit signal is not only final-answer leakage.\n")
        f.write("- If original-minus-trimmed is the strongest feature, the previous result depended heavily on final conclusion text.\n")
    return report


def forward_trimmed_probes(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    labels = parse_labels(args.option_labels)
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_jsonl(Path(args.input)), args.clean_only, args.limit_rollouts)
    if not rows:
        raise ValueError("No rows selected.")

    print("Trimmed-conclusion option logit probe")
    print(f"input: {args.input}")
    print(f"selected rollouts: {len(rows)}")
    print(f"output_dir: {args.output_dir}")
    print(f"model: {args.model}")
    print(f"labels: {labels}")

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
    for row in tqdm(rows, desc="trimmed option logits"):
        qid = str(row.get("question_id", ""))
        rid = int(row.get("rollout_id", -1))
        correct = normalize_label(row.get("answer"))
        chosen = normalize_label(row.get("pred_answer"))
        try:
            response = str(row.get("response", ""))
            think_text, _, _, think_status = think_text_and_char_span(response)
            trimmed_text, trim_status = trim_final_conclusion(
                think_text,
                min_keep_chars=args.min_keep_chars,
                tail_window_chars=args.tail_window_chars,
            )
            messages = build_probe_messages(row, trimmed_text, args.probe_suffix)
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
                    "think_status": think_status,
                    "trim_status": trim_status,
                    "think_chars": int(len(think_text)),
                    "trimmed_chars": int(len(trimmed_text)),
                    "removed_chars": int(len(think_text) - len(trimmed_text)),
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
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    probes, skipped, label_ids = forward_trimmed_probes(args)
    original = pd.read_parquet(args.original_features) if args.original_features else None
    features = build_trimmed_features(probes, original)
    eval_df = evaluate_trimmed_features(features, args.bootstrap, args.seed)

    probes_path = output_dir / "trimmed_conclusion_probes.parquet"
    features_path = output_dir / "trimmed_conclusion_features.parquet"
    eval_path = output_dir / "trimmed_conclusion_eval.csv"
    skipped_path = output_dir / "trimmed_conclusion_skipped.jsonl"
    probes.to_parquet(probes_path, index=False)
    features.to_parquet(features_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    with skipped_path.open("w", encoding="utf-8") as f:
        for item in skipped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    figures: list[Path] = []
    fig = figures_dir / "T1_trimmed_top_auc.png"
    plot_top_auc(eval_df, fig)
    if fig.exists():
        figures.append(fig)
    fig = figures_dir / "T2_original_vs_trimmed_margin.png"
    plot_original_vs_trimmed(features, fig)
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
