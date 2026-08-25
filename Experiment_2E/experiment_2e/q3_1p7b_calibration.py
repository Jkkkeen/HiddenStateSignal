from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .four_level_reward import extract_unboxed_final_answer
from .math_reward import score_response


def _is_cap_truncated(token_count: int, cap: int, full_length_truncated: bool) -> bool:
    return token_count > cap or (token_count == cap and full_length_truncated)


def _records(roots: Path | list[Path]) -> pd.DataFrame:
    if isinstance(roots, Path):
        roots = [roots]
    rows = []
    for root in roots:
        for path in sorted(root.glob("**/parts/part_*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError(f"no rollout records under {roots}")
    return pd.DataFrame(rows)


def _bootstrap_difference(frame: pd.DataFrame, *, repeats: int, seed: int) -> tuple[float, float]:
    pivot = frame.groupby("question_id")[["answer_reward", "full_answer_reward"]].mean()
    differences = (pivot["answer_reward"] - pivot["full_answer_reward"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    for index in range(repeats):
        estimates[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def analyze(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    records = _records([args.rollout_dir, *args.additional_rollout_dir])
    expected = args.questions * args.rollouts
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, found {len(records)}")
    if records.duplicated(["question_id", "rollout_slot"]).any():
        raise ValueError("duplicate question/rollout slots")
    records["full_answer_reward"] = records["answer_reward"].astype(float)
    rows = []
    for cap in args.caps:
        scored = []
        for row in records.itertuples(index=False):
            token_ids = list(row.response_token_ids)
            response = row.response if len(token_ids) <= cap else tokenizer.decode(token_ids[:cap], skip_special_tokens=True)
            score = score_response(response, row.ground_truth)
            parseable_answer = score["parsed_answer"] or extract_unboxed_final_answer(response)
            scored.append(
                {
                    "question_id": row.question_id,
                    "rollout_slot": int(row.rollout_slot),
                    "answer_reward": float(score["answer_reward"]),
                    "format_reward": float(score["format_reward"]),
                    "parse_correct": parseable_answer is not None,
                    "full_answer_reward": float(row.full_answer_reward),
                    "cap_truncated": _is_cap_truncated(len(token_ids), cap, bool(row.truncated)),
                    "full_length_truncated": bool(row.truncated),
                    "response_token_count": len(token_ids),
                }
            )
        work = pd.DataFrame(scored)
        work["non_format_truncated"] = work["cap_truncated"] & ~work["parse_correct"]
        grouped = work.groupby("question_id")["answer_reward"].sum()
        ci_low, ci_high = _bootstrap_difference(work, repeats=args.bootstrap, seed=args.seed + cap)
        rows.append(
            {
                "cap": int(cap),
                "pass_at_1": float(work["answer_reward"].mean()),
                "mixed_fraction": float(((grouped >= 1) & (grouped <= args.rollouts - 1)).mean()),
                "useful_mixed_fraction": float(((grouped >= 2) & (grouped <= args.rollouts - 2)).mean()),
                "all_wrong_fraction": float((grouped == 0).mean()),
                "all_correct_fraction": float((grouped == args.rollouts).mean()),
                "boxed_rate": float(work["format_reward"].mean()),
                "parse_rate": float(work["parse_correct"].mean()),
                "cap_truncation_rate": float(work["cap_truncated"].mean()),
                "non_format_truncation_rate": float(work["non_format_truncated"].mean()),
                "full_length_truncation_rate": float(work["full_length_truncated"].mean()),
                "cap_minus_full_accuracy": float((work["answer_reward"] - work["full_answer_reward"]).mean()),
                "cap_minus_full_accuracy_ci95": [ci_low, ci_high],
                "response_tokens_mean": float(work["response_token_count"].mean()),
                "response_tokens_p50": float(work["response_token_count"].quantile(0.50)),
                "response_tokens_p90": float(work["response_token_count"].quantile(0.90)),
                "response_tokens_p95": float(work["response_token_count"].quantile(0.95)),
            }
        )
    acceptable_caps = [
        row["cap"]
        for row in rows
        if row["non_format_truncation_rate"] <= 0.05
        and row["cap_minus_full_accuracy_ci95"][0] <= 0.0
    ]
    full = next(row for row in rows if row["cap"] == max(args.caps))
    gates = {
        "pass_at_1_in_range": 0.15 <= full["pass_at_1"] <= 0.50,
        "mixed_fraction_ge_0p35": full["mixed_fraction"] >= 0.35,
        "useful_mixed_fraction_ge_0p20": full["useful_mixed_fraction"] >= 0.20,
        "boxed_rate_ge_0p70": full["boxed_rate"] >= 0.70,
        "parse_rate_ge_0p90": full["parse_rate"] >= 0.90,
        "non_format_truncation_le_0p05": full["non_format_truncation_rate"] <= 0.05,
        "at_least_one_acceptable_cap": bool(acceptable_caps),
    }
    payload = {
        "dataset": args.dataset,
        "n_questions": args.questions,
        "rollouts_per_question": args.rollouts,
        "caps": rows,
        "selected_cap": min(acceptable_caps) if acceptable_caps else None,
        "gates": gates,
        "status": "passed" if all(gates.values()) else "failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--additional-rollout-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=256)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--caps", type=lambda value: [int(item) for item in value.split(",")], default=[4096, 8192, 12288])
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260814)
    print(json.dumps(analyze(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
