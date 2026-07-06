import json
import math
import re
from pathlib import Path
from statistics import mean

METRICS = [
    "training/global_step",
    "critic/score/mean",
    "critic/score/max",
    "critic/score/min",
    "critic/rewards/mean",
    "actor/grad_norm",
    "actor/loss",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
    "response_length/clip_ratio",
    "prompt_length/mean",
    "timing_s/gen",
    "timing_s/adv",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/step",
    "actor/perf/max_memory_allocated_gb",
    "actor/perf/max_memory_reserved_gb",
    "actor/perf/cpu_memory_used_gb",
    "c2_frac_groups_with_nonzero_b_gain_std",
    "c2_frac_groups_where_b_gain_changes_ranking",
    "c2_frac_all_same_answer_groups_with_b_gain_ranking",
    "c2_within_group_corr_answer_b_gain_mean",
]

num = r"(?:np\.float64\()?(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\)?"

def parse_log(path: Path):
    text = path.read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        if "training/global_step:" not in line:
            continue
        row = {}
        for key in METRICS:
            m = re.search(re.escape(key) + r":" + num, line)
            if m:
                row[key] = float(m.group(1))
        if "training/global_step" in row:
            rows.append(row)
    errors = []
    for pat in ["CUDA Error: out of memory", "OutOfMemory", "Traceback", "RuntimeError", "EXIT_CODE=1", "EXIT_CODE=0"]:
        count = text.count(pat)
        if count:
            errors.append({"pattern": pat, "count": count})
    return rows, errors, text

def summarize(rows):
    if not rows:
        return {}
    out = {"steps_seen": len(rows), "last_step": int(rows[-1].get("training/global_step", -1))}
    for key in METRICS:
        vals = [r[key] for r in rows if key in r and math.isfinite(r[key])]
        if vals:
            out[key.replace('/', '_')] = {
                "first": vals[0],
                "last": vals[-1],
                "mean": mean(vals),
                "min": min(vals),
                "max": max(vals),
            }
    return out

root = Path(__file__).resolve().parent
summaries = {}
for path in sorted(root.glob("*.log")):
    rows, errors, text = parse_log(path)
    summaries[path.name] = {"summary": summarize(rows), "markers": errors}

(root / "goal4_log_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
for name, payload in summaries.items():
    s = payload["summary"]
    print("\n", name)
    print("  steps", s.get("steps_seen"), "last", s.get("last_step"))
    for key in ["critic_score_mean", "response_length_mean", "timing_s_gen", "timing_s_adv", "timing_s_step", "c2_frac_groups_with_nonzero_b_gain_std", "c2_frac_all_same_answer_groups_with_b_gain_ranking"]:
        if key in s:
            print(" ", key, s[key])
    print("  markers", payload["markers"])
