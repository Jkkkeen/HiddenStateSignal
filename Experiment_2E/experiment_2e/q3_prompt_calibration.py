from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


STRICT_INSTRUCTION = r"""You must follow this output contract exactly:
1. Reason step by step.
2. End with one final line in the form: Final Answer: \boxed{YOUR_ANSWER}
3. Replace YOUR_ANSWER with the answer. Never omit \boxed{}.
4. Do not write anything after that final boxed line.

For example, if the answer is 42, the final line must be:
Final Answer: \boxed{42}"""


def build(input_path: Path, output_path: Path, audit_path: Path, size: int, start: int = 0) -> dict:
    frame = pd.read_parquet(input_path).iloc[start : start + size].copy()
    if len(frame) != size:
        raise ValueError(f"requested rows [{start}, {start + size}), found {len(frame)}")

    def rewrite(value):
        prompt = value.tolist() if hasattr(value, "tolist") else list(value)
        question = str(prompt[0]["content"]).split("\n\nPlease reason step by step.", 1)[0].strip()
        return [{"role": "user", "content": f"{question}\n\n{STRICT_INSTRUCTION}"}]

    frame["prompt"] = frame["prompt"].map(rewrite)
    frame["extra_info"] = frame["extra_info"].map(
        lambda value: {**dict(value), "prompt_protocol": "strict_boxed_v2", "experiment_split": "prompt_calibration"}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    payload = {
        "source": str(input_path),
        "output": str(output_path),
        "size": size,
        "start": start,
        "prompt_protocol": "strict_boxed_v2",
        "instruction": STRICT_INSTRUCTION,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "formal_exclusion": True,
    }
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, args.audit, args.size, args.start), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
