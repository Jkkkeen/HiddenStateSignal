from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepmath", type=Path, required=True)
    parser.add_argument("--simplerl", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deepmath = json.loads(args.deepmath.read_text(encoding="utf-8"))
    simplerl = json.loads(args.simplerl.read_text(encoding="utf-8"))
    if deepmath["status"] == "passed":
        selected = "deepmath"
        train_file = args.data_root / "deepmath_train_candidates.parquet"
        calibration_file = args.data_root / "deepmath_calibration_256.parquet"
        heldout_file = args.data_root / "deepmath_heldout_256.parquet"
        smoke_file = args.data_root / "deepmath_smoke_128.parquet"
        smoke_val_file = args.data_root / "deepmath_smoke_val_16.parquet"
        audit = deepmath
    elif simplerl["status"] == "passed":
        selected = "simplerl"
        train_file = args.data_root / "simplerl_train_candidates.parquet"
        calibration_file = args.data_root / "simplerl_calibration_256.parquet"
        heldout_file = args.data_root / "simplerl_heldout_256.parquet"
        smoke_file = args.data_root / "simplerl_smoke_128.parquet"
        smoke_val_file = args.data_root / "simplerl_smoke_val_16.parquet"
        audit = simplerl
    else:
        selected = None
        train_file = calibration_file = heldout_file = smoke_file = smoke_val_file = None
        audit = None
    payload = {
        "status": "go" if selected else "no-go",
        "selected_dataset": selected,
        "selected_cap": audit["selected_cap"] if audit else None,
        "train_file": str(train_file) if train_file else None,
        "calibration_file": str(calibration_file) if calibration_file else None,
        "heldout_file": str(heldout_file) if heldout_file else None,
        "smoke_file": str(smoke_file) if smoke_file else None,
        "smoke_val_file": str(smoke_val_file) if smoke_val_file else None,
        "formal_steps": 300 if selected == "deepmath" else (250 if selected == "simplerl" else None),
        "formal_accepted_groups": 9600 if selected == "deepmath" else (8000 if selected == "simplerl" else None),
        "deepmath_status": deepmath["status"],
        "simplerl_status": simplerl["status"],
        "input_sha256": {
            "deepmath": hashlib.sha256(args.deepmath.read_bytes()).hexdigest(),
            "simplerl": hashlib.sha256(args.simplerl.read_bytes()).hexdigest(),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not selected:
        raise SystemExit("both calibration candidates failed; formal GRPO is NO-GO")


if __name__ == "__main__":
    main()
