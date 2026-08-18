import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from vertical.pipeline import run_pipeline


def synthetic_mimo_config(tmp_path: Path) -> Path:
    fixture_root = tmp_path / "fixture"
    result = subprocess.run(
        [
            sys.executable,
            "examples/synthetic_hidden/make_fixture.py",
            "--output",
            str(fixture_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    source = fixture_root / "synthetic_mimo.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["representations"] = ["last_s32"]
    path = fixture_root / "synthetic_mimo_e2e.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_v01_pipeline_produces_audited_profiles_figures_and_status(tmp_path):
    config = synthetic_mimo_config(tmp_path)
    output_root = tmp_path / "run"

    result = run_pipeline(
        config_path=config,
        output_root=output_root,
        mode="smoke",
    )

    assert result["passed"] is True
    assert result["status"] == "completed"
    assert (output_root / "audit" / "final_audit.json").is_file()
    assert (output_root / "smoke_approval.json").is_file()
    assert list((output_root / "analysis" / "figures").glob("*.png"))
    assert not list(output_root.glob("**/*hidden*.npz"))

    profiles = pd.read_parquet(output_root / "profiles" / "profiles.parquet")
    assert set(profiles["condition"]) == {"base", "sft"}
    assert profiles["layer_index"].max() == 4
    assert profiles["hidden_state_count"].max() == 6

    final_audit = json.loads(
        (output_root / "audit" / "final_audit.json").read_text(encoding="utf-8")
    )
    assert final_audit["passed"] is True
    assert set(final_audit["audits"]) == {
        "analysis",
        "calibration",
        "input",
        "profiles",
    }


def test_formal_pipeline_requires_external_passing_approval(tmp_path):
    config = synthetic_mimo_config(tmp_path)

    try:
        run_pipeline(
            config_path=config,
            output_root=tmp_path / "formal",
            mode="formal",
            smoke_approval=tmp_path / "missing.json",
        )
    except ValueError as exc:
        assert "smoke approval" in str(exc).lower()
    else:
        raise AssertionError("formal pipeline must require smoke approval")
