import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from vertical.cli import render_tmux_command, validate_smoke_approval


def write_config(tmp_path: Path) -> Path:
    source = tmp_path / "data"
    source.mkdir()
    metadata = {
        "record_id": "q1:r0",
        "question_id": "q1",
        "rollout_id": "r0",
        "is_correct": True,
        "response_token_count": 4,
        "num_decoder_layers": 2,
    }
    np.savez(
        source / "record.npz",
        hidden_states=np.zeros((4, 3, 5), dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "run_id: cli_test\n"
        f"output_root: {(tmp_path / 'runs').as_posix()}\n"
        "datasets:\n"
        "  - name: qwen_base\n"
        "    input_mode: raw\n"
        "    adapter: npz\n"
        f"    source: {source.as_posix()}\n"
        "    model_family: qwen3\n"
        "    model_name: Qwen3-Test\n"
        "    condition: base\n"
        "    decoder_layers: 2\n"
        "    hidden_dimension: 5\n"
        "    tensor_key: hidden_states\n"
        "    metadata_json_key: metadata_json\n"
        "    axis_order: token,layer,dim\n"
        "representations:\n"
        "  - last_s32\n"
        "question_equal: true\n",
        encoding="utf-8",
    )
    return config


def run_cli(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "vertical.cli", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_inspect_cli_writes_combined_audit(tmp_path):
    config = write_config(tmp_path)
    output = tmp_path / "input_audit.json"

    result = run_cli("inspect", "--config", str(config), "--output", str(output))

    assert result.returncode == 0, result.stderr
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["datasets"]["qwen_base"]["sampled_records"] == 1


def test_formal_cli_requires_passing_smoke_approval(tmp_path):
    config = write_config(tmp_path)

    result = run_cli(
        "formal",
        "--config",
        str(config),
        "--output-root",
        str(tmp_path / "formal"),
        "--smoke-approval",
        str(tmp_path / "missing.json"),
    )

    assert result.returncode != 0
    assert "smoke approval" in result.stderr.lower()


def test_smoke_approval_validator_rejects_failed_file(tmp_path):
    path = tmp_path / "smoke_approval.json"
    path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    try:
        validate_smoke_approval(path)
    except ValueError as exc:
        assert "smoke approval" in str(exc).lower()
    else:
        raise AssertionError("failed smoke approval must be rejected")


def test_tmux_command_is_named_detached_and_persistent(tmp_path):
    command = render_tmux_command(
        "vertical_formal_mimo",
        Path("scripts/run_vertical_formal.sh"),
        config=Path("configs/mimo_7b.example.yaml"),
        output_root=tmp_path / "run",
        smoke_approval=tmp_path / "smoke_approval.json",
        log_path=tmp_path / "logs" / "formal.log",
    )

    assert command[:5] == ["tmux", "new-session", "-d", "-s", "vertical_formal_mimo"]
    assert "run_vertical_formal.sh" in command[-1]
    assert "tee" in command[-1]
