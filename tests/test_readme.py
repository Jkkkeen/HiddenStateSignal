import subprocess
import sys
from pathlib import Path


def test_readme_contains_required_runbook_sections():
    text = Path("vertical_readme.md").read_text(encoding="utf-8").lower()
    for heading in (
        "preflight",
        "smoke",
        "tmux",
        "base calibration",
        "troubleshooting",
        "do not push",
    ):
        assert heading in text


def test_synthetic_generator_writes_small_raw_and_pooled_examples(tmp_path):
    output = tmp_path / "generated"

    result = subprocess.run(
        [
            sys.executable,
            "examples/synthetic_hidden/make_fixture.py",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list((output / "qwen3_base_raw").glob("*.npz"))
    assert list((output / "pooled_example").glob("*.npz"))
    assert (output / "synthetic_qwen.yaml").is_file()
    assert max(path.stat().st_size for path in output.rglob("*.npz")) < 1024 * 1024

