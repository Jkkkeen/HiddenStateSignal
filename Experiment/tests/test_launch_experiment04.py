from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "launch_experiment04_tmux.sh"


def test_launcher_keeps_discovery_and_confirm_isolated() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"' in text
    assert 'SMOKE_SESSION="${SMOKE_SESSION:-exp04_fixed256_smoke}"' in text
    assert 'DISCOVERY_SESSION="${DISCOVERY_SESSION:-exp04_fixed256_discovery}"' in text
    assert "manifest_discovery.jsonl" in text
    assert "manifest_confirm.jsonl" in text
    assert "frozen_hypotheses.json" in text
    assert "validate_confirm_guard" in text
    assert "tmux has-session" in text
    discovery_block = text.split("run_discovery_pipeline()", 1)[1].split(
        "run_confirm_pipeline()", 1
    )[0]
    assert "manifest_confirm.jsonl" not in discovery_block


def test_launcher_requires_smoke_success_before_discovery() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    discovery_branch = text.split("--discovery)", 1)[1]
    assert "SMOKE_SUCCESS" in discovery_branch
    assert "start_tmux" in discovery_branch
