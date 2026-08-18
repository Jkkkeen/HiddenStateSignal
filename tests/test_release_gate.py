from pathlib import Path

from vertical.release import run_release_gate


def test_release_gate_rejects_real_tensor_candidate(tmp_path):
    (tmp_path / "runs").mkdir(parents=True)
    (tmp_path / "runs" / "raw_hidden.npz").write_bytes(b"tensor")

    result = run_release_gate(tmp_path)

    assert result["tracked_data_gate"] is False
    assert result["passed"] is False


def test_release_gate_reports_metric_and_readme_statuses():
    result = run_release_gate(Path.cwd())

    assert "tests_gate" in result
    assert "shell_syntax_gate" in result
    assert "fixture_size_gate" in result
    assert "readme_command_gate" in result
    assert "metric_registry_gate" in result
    assert result["readme_command_gate"] is True
    assert result["metric_registry_gate"] is True
