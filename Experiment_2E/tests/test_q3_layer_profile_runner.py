from pathlib import Path


RUNNER = Path("Experiment_2E/scripts/run_q3_layer_profiles.sh")
LAUNCHER = Path("Experiment_2E/scripts/launch_q3_layer_profiles_tmux.sh")


def test_runner_freezes_smoke_and_formal_cohorts():
    text = RUNNER.read_text(encoding="utf-8")
    assert "STEPS=(0 50 250)" in text
    assert "STEPS=(0 25 50 75 100 125 150 175 200 225 250)" in text
    assert "QUESTION_LIMIT=16" in text
    assert "QUESTION_LIMIT=256" in text
    assert "verl.model_merger merge --backend fsdp" in text
    assert 'safe_remove_merged "${MERGED}"' in text
    assert 'rm -rf -- "${target}"' in text
    assert "smoke_approval.json" in text
    assert "--source-checkpoint-dir" in text
    assert "--base-parameter-sample" in text
    assert "experiment_2e.q3_layer_analysis" in text


def test_runner_validates_cleanup_target_before_removing_merge():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'merged_root=$(realpath -m "${OUTPUT_ROOT}/work/merged")' in text
    assert 'merged_path=$(realpath -m "${target}")' in text
    assert '"${merged_root}"/*)' in text
    assert "refusing unsafe merged-model cleanup" in text


def test_formal_launcher_requires_passing_smoke_and_tmux():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "MODE=${MODE:-smoke}" in text
    assert "q3_layer_profiles_smoke" in text
    assert "q3_layer_profiles_formal" in text
    assert "p.get('status') == 'passed'" in text
    assert "tmux new-session -d" in text
    assert "experiment_2e.q3_layer_extract" in text
    assert "nvidia-smi --query-compute-apps" in text
    assert "tmux has-session" in text
