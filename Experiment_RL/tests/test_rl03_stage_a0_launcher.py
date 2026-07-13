from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_stage_a0_runner_keeps_artifacts_on_data2_and_runs_full_chain():
    text = (SCRIPTS / "run_rl03_mcq_audit_stage_a0.sh").read_text(encoding="utf-8")

    for variable in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "PIP_CACHE_DIR",
        "TMPDIR",
        "RAY_TMPDIR",
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        assert f"export {variable}=/data2/" in text
    assert "build_rl03_mcq_audit_manifest.py" in text
    assert "run_rl03_mcq_audit_vllm.py" in text
    assert "analyze_rl03_mcq_audit.py" in text
    assert "option_logit_trimmed_conclusion_smoke500/trimmed_conclusion_features.parquet" in text
    assert "--bootstrap-samples" in text
    assert "--seed" in text
    assert "--question-count 32" in text
    assert "--request-limit" not in text
    assert "QUESTION_COUNT=${" not in text
    assert "require_data2_path" in text
    assert "realpath -m" in text
    assert "RL03_STAGE_A0_EXIT_CODE=" in text
    assert "resource_samples.csv" in text
    assert "resource_peaks.txt" in text
    assert "sudo " not in text
    assert "ray stop" not in text
    assert "pkill " not in text
    assert "killall " not in text


def test_stage_a0_preflight_aborts_on_busy_gpu_without_cleaning_processes():
    text = (SCRIPTS / "run_rl03_mcq_audit_stage_a0.sh").read_text(encoding="utf-8")

    assert "nvidia-smi --query-compute-apps=pid" in text
    assert "GPU is busy" in text
    assert "pgrep -u" in text
    assert "raylet" in text
    assert "gcs_server" in text


def test_stage_a0_resource_sampler_keeps_xtrace_out_of_csv():
    text = (SCRIPTS / "run_rl03_mcq_audit_stage_a0.sh").read_text(encoding="utf-8")

    assert 'sample_resources > "${RUN_ROOT}/resource_samples.csv" 2> "${RUN_ROOT}/resource_samples.stderr" &' in text
    assert 'sample_resources > "${RUN_ROOT}/resource_samples.csv" 2>&1 &' not in text


def test_stage_a0_tmux_launcher_is_idempotent_and_reports_attach_command():
    text = (SCRIPTS / "launch_rl03_mcq_audit_stage_a0_tmux.sh").read_text(
        encoding="utf-8"
    )

    assert "tmux has-session" in text
    assert "tmux new-session -d" in text
    assert "tmux attach -t" in text
    assert "tail -f" in text
    assert "PROJECT_ROOT='${PROJECT_ROOT}'" in text
    assert "realpath -m" in text
    for variable in (
        "ENV_ROOT",
        "MODEL_PATH",
        "RAW_ROLLOUTS",
        "FEATURES_PATH",
        "MATHVERSE_METADATA",
        "SEED",
        "BATCH_SIZE",
        "MAX_MODEL_LEN",
        "GPU_MEMORY_UTILIZATION",
        "BOOTSTRAP_SAMPLES",
        "OVERWRITE",
    ):
        assert f"{variable}='${{{variable}}}'" in text


def test_legacy_stage_a0_entrypoints_fail_closed():
    for name in (
        "autostart_rl03_mcq_stage_a0_after_gpu_idle.sh",
        "run_rl03_mcq_stage_a0.sh",
        "launch_rl03_mcq_stage_a0_tmux.sh",
    ):
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "RL03_DEPRECATED_ENTRYPOINT" in text
        assert "exit 2" in text
        assert "REQUEST_LIMIT" not in text
        assert "QUESTION_COUNT" not in text
