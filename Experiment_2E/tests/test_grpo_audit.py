from pathlib import Path

from experiment_2e.grpo_audit import audit_grpo, parse_step_metrics


def test_grpo_log_parser_and_resume_audit(tmp_path: Path):
    log = tmp_path / "smoke.log"
    log.write_text(
        "step:5 - training/global_step:5 - critic/score/max:1.0 - critic/score/min:0.0 "
        "- critic/advantages/max:np.float64(0.9) - critic/advantages/min:np.float64(-0.9) "
        "- actor/ppo_kl:np.float64(0.01)\n"
        "step:10 - training/global_step:10 - critic/score/max:0.0 - critic/score/min:0.0 "
        "- critic/advantages/max:np.float64(0.0) - critic/advantages/min:np.float64(0.0)\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoints"
    (checkpoint / "global_step_5").mkdir(parents=True)
    (checkpoint / "global_step_10").mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("10\n", encoding="utf-8")
    assert [row["training/global_step"] for row in parse_step_metrics([log])] == [5.0, 10.0]
    audit = audit_grpo([log], checkpoint)
    assert audit["mixed_reward_group_rate"] == 0.5
    assert audit["n_nonzero_advantage_steps"] == 1
    assert audit["resume_verified"] is True
