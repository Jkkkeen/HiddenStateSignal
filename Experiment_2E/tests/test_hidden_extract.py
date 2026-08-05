import numpy as np

from experiment_2e.hidden_extract import enrich_metric_rows, stage_controls


def test_stage_controls_use_token_progress_and_enrich_metric_rows():
    values = np.arange(8, dtype=float)
    metadata = {"run_id": "r", "rollout_id": "x", "checkpoint": "base"}
    controls = stage_controls(
        token_logprob=-values,
        policy_entropy=values,
        hidden_norm=values + 1,
        response_token_count=8,
        trajectory_point_count=2,
        metadata=metadata,
    )
    assert [row["stage_token_count"] for row in controls] == [1, 2, 2, 3]
    rows = [{"stage": stage, "metric": "m", "value": 1.0} for stage in range(4)]
    enrich_metric_rows(rows, controls)
    assert rows[0]["policy_entropy"] == controls[0]["policy_entropy"]
    assert rows[3]["response_length"] == 8
