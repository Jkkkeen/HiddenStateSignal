import pandas as pd

from experiment_2e.q3_simplerl_manifests import prepare


def _frame(start, count):
    return pd.DataFrame(
        [
            {
                "question": f"Question {index}?",
                "level": 3,
                "reward_model": {"ground_truth": str(index)},
                "extra_info": {"question": f"Question {index}?", "level": 3},
            }
            for index in range(start, start + count)
        ]
    )


def test_simplerl_strict_splits_are_disjoint(tmp_path):
    train_source = tmp_path / "train.parquet"
    test_source = tmp_path / "test.parquet"
    _frame(0, 20).to_parquet(train_source, index=False)
    _frame(100, 8).to_parquet(test_source, index=False)
    output = tmp_path / "out"

    audit = prepare(
        train_source,
        test_source,
        output,
        seed=7,
        calibration_size=4,
        heldout_size=4,
        smoke_size=3,
    )
    assert audit["passed"] is True
    assert audit["counts"] == {
        "simplerl_calibration": 4,
        "simplerl_smoke": 3,
        "simplerl_train": 13,
        "simplerl_heldout": 4,
    }
    assert all(value == 0 for value in audit["normalized_exact_hash_leakage"].values())
    prompt = pd.read_parquet(output / "simplerl_train_candidates.parquet").iloc[0]["prompt"][0]["content"]
    assert "Final Answer: \\boxed{YOUR_ANSWER}" in prompt
    smoke_val = pd.read_parquet(output / "simplerl_smoke_val_16.parquet")
    assert len(smoke_val) == 4
    assert set(smoke_val["extra_info"].map(lambda value: value["prompt_hash"])) <= set(
        pd.read_parquet(output / "simplerl_calibration_256.parquet")["extra_info"].map(
            lambda value: value["prompt_hash"]
        )
    )
