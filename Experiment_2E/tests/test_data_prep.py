import pandas as pd

from experiment_2e.data_prep import build_verl_records, parse_level, prepare_dataset, stratified_sample


def test_parse_level_accepts_math_dataset_forms():
    assert parse_level("Level 3") == 3
    assert parse_level(5) == 5
    assert parse_level("level 4") == 4


def test_build_train_records_filters_level_and_unparseable_gold():
    frame = pd.DataFrame(
        [
            {"problem": "p1", "solution": r"work \boxed{1}", "level": "Level 3", "type": "Algebra"},
            {"problem": "p2", "solution": r"work \boxed{2}", "level": "Level 2", "type": "Algebra"},
            {"problem": "p3", "solution": "no box", "level": "Level 5", "type": "Geometry"},
        ]
    )
    records, rejected = build_verl_records(frame, split="train", levels={3, 4, 5})
    assert [row["extra_info"]["question_id"] for row in records] == ["math:train:0"]
    assert records[0]["reward_model"]["ground_truth"] == "1"
    assert {row["reason"] for row in rejected} == {"level_not_selected", "gold_unparseable"}


def test_stratified_sample_is_deterministic_and_exact_size():
    rows = []
    for level, subject, count in [(3, "A", 8), (4, "A", 4), (5, "B", 8)]:
        for index in range(count):
            rows.append({"level": level, "subject": subject, "question_id": f"{level}-{subject}-{index}"})
    frame = pd.DataFrame(rows)
    left = stratified_sample(frame, n=10, strata=("level", "subject"), seed=17)
    right = stratified_sample(frame, n=10, strata=("level", "subject"), seed=17)
    assert len(left) == 10
    assert left["question_id"].tolist() == right["question_id"].tolist()
    assert set(left["level"]) == {3, 4, 5}


def test_prepare_dataset_writes_verl_parquet_and_seed_manifest(tmp_path):
    def frame(split: str, count: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "problem": f"{split} problem {index}",
                    "solution": rf"work \boxed{{{index}}}",
                    "level": f"Level {3 + index % 3}",
                    "type": "Algebra" if index % 2 else "Geometry",
                }
                for index in range(count)
            ]
        )

    audit = prepare_dataset(frame("train", 12), frame("test", 12), tmp_path, eval_size=6, seed=7)
    assert audit["n_train"] == 12
    assert audit["n_eval"] == 6
    prepared = pd.read_parquet(tmp_path / "eval_256.parquet")
    assert len(prepared) == 6
    seed_manifest = pd.read_csv(tmp_path / "eval_seed_manifest.csv")
    assert len(seed_manifest) == 6 * 6 * 8
    assert set(seed_manifest["rollout_slot"]) == set(range(8))
