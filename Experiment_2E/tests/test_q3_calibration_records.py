import json

from experiment_2e.q3_1p7b_calibration import _is_cap_truncated, _records


def test_records_combines_multiple_read_only_rollout_roots(tmp_path):
    roots = []
    for root_index in range(2):
        root = tmp_path / f"root{root_index}" / "base" / "parts"
        root.mkdir(parents=True)
        record = {"question_id": f"q{root_index}", "rollout_slot": 0}
        (root / "part_00000.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        roots.append(root.parents[1])
    frame = _records(roots)
    assert set(frame["question_id"]) == {"q0", "q1"}


def test_cap_boundary_uses_original_length_finish_status():
    assert _is_cap_truncated(4097, 4096, False) is True
    assert _is_cap_truncated(4096, 4096, True) is True
    assert _is_cap_truncated(4096, 4096, False) is False
    assert _is_cap_truncated(4095, 4096, True) is False


def test_non_format_truncation_is_separate_from_raw_length_truncation():
    import pandas as pd

    work = pd.DataFrame(
        {
            "cap_truncated": [True, True, False],
            "parse_correct": [False, True, False],
        }
    )
    non_format = work["cap_truncated"] & ~work["parse_correct"]
    assert non_format.tolist() == [True, False, False]
