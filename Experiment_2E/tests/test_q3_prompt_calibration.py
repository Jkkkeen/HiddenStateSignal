import pandas as pd
import sys

from experiment_2e.q3_prompt_calibration import build, main


def test_strict_prompt_calibration_is_audited_and_formally_excluded(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "strict.parquet"
    audit = tmp_path / "strict.json"
    pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": "What is 2+2?\n\nPlease reason step by step. Old."}]],
            "extra_info": [{"question_id": "q1", "experiment_split": "calibration"}],
        }
    ).to_parquet(source, index=False)

    payload = build(source, output, audit, 1, 0)
    row = pd.read_parquet(output).iloc[0]
    assert "Final Answer: \\boxed{YOUR_ANSWER}" in row["prompt"][0]["content"]
    assert row["extra_info"]["prompt_protocol"] == "strict_boxed_v2"
    assert payload["formal_exclusion"] is True
    assert audit.exists()


def test_cli_maps_public_flag_names_to_build_arguments(tmp_path, monkeypatch):
    source = tmp_path / "source.parquet"
    output = tmp_path / "strict.parquet"
    audit = tmp_path / "strict.json"
    pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": "What is 2+2?\n\nPlease reason step by step. Old."}]],
            "extra_info": [{"question_id": "q1"}],
        }
    ).to_parquet(source, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["q3_prompt_calibration", "--input", str(source), "--output", str(output), "--audit", str(audit), "--size", "1"],
    )
    main()
    assert output.exists()
    assert audit.exists()


def test_start_selects_a_nonoverlapping_manifest_slice(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "strict.parquet"
    audit = tmp_path / "strict.json"
    pd.DataFrame(
        {
            "prompt": [
                [{"role": "user", "content": "Question zero?\n\nPlease reason step by step. Old."}],
                [{"role": "user", "content": "Question one?\n\nPlease reason step by step. Old."}],
            ],
            "extra_info": [{"question_id": "q0"}, {"question_id": "q1"}],
        }
    ).to_parquet(source, index=False)
    payload = build(source, output, audit, 1, 1)
    assert payload["start"] == 1
    row = pd.read_parquet(output).iloc[0]
    assert row["extra_info"]["question_id"] == "q1"
