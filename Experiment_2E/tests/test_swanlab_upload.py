import json
from pathlib import Path

import pandas as pd
import pytest

from experiment_2e.swanlab_upload import build_step_payloads, upload_analysis_bundle


TEST_KEYS = [
    "axis",
    "family_id",
    "family",
    "representation",
    "anchor_layer",
    "aggregation_mode",
    "metric",
]


def _write_analysis_bundle(root: Path) -> tuple[Path, Path]:
    analysis = root / "analysis"
    figures = root / "figures" / "primary"
    analysis.mkdir(parents=True)
    figures.mkdir(parents=True)
    common = {
        "axis": "horizontal",
        "family_id": "H1",
        "family": "movement",
        "representation": "mean_w128_s32",
        "anchor_layer": 3,
        "aggregation_mode": "local",
        "metric": "median_relative_movement",
        "checkpoint": "20pct",
        "training_progress": 0.2,
        "stage": 0,
        "standardization_ok": True,
    }
    standardized = pd.DataFrame(
        [
            {**common, "question_id": "q1", "rollout_id": "q1-r0", "is_correct": True, "z_value": 0.8},
            {**common, "question_id": "q1", "rollout_id": "q1-r1", "is_correct": False, "z_value": 0.2},
            {**common, "question_id": "q2", "rollout_id": "q2-r0", "is_correct": True, "z_value": 0.6},
            {**common, "question_id": "q2", "rollout_id": "q2-r1", "is_correct": False, "z_value": 0.4},
        ]
    )
    standardized.to_parquet(analysis / "primary_standardized.parquet", index=False)
    pd.DataFrame(
        [
            {
                **{key: common[key] for key in TEST_KEYS},
                "checkpoint": "20pct",
                "training_progress": 0.2,
                "stage": 0,
                "question_equal_auc": 0.75,
                "pair_weighted_auc": 0.75,
                "n_mixed_questions": 2,
                "n_pairs": 2,
            }
        ]
    ).to_csv(analysis / "outcome_effects.csv", index=False)
    (figures / "h1_stage_curves.png").write_bytes(b"not-a-real-image")
    checkpoint_manifest = root / "checkpoint_manifest.json"
    checkpoint_manifest.write_text(
        json.dumps({"checkpoints": [{"checkpoint": "base", "global_step": 0}, {"checkpoint": "20pct", "global_step": 1116}]}),
        encoding="utf-8",
    )
    return analysis, checkpoint_manifest


class FakeSwanLab:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int]] = []

    @staticmethod
    def Image(path: str) -> str:
        return f"image:{Path(path).name}"

    def log(self, data: dict[str, object], step: int) -> None:
        self.logged.append((data, step))


def test_build_step_payloads_has_correct_wrong_gap_and_auc(tmp_path: Path) -> None:
    analysis, checkpoint_manifest = _write_analysis_bundle(tmp_path)

    payloads = build_step_payloads(
        analysis / "primary_standardized.parquet",
        analysis / "outcome_effects.csv",
        checkpoint_manifest,
    )

    assert list(payloads) == [1116]
    payload = payloads[1116]
    prefix = "hidden/H1/mean_w128_s32/L3/median_relative_movement/B1"
    assert payload[f"{prefix}/correct_mean"] == pytest.approx(0.7)
    assert payload[f"{prefix}/wrong_mean"] == pytest.approx(0.3)
    assert payload[f"{prefix}/gap"] == pytest.approx(0.4)
    assert payload[f"{prefix}/question_equal_auroc"] == pytest.approx(0.75)


def test_upload_bundle_logs_scalars_and_primary_figures(tmp_path: Path) -> None:
    analysis, checkpoint_manifest = _write_analysis_bundle(tmp_path)
    client = FakeSwanLab()

    result = upload_analysis_bundle(
        client=client,
        analysis_dir=analysis,
        figures_dir=tmp_path / "figures" / "primary",
        checkpoint_manifest=checkpoint_manifest,
    )

    assert result == {"n_steps": 1, "n_scalar_payloads": 1, "n_figures": 1}
    assert client.logged[0][1] == 1116
    assert any("figures/primary/h1_stage_curves" in data for data, _ in client.logged)


def test_hidden_upload_runner_uses_the_external_credential_file() -> None:
    runner = (Path(__file__).resolve().parents[1] / "scripts" / "run_swanlab_hidden_upload.sh").read_text(
        encoding="utf-8"
    )

    assert 'SWANLAB_ENV_FILE=${SWANLAB_ENV_FILE:-/data2/hjk/secrets/experiment_2e_swanlab.env}' in runner
    assert 'source "${SWANLAB_ENV_FILE}"' in runner
    assert 'SWANLAB_API_KEY must be defined' in runner
    assert 'python" -m experiment_2e.swanlab_upload' in runner
