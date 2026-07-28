from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from roll_mathverse_vllm import existing_rollout_ids  # noqa: E402


def test_resume_tracks_rollout_ids_instead_of_any_question_line(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    rows = [
        {"question_id": "partial", "rollout_id": 0},
        {"question_id": "partial", "rollout_id": 2},
        *({"question_id": "complete", "rollout_id": rid} for rid in range(8)),
    ]
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    existing = existing_rollout_ids(output)

    assert existing["partial"] == {0, 2}
    assert existing["complete"] == set(range(8))
