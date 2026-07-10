from __future__ import annotations

import math

from scripts.qwen3vl_actor_option_probe import (
    option_gain_from_actor_logits,
    probe_positions_for_lengths,
)


def test_probe_positions_use_prompt_last_token_for_zero_frac() -> None:
    positions = probe_positions_for_lengths(
        total_lengths=[10],
        response_lengths=[4],
        fracs=[0.0, 0.25, 0.5, 0.9],
    )

    assert positions == [[5, 6, 7, 9]]


def test_option_gain_from_actor_logits_uses_late_margin_deltas() -> None:
    result = option_gain_from_actor_logits(
        option_logits=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.1, 0.0],
            [0.0, 0.9, 0.2, 0.0],
            [0.0, 1.2, 0.2, 0.1],
        ],
        correct="B",
        clip_value=0.4,
        reward_start_index=1,
    )

    assert [round(value, 6) for value in result.margins] == [0.0, 0.2, 0.7, 1.0]
    assert math.isclose(result.gain, 0.35)
