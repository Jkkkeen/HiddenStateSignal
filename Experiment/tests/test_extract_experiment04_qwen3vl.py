from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_experiment04_qwen3vl import (  # noqa: E402
    cache_is_complete,
    reduce_hidden_states,
    save_rollout_cache,
)


def _hidden_states(tokens: int = 10, layers: int = 4, dim: int = 6) -> tuple[torch.Tensor, ...]:
    states = []
    base = torch.arange(tokens * dim, dtype=torch.float32).reshape(tokens, dim)
    for layer in range(layers):
        states.append((base + layer * 10.0).unsqueeze(0))
    return tuple(states)


def test_reduce_hidden_states_builds_three_representations_and_entropy() -> None:
    reduced = reduce_hidden_states(
        _hidden_states(),
        segment_start=0,
        segment_end=10,
        chunk_size=4,
        last_n=3,
    )

    assert reduced["full_vectors"].shape == (2, 3, 4, 6)
    assert reduced["partial_vectors"].shape == (1, 3, 4, 6)
    assert reduced["full_bounds"].tolist() == [[0, 4], [4, 8]]
    assert reduced["partial_bounds"].tolist() == [[8, 10]]
    assert reduced["token_entropy_full"].shape == (2, 4, 3, 4)
    assert reduced["token_entropy_partial"].shape == (1, 4, 3, 4)
    assert np.isnan(reduced["token_entropy_full"][:, 0, 2]).all()
    assert np.isfinite(reduced["token_entropy_full"][:, 1:, 2]).all()
    assert np.allclose(reduced["full_vectors"][0, 0, 0], _hidden_states()[0][0, 3])


def test_rollout_cache_is_atomic_and_validated(tmp_path: Path) -> None:
    reduced = reduce_hidden_states(
        _hidden_states(),
        segment_start=0,
        segment_end=10,
        chunk_size=4,
        last_n=3,
    )
    output = tmp_path / "rollout_0.npz"
    save_rollout_cache(
        output,
        reduced,
        metadata={"question_id": "q", "rollout_id": 0, "is_correct": True},
    )

    assert cache_is_complete(output, expected_layers=4)
    with np.load(output, allow_pickle=False) as cached:
        assert cached["question_id"].item() == "q"
        assert bool(cached["complete"].item())

    output.write_bytes(b"corrupt")
    assert not cache_is_complete(output, expected_layers=4)
