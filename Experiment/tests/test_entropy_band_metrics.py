from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from entropy_band_metrics import (  # noqa: E402
    centered_coordinate_entropy,
    frozen_band_score,
    progress_bin_ids,
)


def test_centered_entropy_is_normalized_and_offset_invariant() -> None:
    values = np.asarray([[1.0, -1.0, 0.0], [2.0, 0.0, -2.0]])
    entropy = centered_coordinate_entropy(values)
    shifted = centered_coordinate_entropy(values + 10_000.0)

    assert np.allclose(entropy, shifted)
    assert np.all((entropy >= 0.0) & (entropy <= 1.0))


def test_progress_bin_six_is_relative_interval() -> None:
    bins = progress_bin_ids(100, 10)

    selected = np.flatnonzero(bins == 6)
    assert selected[0] == 59
    assert selected[-1] == 68
    assert selected.size == 10


def test_frozen_band_score_is_six_layer_arithmetic_mean() -> None:
    layer_values = {layer: 0.01 * layer for layer in range(14, 20)}

    score = frozen_band_score(layer_values, layers=tuple(range(14, 20)))

    assert np.isclose(score, np.mean(list(layer_values.values())))
