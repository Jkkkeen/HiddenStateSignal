from pathlib import Path

import numpy as np

from experiment_2e.calibrators import (
    fit_base_calibrators,
    load_calibrators,
    load_pooled_cache,
    write_pooled_cache,
)


def test_pooled_cache_roundtrip_and_rollout_equal_common(tmp_path: Path):
    first = np.array([[[1.0, 0.0]], [[3.0, 0.0]]])
    second = np.array([[[9.0, 0.0]]])
    paths = []
    for index, trajectory in enumerate((first, second)):
        path = tmp_path / f"r{index}.npz"
        write_pooled_cache(
            path,
            pooled={"mean_w128_s32": trajectory, "last_s32": trajectory},
            endpoints=np.arange(1, len(trajectory) + 1),
            progress=np.arange(1, len(trajectory) + 1) / len(trajectory),
            metadata={"rollout_id": f"r{index}"},
        )
        paths.append(path)
    loaded = load_pooled_cache(paths[0])
    np.testing.assert_allclose(loaded["mean_w128_s32"], first)

    calibrator = tmp_path / "calibrator.npz"
    audit = fit_base_calibrators(paths, calibrator)
    common, coordinate_mean, coordinate_sigma = load_calibrators(calibrator, "mean_w128_s32")
    # rollout equal: mean((1+3)/2, 9) = 5.5, not chunk-equal 13/3.
    np.testing.assert_allclose(common, [[5.5, 0.0]])
    # coordinate stats are chunk equal by the frozen V7 definition.
    np.testing.assert_allclose(coordinate_mean, [[13.0 / 3.0, 0.0]])
    assert coordinate_sigma[0, 0] > 0
    assert audit["n_rollouts"] == 2
