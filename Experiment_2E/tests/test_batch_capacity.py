import json
from pathlib import Path

from experiment_2e.batch_capacity import build_audit


def write_candidate(
    root: Path,
    batch: int,
    *,
    status: int = 0,
    steps: int = 7,
    throughput: float = 400.0,
    used_mib: int = 100_000,
    failure: str = "",
) -> None:
    candidate = root / f"batch_{batch}"
    candidate.mkdir(parents=True)
    lines = []
    for offset in range(1, steps + 1):
        step = 3348 + offset
        lines.append(
            f"training/global_step:{step} - timing_s/step:{10 + offset}.0 - "
            f"perf/throughput:{throughput} - "
            "actor/perf/max_memory_allocated_gb:80.0 - "
            "actor/perf/max_memory_reserved_gb:90.0"
        )
    if failure:
        lines.append(failure)
    (candidate / "train.log").write_text("\n".join(lines), encoding="utf-8")
    (candidate / "exit_status.txt").write_text(str(status), encoding="utf-8")
    (candidate / "gpu_telemetry.csv").write_text(
        "timestamp,memory.total [MiB],memory.used [MiB],memory.free [MiB],utilization.gpu [%]\n"
        f"now,143771,{used_mib},{143771-used_mib},100\n",
        encoding="utf-8",
    )


def test_build_audit_reports_stable_and_useful_maxima(tmp_path: Path) -> None:
    write_candidate(tmp_path, 2, throughput=350.0)
    write_candidate(tmp_path, 4, throughput=410.0)
    write_candidate(tmp_path, 8, throughput=405.0)

    audit = build_audit(tmp_path)

    assert audit["maximum_stable_batch"] == 8
    assert audit["maximum_useful_batch"] == 4
    assert audit["candidates"]["4"]["measured_steps"] == [3351, 3352, 3353, 3354, 3355]


def test_oom_and_short_runs_are_not_stable(tmp_path: Path) -> None:
    write_candidate(tmp_path, 2)
    write_candidate(tmp_path, 4, status=1, failure="CUDA out of memory")
    write_candidate(tmp_path, 8, steps=4)

    audit = build_audit(tmp_path)

    assert audit["candidates"]["4"]["failure_signatures"] == ["cuda_oom"]
    assert audit["candidates"]["4"]["stable"] is False
    assert audit["candidates"]["8"]["stable"] is False
    assert audit["maximum_stable_batch"] == 2


def test_ten_percent_headroom_is_required(tmp_path: Path) -> None:
    write_candidate(tmp_path, 2, used_mib=129_393)
    write_candidate(tmp_path, 4, used_mib=129_395)

    audit = build_audit(tmp_path, candidates=(2, 4))

    assert audit["candidates"]["2"]["stable"] is True
    assert audit["candidates"]["4"]["stable"] is False


def test_audit_is_json_serializable(tmp_path: Path) -> None:
    write_candidate(tmp_path, 2)
    audit = build_audit(tmp_path, candidates=(2,))
    assert json.loads(json.dumps(audit))["source_step"] == 3348
