from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

from .adapters import get_adapter
from .audit import (
    calibrator_key,
    sha256_file,
    write_json_atomic,
    write_profile_partition,
)
from .calibrate import BaseCalibrator, fit_base_calibrator, save_calibrator
from .config import DatasetConfig, VerticalConfig
from .inspect import inspect_source
from .plotting import render_v01_report
from .profiles import PROFILE_COLUMNS
from .schema import VerticalRecord


DEFAULT_SMOKE_RECORDS = 16


def _approval(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        raise ValueError(f"smoke approval file is missing: {path}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"smoke approval is unreadable: {path}") from exc
    if value.get("status") != "passed" and value.get("passed") is not True:
        raise ValueError(f"smoke approval did not pass: {path}")
    return value


def _single_dataset_config(
    config: VerticalConfig,
    dataset: DatasetConfig,
) -> VerticalConfig:
    return replace(config, datasets=(dataset,))


def _iter_records(
    config: VerticalConfig,
    datasets: Iterable[DatasetConfig],
    *,
    record_limit: int | None,
) -> Iterable[VerticalRecord]:
    for dataset in datasets:
        single = _single_dataset_config(config, dataset)
        records = get_adapter(dataset.adapter).iter_records(dataset.source, single)
        yield from records if record_limit is None else islice(records, record_limit)


def _inspect_inputs(
    config: VerticalConfig,
    *,
    sample_limit: int,
) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    for dataset in config.datasets:
        single = _single_dataset_config(config, dataset)
        reports[dataset.name] = inspect_source(
            dataset.source,
            single,
            sample_limit=sample_limit,
        ).to_dict()
    return {
        "run_id": config.run_id,
        "passed": bool(reports) and all(report["passed"] for report in reports.values()),
        "datasets": reports,
    }


def _fit_calibrators(
    config: VerticalConfig,
    output_root: Path,
    *,
    record_limit: int | None,
) -> tuple[dict[str, BaseCalibrator], dict[str, Any]]:
    calibrators: dict[str, BaseCalibrator] = {}
    entries: dict[str, dict[str, Any]] = {}
    families = sorted({dataset.model_family for dataset in config.datasets})
    for family in families:
        base_datasets = [
            dataset
            for dataset in config.datasets
            if dataset.model_family == family and dataset.condition == "base"
        ]
        if not base_datasets:
            raise ValueError(f"missing Base calibration dataset for family={family}")
        for representation in config.representations:
            calibrator = fit_base_calibrator(
                _iter_records(
                    config,
                    base_datasets,
                    record_limit=record_limit,
                ),
                representation,
            )
            key = calibrator_key(family, representation)
            path = output_root / "calibrators" / f"{family}__{representation}.npz"
            save_calibrator(calibrator, path)
            calibrators[key] = calibrator
            entries[key] = {
                "model_family": family,
                "representation": representation,
                "n_records": calibrator.n_records,
                "n_chunks": calibrator.n_chunks,
                "input_sha256": calibrator.input_sha256,
                "artifact_sha256": sha256_file(path),
                "path": str(path),
            }
    audit = {
        "passed": bool(entries),
        "label_blind": True,
        "base_only": True,
        "calibrators": entries,
    }
    write_json_atomic(audit, output_root / "audit" / "calibration_audit.json")
    return calibrators, audit


def finalize_run(
    audits: Mapping[str, Path] | Sequence[Path],
    output_root: Path,
) -> dict[str, Any]:
    output_root = Path(output_root)
    named = (
        dict(audits)
        if isinstance(audits, Mapping)
        else {Path(path).stem: Path(path) for path in audits}
    )
    results: dict[str, dict[str, Any]] = {}
    for name, path_value in named.items():
        path = Path(path_value)
        if not path.is_file():
            raise ValueError(f"required audit is missing: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"required audit is unreadable: {path}") from exc
        passed = value.get("passed") is True or value.get("status") == "passed"
        results[str(name)] = {
            "passed": bool(passed),
            "path": str(path),
            "sha256": sha256_file(path),
        }
    gates = {
        "all_audits_present": len(results) == len(named) and bool(results),
        "all_audits_passed": bool(results)
        and all(result["passed"] for result in results.values()),
    }
    final = {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "audits": results,
    }
    write_json_atomic(final, output_root / "audit" / "final_audit.json")
    if not final["passed"]:
        raise RuntimeError(f"final audit failed: {final}")
    return final


def _manifest(
    config: VerticalConfig,
    config_path: Path,
    *,
    mode: str,
    record_limit: int | None,
    smoke_approval: Path | None,
) -> dict[str, Any]:
    datasets = [
        {
            "name": dataset.name,
            "adapter": dataset.adapter,
            "input_mode": dataset.input_mode,
            "source": str(dataset.source),
            "model_family": dataset.model_family,
            "model_name": dataset.model_name,
            "condition": dataset.condition,
        }
        for dataset in config.datasets
    ]
    value: dict[str, Any] = {
        "run_id": config.run_id,
        "mode": mode,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "record_limit_per_dataset": record_limit,
        "representations": list(config.representations),
        "stable_metrics": list(PROFILE_COLUMNS),
        "datasets": datasets,
    }
    if smoke_approval is not None:
        value["smoke_approval"] = {
            "path": str(smoke_approval),
            "sha256": sha256_file(smoke_approval),
        }
    return value


def run_pipeline(
    *,
    config_path: Path,
    output_root: Path,
    mode: str,
    smoke_approval: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"smoke", "formal"}:
        raise ValueError("pipeline mode must be 'smoke' or 'formal'")
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    if mode == "formal":
        _approval(smoke_approval)
    config = VerticalConfig.from_yaml(config_path)
    record_limit = DEFAULT_SMOKE_RECORDS if mode == "smoke" else None
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        _manifest(
            config,
            config_path,
            mode=mode,
            record_limit=record_limit,
            smoke_approval=smoke_approval,
        ),
        output_root / "manifest.json",
    )
    write_json_atomic(
        {"run_id": config.run_id, "mode": mode, "status": "running", "passed": False},
        output_root / "run_status.json",
    )

    try:
        input_audit = _inspect_inputs(
            config,
            sample_limit=record_limit or DEFAULT_SMOKE_RECORDS,
        )
        input_audit_path = output_root / "audit" / "input_audit.json"
        write_json_atomic(input_audit, input_audit_path)
        if not input_audit["passed"]:
            raise RuntimeError("input audit failed")

        calibrators, _ = _fit_calibrators(
            config,
            output_root,
            record_limit=record_limit,
        )
        profiles_path = output_root / "profiles" / "profiles.parquet"
        write_profile_partition(
            _iter_records(
                config,
                config.datasets,
                record_limit=record_limit,
            ),
            calibrators,
            profiles_path,
        )
        render_v01_report(
            profiles_path,
            output_root / "analysis",
            PROFILE_COLUMNS,
        )
        audit_paths = {
            "input": input_audit_path,
            "calibration": output_root / "audit" / "calibration_audit.json",
            "profiles": profiles_path.with_suffix(".audit.json"),
            "analysis": output_root / "analysis" / "analysis_audit.json",
        }
        final = finalize_run(audit_paths, output_root)
        result = {
            "run_id": config.run_id,
            "mode": mode,
            "status": "completed",
            "passed": bool(final["passed"]),
            "output_root": str(output_root),
        }
        write_json_atomic(result, output_root / "run_status.json")
        if mode == "smoke":
            approval = {
                "run_id": config.run_id,
                "status": "passed",
                "passed": True,
                "final_audit_sha256": sha256_file(
                    output_root / "audit" / "final_audit.json"
                ),
            }
            write_json_atomic(approval, output_root / "smoke_approval.json")
        return result
    except Exception as exc:
        write_json_atomic(
            {
                "run_id": config.run_id,
                "mode": mode,
                "status": "failed",
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            output_root / "run_status.json",
        )
        raise


def run_v01_pipeline(config_path: Path, output_root: Path) -> Mapping[str, Any]:
    return run_pipeline(
        config_path=config_path,
        output_root=output_root,
        mode="smoke",
    )
