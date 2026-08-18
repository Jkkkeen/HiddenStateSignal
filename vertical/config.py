from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


KNOWN_REPRESENTATIONS = ("mean_w128_s32", "last_s32")
KNOWN_ADAPTERS = ("npz", "numpy_directory", "custom_template")
KNOWN_INPUT_MODES = ("raw", "pooled")
DATASET_FIELDS = {
    "name",
    "input_mode",
    "adapter",
    "source",
    "model_family",
    "model_name",
    "condition",
    "decoder_layers",
    "hidden_dimension",
}


def _auto_int(value: Any, field_name: str) -> int | None:
    if value in (None, "auto"):
        return None
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{field_name} must be positive or 'auto'")
    return converted


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    input_mode: str
    adapter: str
    source: Path
    model_family: str
    model_name: str
    condition: str
    decoder_layers: int | None
    hidden_dimension: int | None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> DatasetConfig:
        missing = sorted(DATASET_FIELDS - raw.keys())
        if missing:
            raise ValueError(f"dataset is missing required fields: {missing}")
        dataset = cls(
            name=str(raw["name"]),
            input_mode=str(raw["input_mode"]),
            adapter=str(raw["adapter"]),
            source=Path(str(raw["source"])),
            model_family=str(raw["model_family"]),
            model_name=str(raw["model_name"]),
            condition=str(raw["condition"]),
            decoder_layers=_auto_int(raw["decoder_layers"], "decoder_layers"),
            hidden_dimension=_auto_int(raw["hidden_dimension"], "hidden_dimension"),
            options={key: value for key, value in raw.items() if key not in DATASET_FIELDS},
        )
        dataset.validate()
        return dataset

    def validate(self) -> None:
        for field_name in ("name", "model_family", "model_name", "condition"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"dataset {field_name} must be nonempty")
        if not str(self.source):
            raise ValueError("dataset source must be nonempty")
        if self.input_mode not in KNOWN_INPUT_MODES:
            raise ValueError(f"unknown input_mode: {self.input_mode}")
        if self.adapter not in KNOWN_ADAPTERS:
            raise ValueError(f"unknown adapter: {self.adapter}")


@dataclass(frozen=True)
class VerticalConfig:
    run_id: str
    output_root: Path
    datasets: tuple[DatasetConfig, ...]
    representations: tuple[str, ...]
    question_equal: bool = True
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> VerticalConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("config root must be a mapping")
        datasets_raw = raw.get("datasets")
        if not isinstance(datasets_raw, list) or not datasets_raw:
            raise ValueError("config requires a nonempty datasets list")
        if not all(isinstance(item, dict) for item in datasets_raw):
            raise ValueError("every dataset entry must be a mapping")
        representations_raw = raw.get("representations")
        if not isinstance(representations_raw, list) or not representations_raw:
            raise ValueError("config requires a nonempty representations list")
        config = cls(
            run_id=str(raw.get("run_id", "")),
            output_root=Path(str(raw.get("output_root", ""))),
            datasets=tuple(DatasetConfig.from_mapping(item) for item in datasets_raw),
            representations=tuple(str(item) for item in representations_raw),
            question_equal=bool(raw.get("question_equal", True)),
            config_path=path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be nonempty")
        if not str(self.output_root):
            raise ValueError("output_root must be nonempty")
        names = [dataset.name for dataset in self.datasets]
        if len(set(names)) != len(names):
            raise ValueError("dataset names must be unique")
        if len(set(self.representations)) != len(self.representations):
            raise ValueError("representations must be unique")
        unknown = sorted(set(self.representations) - set(KNOWN_REPRESENTATIONS))
        if unknown:
            raise ValueError(f"unknown representation: {unknown}")
        base_families = {
            dataset.model_family for dataset in self.datasets if dataset.condition == "base"
        }
        missing_base = sorted(
            {
                dataset.model_family
                for dataset in self.datasets
                if dataset.condition != "base" and dataset.model_family not in base_families
            }
        )
        if missing_base:
            raise ValueError(
                "every non-Base condition requires a Base dataset for its model family: "
                f"{missing_base}"
            )

