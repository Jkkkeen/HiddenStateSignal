from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from vertical.config import VerticalConfig
from vertical.schema import VerticalRecord

from .base import AdapterInspection, axis_order, resolve_dataset
from .numpy_directory import NumpyDirectoryAdapter


class CustomTemplateAdapter:
    """Strict manifest adapter for source-specific column and axis mappings."""

    name = "custom_template"

    def _delegate_config(self, source: Path, config: VerticalConfig) -> VerticalConfig:
        dataset = resolve_dataset(config, source, adapter=self.name)
        axis_order(dataset)
        for key in ("manifest", "tensor_path_column"):
            if not dataset.options.get(key):
                raise ValueError(f"custom_template adapter requires {key}")
        delegated = type(dataset)(
            name=dataset.name,
            input_mode=dataset.input_mode,
            adapter="numpy_directory",
            source=dataset.source,
            model_family=dataset.model_family,
            model_name=dataset.model_name,
            condition=dataset.condition,
            decoder_layers=dataset.decoder_layers,
            hidden_dimension=dataset.hidden_dimension,
            options=dataset.options,
        )
        return type(config)(
            run_id=config.run_id,
            output_root=config.output_root,
            datasets=(delegated,),
            representations=config.representations,
            question_equal=config.question_equal,
            config_path=config.config_path,
        )

    def iter_records(self, source: Path, config: VerticalConfig) -> Iterator[VerticalRecord]:
        delegated = self._delegate_config(source, config)
        yield from NumpyDirectoryAdapter().iter_records(source, delegated)

    def inspect(
        self,
        source: Path,
        config: VerticalConfig,
        sample_limit: int,
    ) -> AdapterInspection:
        delegated = self._delegate_config(source, config)
        report = NumpyDirectoryAdapter().inspect(source, delegated, sample_limit)
        report.adapter = self.name
        return report

