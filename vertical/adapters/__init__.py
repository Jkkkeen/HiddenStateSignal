from __future__ import annotations

from .base import AdapterInspection, VerticalAdapter
from .custom_template import CustomTemplateAdapter
from .npz import NpzAdapter
from .numpy_directory import NumpyDirectoryAdapter


_ADAPTERS: dict[str, VerticalAdapter] = {
    "npz": NpzAdapter(),
    "numpy_directory": NumpyDirectoryAdapter(),
    "custom_template": CustomTemplateAdapter(),
}


def get_adapter(name: str) -> VerticalAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown adapter: {name}") from exc


__all__ = ["AdapterInspection", "VerticalAdapter", "get_adapter"]

