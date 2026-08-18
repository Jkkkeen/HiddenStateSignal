from pathlib import Path

import pytest

from vertical.config import VerticalConfig


def write_config(path: Path, *, datasets: str, representations: str = "  - last_s32\n") -> Path:
    path.write_text(
        "run_id: test_run\n"
        "output_root: ./runs/test_run\n"
        "datasets:\n"
        f"{datasets}"
        "representations:\n"
        f"{representations}"
        "question_equal: true\n",
        encoding="utf-8",
    )
    return path


def test_config_loads_dataset_and_preserves_adapter_options(tmp_path):
    path = write_config(
        tmp_path / "config.yaml",
        datasets=(
            "  - name: qwen_base\n"
            "    input_mode: raw\n"
            "    adapter: npz\n"
            "    source: ./data\n"
            "    model_family: qwen3\n"
            "    model_name: Qwen3-8B-Base\n"
            "    condition: base\n"
            "    decoder_layers: auto\n"
            "    hidden_dimension: auto\n"
            "    tensor_key: hidden_states\n"
        ),
    )

    config = VerticalConfig.from_yaml(path)

    assert config.run_id == "test_run"
    assert config.datasets[0].adapter == "npz"
    assert config.datasets[0].options["tensor_key"] == "hidden_states"
    assert config.representations == ("last_s32",)


def test_config_rejects_duplicate_dataset_names(tmp_path):
    dataset = (
        "  - name: duplicate\n"
        "    input_mode: raw\n"
        "    adapter: npz\n"
        "    source: ./data\n"
        "    model_family: mimo\n"
        "    model_name: MiMo-7B\n"
        "    condition: base\n"
        "    decoder_layers: auto\n"
        "    hidden_dimension: auto\n"
    )
    path = write_config(tmp_path / "config.yaml", datasets=dataset + dataset)

    with pytest.raises(ValueError, match="dataset names"):
        VerticalConfig.from_yaml(path)


def test_nonbase_condition_requires_family_base_dataset(tmp_path):
    path = write_config(
        tmp_path / "config.yaml",
        datasets=(
            "  - name: mimo_sft\n"
            "    input_mode: raw\n"
            "    adapter: numpy_directory\n"
            "    source: ./mimo_sft\n"
            "    model_family: mimo\n"
            "    model_name: MiMo-7B-SFT\n"
            "    condition: sft\n"
            "    decoder_layers: auto\n"
            "    hidden_dimension: auto\n"
        ),
    )

    with pytest.raises(ValueError, match="Base dataset"):
        VerticalConfig.from_yaml(path)


def test_config_rejects_unknown_representation(tmp_path):
    path = write_config(
        tmp_path / "config.yaml",
        datasets=(
            "  - name: qwen_base\n"
            "    input_mode: raw\n"
            "    adapter: npz\n"
            "    source: ./data\n"
            "    model_family: qwen3\n"
            "    model_name: Qwen3-8B-Base\n"
            "    condition: base\n"
            "    decoder_layers: auto\n"
            "    hidden_dimension: auto\n"
        ),
        representations="  - made_up\n",
    )

    with pytest.raises(ValueError, match="representation"):
        VerticalConfig.from_yaml(path)

