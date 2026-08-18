from pathlib import Path


def test_vertical_package_imports_without_experiment_2e():
    import vertical

    assert vertical.__version__


def test_config_examples_are_present():
    assert Path("configs/qwen3_8b.example.yaml").is_file()
    assert Path("configs/mimo_7b.example.yaml").is_file()

