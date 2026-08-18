from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vertical.pooling import pool_response_states, progress_from_endpoints, trajectory_endpoints


RESPONSE_TOKENS = 512
HIDDEN_DIMENSION = 16
DECODER_LAYERS = 4


def hidden_values(
    *,
    seed: int,
    hidden_states: int,
    condition_shift: float,
    outcome_shift: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(
        0.0,
        0.2,
        size=(RESPONSE_TOKENS, hidden_states, HIDDEN_DIMENSION),
    ).astype(np.float32)
    token_progress = np.linspace(0.0, 1.0, RESPONSE_TOKENS, dtype=np.float32)[:, None, None]
    layer_progress = np.linspace(0.0, 1.0, hidden_states, dtype=np.float32)[None, :, None]
    values += token_progress * layer_progress
    values += condition_shift * layer_progress
    values += outcome_shift * token_progress * (layer_progress > 0.5)
    return values


def metadata(
    *,
    question_id: str,
    rollout_id: str,
    is_correct: bool,
    num_decoder_layers: int,
    layer_kind: list[str],
) -> dict[str, object]:
    return {
        "record_id": f"{question_id}:{rollout_id}",
        "question_id": question_id,
        "rollout_id": rollout_id,
        "is_correct": is_correct,
        "response_token_count": RESPONSE_TOKENS,
        "num_decoder_layers": num_decoder_layers,
        "layer_kind": layer_kind,
        "reward": float(is_correct),
        "difficulty": "synthetic",
    }


def write_raw_family(
    root: Path,
    *,
    condition_shift: float,
    include_mtp: bool,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    hidden_states = DECODER_LAYERS + 1 + int(include_mtp)
    layer_kind = ["embedding"] + ["decoder"] * DECODER_LAYERS
    if include_mtp:
        layer_kind.append("mtp")
    for question_index in range(2):
        for rollout_index, is_correct in enumerate((False, True)):
            question_id = f"q{question_index + 1}"
            rollout_id = f"r{rollout_index}"
            values = hidden_values(
                seed=100 * question_index + rollout_index,
                hidden_states=hidden_states,
                condition_shift=condition_shift,
                outcome_shift=0.15 if is_correct else 0.0,
            )
            info = metadata(
                question_id=question_id,
                rollout_id=rollout_id,
                is_correct=is_correct,
                num_decoder_layers=DECODER_LAYERS,
                layer_kind=layer_kind,
            )
            np.savez(
                root / f"{question_id}_{rollout_id}.npz",
                hidden_states=values,
                metadata_json=np.asarray(json.dumps(info)),
            )


def write_pooled_example(root: Path, raw_path: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with np.load(raw_path, allow_pickle=False) as data:
        raw = data["hidden_states"]
        metadata_json = data["metadata_json"]
    endpoints = trajectory_endpoints(len(raw))
    pooled = pool_response_states(raw, endpoints)
    np.savez(
        root / "record.npz",
        **pooled,
        endpoints=endpoints.astype(np.int32),
        progress=progress_from_endpoints(endpoints, len(raw)).astype(np.float32),
        metadata_json=metadata_json,
    )


def dataset_entry(
    *,
    name: str,
    source: Path,
    family: str,
    model_name: str,
    condition: str,
) -> dict[str, object]:
    return {
        "name": name,
        "input_mode": "raw",
        "adapter": "npz",
        "source": source.resolve().as_posix(),
        "model_family": family,
        "model_name": model_name,
        "condition": condition,
        "decoder_layers": "auto",
        "hidden_dimension": "auto",
        "tensor_key": "hidden_states",
        "metadata_json_key": "metadata_json",
        "axis_order": "token,layer,dim",
    }


def write_configs(output: Path) -> None:
    qwen_config = {
        "run_id": "synthetic_qwen",
        "output_root": (output / "runs" / "synthetic_qwen").resolve().as_posix(),
        "datasets": [
            dataset_entry(
                name="qwen3_base",
                source=output / "qwen3_base_raw",
                family="qwen3",
                model_name="Qwen3-Synthetic-Base",
                condition="base",
            )
        ],
        "representations": ["mean_w128_s32", "last_s32"],
        "question_equal": True,
    }
    mimo_config = {
        "run_id": "synthetic_mimo",
        "output_root": (output / "runs" / "synthetic_mimo").resolve().as_posix(),
        "datasets": [
            dataset_entry(
                name="mimo_base",
                source=output / "mimo_base_raw",
                family="mimo",
                model_name="MiMo-Synthetic-Base",
                condition="base",
            ),
            dataset_entry(
                name="mimo_sft",
                source=output / "mimo_sft_raw",
                family="mimo",
                model_name="MiMo-Synthetic-SFT",
                condition="sft",
            ),
        ],
        "representations": ["mean_w128_s32", "last_s32"],
        "question_equal": True,
    }
    (output / "synthetic_qwen.yaml").write_text(
        yaml.safe_dump(qwen_config, sort_keys=False),
        encoding="utf-8",
    )
    (output / "synthetic_mimo.yaml").write_text(
        yaml.safe_dump(mimo_config, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create small vertical-toolkit fixtures")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_raw_family(output / "qwen3_base_raw", condition_shift=0.0, include_mtp=False)
    write_raw_family(output / "mimo_base_raw", condition_shift=0.0, include_mtp=True)
    write_raw_family(output / "mimo_sft_raw", condition_shift=0.25, include_mtp=True)
    write_pooled_example(
        output / "pooled_example",
        next((output / "qwen3_base_raw").glob("*.npz")),
    )
    write_configs(output)
    print(json.dumps({"output": str(output), "status": "created"}, indent=2))


if __name__ == "__main__":
    main()
