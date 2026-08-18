# Qwen3-1.7B Layer-Resolved Hidden Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract and analyze layer-resolved V1/V3/V4/V6/V7 profiles from the completed Qwen3-1.7B GRPO checkpoints and launch the audited full extraction in tmux.

**Architecture:** Adapt the frozen `256 x 8` generation files into stable rollout records, run response-only teacher-forced forwards with GPU-side chunk pooling, and feed the pooled trajectories to new pure-NumPy layer-profile reducers plus the existing all-layer V1-V9 reducers. Process one merged FSDP checkpoint at a time, write atomic checkpoint partitions and audits, then render question-equal profile heatmaps before a smoke-gated formal tmux launch.

**Tech Stack:** Python 3.11, PyTorch 2.8, Transformers 4.57, NumPy, pandas/pyarrow, matplotlib, pytest, veRL FSDP model merger, Bash, tmux, one NVIDIA H200.

## Global Constraints

- Use only each checkpoint's own saved native responses in this first run; do not add the fixed-response sensitivity arm.
- Do not retrain the model and do not regenerate held-out responses.
- Freeze response pooling to window 128, stride 32, representations `mean_w128_s32` and `last_s32`, and stages B1-B4.
- Permanently store pooled/profile values and scalars only; never store `response x token x layer x hidden_dim` tensors.
- Treat hidden-state index 0 as the embedding state and indices 1-28 as decoder destinations; use `relative_depth = layer_index / 28`.
- Fit V3 common components from step-0 caches without using correctness labels.
- Keep the existing V1-V9 all-layer output as a compatibility baseline without changing formulas in `metrics.py` or `reduction.py`.
- Smoke uses steps 0, 50, and 250, 16 questions, and all 8 rollout slots.
- Full extraction uses all 11 steps and may start only after smoke approval; both smoke and formal execution run in detached, explicitly named tmux sessions.
- Preserve unrelated dirty-worktree changes and commit only files belonging to the current task.

---

## File Map

- Create `Experiment_2E/experiment_2e/q3_layer_records.py`: adapt and audit Q3 generation JSONL plus held-out Parquet.
- Create `Experiment_2E/experiment_2e/q3_layer_forward.py`: tokenize, teacher-force, pool response hidden states on the model device, and reduce controls.
- Create `Experiment_2E/experiment_2e/vertical_profiles.py`: pure-NumPy V1/V3/V4/V6/V7 local arrays and stage/layer wide rows.
- Create `Experiment_2E/experiment_2e/q3_layer_extract.py`: resumable checkpoint extraction, calibrator fitting, aggregate compatibility output, model identity, and atomic audits.
- Create `Experiment_2E/experiment_2e/q3_layer_analysis.py`: question-equal summaries, within-question AUROC, depth summaries, plots, and run approval.
- Create `Experiment_2E/scripts/run_q3_layer_profiles.sh`: serial base/merge/extract/analyze pipeline for smoke and formal modes.
- Create `Experiment_2E/scripts/launch_q3_layer_profiles_tmux.sh`: guarded detached launcher.
- Create `Experiment_2E/tests/test_q3_layer_records.py`.
- Create `Experiment_2E/tests/test_q3_layer_forward.py`.
- Create `Experiment_2E/tests/test_vertical_profiles.py`.
- Create `Experiment_2E/tests/test_q3_layer_extract.py`.
- Create `Experiment_2E/tests/test_q3_layer_analysis.py`.
- Create `Experiment_2E/tests/test_q3_layer_profile_runner.py`.

### Task 1: Frozen Generation Adapter

**Files:**
- Create: `Experiment_2E/experiment_2e/q3_layer_records.py`
- Test: `Experiment_2E/tests/test_q3_layer_records.py`

**Interfaces:**
- Consumes: one `heldout/generations/<step>.jsonl`, the frozen held-out Parquet, `global_step`, `total_steps`, and optional `question_limit`.
- Produces: `render_manifest_prompt(row: pd.Series) -> str` and `load_q3_generation_records(...) -> pd.DataFrame` with stable rollout metadata.

- [ ] **Step 1: Write failing adapter tests**

```python
import json

import pandas as pd
import pytest

from experiment_2e.q3_layer_records import load_q3_generation_records


def _manifest(path):
    rows = []
    for index, content in enumerate(("question zero", "question one")):
        rows.append({
            "prompt": [{"role": "user", "content": content}],
            "reward_model": {"ground_truth": str(index), "style": "rule"},
            "extra_info": {
                "question_id": f"q{index}",
                "difficulty": float(index + 1),
                "prompt_hash": f"hash{index}",
            },
        })
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_adapter_maps_manifest_order_and_slots(tmp_path):
    manifest = tmp_path / "heldout.parquet"
    generation = tmp_path / "50.jsonl"
    _manifest(manifest)
    rows = []
    for question in range(2):
        prompt = f"user\nquestion {'zero' if question == 0 else 'one'}\nassistant\n"
        for slot in range(8):
            rows.append({
                "input": prompt,
                "output": f"response {question}-{slot}",
                "gts": str(question),
                "score": float(slot == 0),
                "reward": float(slot == 0),
                "acc": slot == 0,
                "format_correct": True,
                "parse_correct": True,
                "step": 50,
            })
    generation.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    frame = load_q3_generation_records(generation, manifest, global_step=50, total_steps=250)
    assert len(frame) == 16
    assert frame["question_id"].tolist()[:8] == ["q0"] * 8
    assert frame["rollout_slot"].tolist() == list(range(8)) * 2
    assert frame.iloc[8]["rollout_id"] == "step050/q1/r00"
    assert frame["training_progress"].unique().tolist() == [0.2]


def test_adapter_rejects_prompt_or_count_mismatch(tmp_path):
    manifest = tmp_path / "heldout.parquet"
    generation = tmp_path / "0.jsonl"
    _manifest(manifest)
    bad = [{"input": "wrong", "output": "x", "step": 0}] * 16
    generation.write_text("".join(json.dumps(x) + "\n" for x in bad), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt mismatch"):
        load_q3_generation_records(generation, manifest, global_step=0, total_steps=250)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_records.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'experiment_2e.q3_layer_records'`.

- [ ] **Step 3: Implement the strict adapter**

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROLLOUTS_PER_QUESTION = 8


def render_manifest_prompt(row: pd.Series) -> str:
    messages = list(row["prompt"])
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("Q3 held-out manifest must contain one user message")
    return f"user\n{messages[0]['content']}\nassistant\n"


def load_q3_generation_records(
    generation_file: Path,
    heldout_file: Path,
    *,
    global_step: int,
    total_steps: int,
    question_limit: int | None = None,
) -> pd.DataFrame:
    manifest = pd.read_parquet(heldout_file)
    if question_limit is not None:
        if question_limit < 1:
            raise ValueError("question_limit must be positive")
        manifest = manifest.head(question_limit)
    with generation_file.open("r", encoding="utf-8") as handle:
        generated = [json.loads(line) for line in handle if line.strip()]
    expected = len(manifest) * ROLLOUTS_PER_QUESTION
    if len(generated) < expected or (question_limit is None and len(generated) != expected):
        raise ValueError(f"expected {expected} generation rows, found {len(generated)}")
    generated = generated[:expected]
    rows = []
    checkpoint = f"step{global_step:03d}"
    for question_index, manifest_row in manifest.reset_index(drop=True).iterrows():
        extra = dict(manifest_row["extra_info"])
        reward_model = dict(manifest_row["reward_model"])
        expected_prompt = render_manifest_prompt(manifest_row)
        group = generated[
            question_index * ROLLOUTS_PER_QUESTION : (question_index + 1) * ROLLOUTS_PER_QUESTION
        ]
        for slot, generated_row in enumerate(group):
            if generated_row.get("input") != expected_prompt:
                raise ValueError(f"prompt mismatch question={question_index} slot={slot}")
            if int(generated_row.get("step", global_step)) != global_step:
                raise ValueError(f"generation step mismatch at question={question_index} slot={slot}")
            question_id = str(extra["question_id"])
            rows.append({
                "checkpoint": checkpoint,
                "global_step": int(global_step),
                "training_progress": float(global_step / total_steps),
                "question_id": question_id,
                "rollout_id": f"{checkpoint}/{question_id}/r{slot:02d}",
                "rollout_slot": slot,
                "prompt": expected_prompt,
                "response": str(generated_row["output"]),
                "ground_truth": str(generated_row.get("gts", reward_model["ground_truth"])),
                "is_correct": bool(generated_row.get("acc", False)),
                "answer_reward": float(generated_row.get("reward", generated_row.get("score", 0.0))),
                "format_correct": bool(generated_row.get("format_correct", False)),
                "parse_correct": bool(generated_row.get("parse_correct", False)),
                "difficulty": extra.get("difficulty"),
                "prompt_hash": extra.get("prompt_hash"),
            })
    frame = pd.DataFrame(rows)
    if frame.duplicated(["question_id", "rollout_slot"]).any():
        raise ValueError("duplicate question/rollout slots")
    return frame
```

- [ ] **Step 4: Run adapter tests**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_records.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the adapter**

```bash
git add Experiment_2E/experiment_2e/q3_layer_records.py Experiment_2E/tests/test_q3_layer_records.py
git commit -m "feat: adapt q3 heldout generations"
```

### Task 2: Response-Only Pooled Forward

**Files:**
- Create: `Experiment_2E/experiment_2e/q3_layer_forward.py`
- Test: `Experiment_2E/tests/test_q3_layer_forward.py`

**Interfaces:**
- Consumes: tokenizer, causal LM, prompt/response strings, and frozen window/stride.
- Produces: `tokenize_response(...)`, `pool_hidden_states_device(...)`, and `teacher_forced_pooled_forward(...) -> dict[str, Any]` containing both pooled representations and scalar token controls.

- [ ] **Step 1: Write failing token-boundary and pooling-equivalence tests**

```python
import numpy as np
import pytest
import torch

from experiment_2e.q3_layer_forward import pool_hidden_states_device, tokenize_response
from experiment_2e.reduction import pool_token_hidden, trajectory_endpoints


class RoundTripTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(value) for value in ids)


def test_tokenize_response_preserves_separate_boundary():
    prompt_ids, response_ids = tokenize_response(RoundTripTokenizer(), "ab", " cd")
    assert prompt_ids == [97, 98]
    assert response_ids == [32, 99, 100]


def test_tokenize_response_rejects_non_roundtrip():
    tokenizer = RoundTripTokenizer()
    tokenizer.decode = lambda *args, **kwargs: "different"
    with pytest.raises(ValueError, match="round-trip"):
        tokenize_response(tokenizer, "ab", "cd")


def test_device_pool_matches_numpy_reference():
    token_hidden = np.arange(9 * 3 * 4, dtype=np.float32).reshape(9, 3, 4)
    states = tuple(torch.tensor(token_hidden[:, layer, :])[None, :, :] for layer in range(3))
    endpoints = trajectory_endpoints(9, window=4, stride=2)
    actual = pool_hidden_states_device(states, start=0, stop=9, endpoints=endpoints, window=4)
    expected = pool_token_hidden(token_hidden, endpoints, window=4)
    assert np.allclose(actual["mean_w128_s32"], expected["mean_w128_s32"])
    assert np.allclose(actual["last_s32"], expected["last_s32"])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_forward.py -q`

Expected: collection fails because `q3_layer_forward` does not exist.

- [ ] **Step 3: Implement tokenization and device-side pooling**

```python
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .reduction import trajectory_endpoints


def tokenize_response(tokenizer: Any, prompt: str, response: str) -> tuple[list[int], list[int]]:
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    decoded = tokenizer.decode(
        response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not prompt_ids or not response_ids:
        raise ValueError("prompt and response must both contain tokens")
    if decoded != response:
        raise ValueError("response token round-trip mismatch")
    return prompt_ids, response_ids


def pool_hidden_states_device(
    hidden_states: Iterable[Any],
    *,
    start: int,
    stop: int,
    endpoints: np.ndarray,
    window: int = 128,
) -> dict[str, np.ndarray]:
    import torch

    endpoint_tensor = torch.as_tensor(endpoints, dtype=torch.long)
    means = []
    lasts = []
    for state in hidden_states:
        response = state[0, start:stop].detach().float()
        device_endpoints = endpoint_tensor.to(response.device)
        starts = torch.clamp(device_endpoints - window, min=0)
        prefix = torch.cat(
            [torch.zeros((1, response.shape[1]), device=response.device), response.cumsum(dim=0)],
            dim=0,
        )
        denominator = (device_endpoints - starts).to(response.dtype)[:, None]
        means.append(((prefix[device_endpoints] - prefix[starts]) / denominator).cpu())
        lasts.append(response[device_endpoints - 1].cpu())
    return {
        "mean_w128_s32": torch.stack(means, dim=1).numpy(),
        "last_s32": torch.stack(lasts, dim=1).numpy(),
    }
```

- [ ] **Step 4: Add the teacher-forced wrapper and a fake-model test**

Add `teacher_forced_pooled_forward(model, tokenizer, prompt, response, window=128, stride=32)` that:

```python
def teacher_forced_pooled_forward(model, tokenizer, prompt, response, *, window=128, stride=32):
    import torch

    prompt_ids, response_ids = tokenize_response(tokenizer, prompt, response)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    start = len(prompt_ids)
    stop = start + len(response_ids)
    prediction_positions = torch.arange(start - 1, stop - 1, device=device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=prediction_positions,
        )
    endpoints = trajectory_endpoints(len(response_ids), window=window, stride=stride)
    pooled = pool_hidden_states_device(
        output.hidden_states, start=start, stop=stop, endpoints=endpoints, window=window
    )
    logits = output.logits[0].float()
    targets = torch.tensor(response_ids, dtype=torch.long, device=device)
    log_normalizer = torch.logsumexp(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = log_normalizer - (probabilities * logits).sum(dim=-1)
    logprob = logits.gather(1, targets[:, None]).squeeze(1) - log_normalizer
    final_hidden = output.hidden_states[-1][0, start:stop].detach().float()
    result = {
        **pooled,
        "endpoints": endpoints,
        "progress": endpoints.astype(np.float32) / len(response_ids),
        "token_logprob": logprob.cpu().numpy(),
        "policy_entropy": entropy.cpu().numpy(),
        "hidden_norm": torch.linalg.vector_norm(final_hidden, dim=1).cpu().numpy(),
        "prompt_token_count": len(prompt_ids),
        "response_token_count": len(response_ids),
        "n_hidden_states": len(output.hidden_states),
    }
    return result
```

The fake-model test must return three deterministic hidden states and response-position logits, then assert response token count, hidden-state count, endpoint count, and finite controls.

- [ ] **Step 5: Run forward tests and the existing pooling tests**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_forward.py Experiment_2E/tests/test_reduction.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the forward module**

```bash
git add Experiment_2E/experiment_2e/q3_layer_forward.py Experiment_2E/tests/test_q3_layer_forward.py
git commit -m "feat: pool q3 response hidden states on device"
```

### Task 3: Layer-Local Vertical Profiles

**Files:**
- Create: `Experiment_2E/experiment_2e/vertical_profiles.py`
- Test: `Experiment_2E/tests/test_vertical_profiles.py`

**Interfaces:**
- Consumes: one pooled trajectory shaped `(K, L+1, D)`, progress `(K,)`, metadata, representation, and V3 base common `(L+1, D)`.
- Produces: `local_profile_arrays(layers, base_common) -> dict[str, np.ndarray]` and `reduce_vertical_profiles(...) -> pd.DataFrame` with one row per stage/layer.

- [ ] **Step 1: Write exact formula and layer-index tests**

```python
import numpy as np

from experiment_2e.vertical_profiles import local_profile_arrays, reduce_vertical_profiles


def test_local_profiles_have_frozen_layer_semantics():
    layers = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 3.0]])
    common = np.zeros_like(layers)
    out = local_profile_arrays(layers, common)
    assert len(out["v6_raw_activation_entropy"]) == 4
    assert np.isnan(out["v1_raw_update_norm"][0])
    assert np.allclose(out["v1_raw_update_norm"][1:], [1.0, 1.0, 2.0])
    assert np.isnan(out["v4_layer_update_turning_angle"][:2]).all()
    assert np.allclose(out["v4_layer_update_turning_angle"][2:], [np.pi / 2, np.pi / 2])


def test_stage_reduction_is_chunk_median_and_wide():
    base = np.arange(4 * 3, dtype=float).reshape(4, 3) + 1.0
    trajectory = np.stack([base, base * 2, base * 3, base * 4], axis=0)
    frame = reduce_vertical_profiles(
        trajectory,
        np.array([0.10, 0.30, 0.60, 1.00]),
        representation="mean_w128_s32",
        metadata={"checkpoint": "step000", "question_id": "q", "rollout_id": "r"},
        base_common=np.zeros_like(base),
    )
    assert len(frame) == 4 * 4
    assert frame.columns.isin([
        "v1_raw_update_norm", "v1_relative_update_norm",
        "v3_demean_state_angle", "v4_layer_update_turning_angle",
        "v6_raw_activation_entropy", "v7_layer_difference_entropy",
    ]).sum() == 6
    assert frame["relative_depth"].drop_duplicates().tolist() == [0.0, 1/3, 2/3, 1.0]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest Experiment_2E/tests/test_vertical_profiles.py -q`

Expected: collection fails because `vertical_profiles` does not exist.

- [ ] **Step 3: Implement vectorized local arrays**

```python
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import EPS, NORM_FLOOR, energy_entropy_rows, stage_mask


PROFILE_COLUMNS = (
    "v1_raw_update_norm",
    "v1_relative_update_norm",
    "v3_demean_state_angle",
    "v4_layer_update_turning_angle",
    "v6_raw_activation_entropy",
    "v7_layer_difference_entropy",
)


def _adjacent_angles(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    output = np.full(len(values), np.nan, dtype=float)
    valid = (norms[1:] >= NORM_FLOOR) & (norms[:-1] >= NORM_FLOOR)
    cosine = np.sum(values[1:] * values[:-1], axis=1) / (norms[1:] * norms[:-1] + EPS)
    output[1:][valid] = np.arccos(np.clip(cosine[valid], -1.0, 1.0))
    return output


def local_profile_arrays(layers: np.ndarray, base_common: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(layers, dtype=np.float32)
    common = np.asarray(base_common, dtype=np.float32)
    if values.ndim != 2 or common.shape != values.shape:
        raise ValueError("layers and base_common must share shape (L+1,D)")
    updates = np.diff(values, axis=0)
    raw = np.linalg.norm(updates, axis=1)
    relative = raw / (np.linalg.norm(values[:-1], axis=1) + EPS)
    v1_raw = np.r_[np.nan, raw]
    v1_relative = np.r_[np.nan, relative]
    v3 = _adjacent_angles(values - common)
    update_angles = _adjacent_angles(updates)
    v4 = np.full(len(values), np.nan, dtype=float)
    v4[2:] = update_angles[1:]
    v6 = energy_entropy_rows(values, center=False, zero_value=np.nan)
    v7 = np.r_[np.nan, energy_entropy_rows(updates, center=True, zero_value=np.nan)]
    return dict(zip(PROFILE_COLUMNS, (v1_raw, v1_relative, v3, v4, v6, v7), strict=True))
```

- [ ] **Step 4: Implement stage medians and coverage columns**

`reduce_vertical_profiles` must compute local arrays for every selected chunk,
take `np.nanmedian` across chunks for each layer, and return rows containing all
metadata plus `representation`, `stage`, `layer_index`, `relative_depth`, the six
profile columns, `profile_chunk_count`, and `profile_coverage_count_<metric>`.
Use exactly this selection:

```python
selected = np.flatnonzero(stage_mask(np.asarray(progress, dtype=float), stage))
arrays = [local_profile_arrays(trajectory[index], base_common) for index in selected]
```

For an empty stage, emit all layers with NaN metric values and zero coverage.

- [ ] **Step 5: Add compatibility tests against existing aggregate metrics**

For deterministic random layers, assert that flattening valid local values
reproduces the existing per-chunk values used by `v1_layer_update_norm`,
`v3_state_angle_demean`, `v4_layer_update_turning`, `v6_raw_activation_entropy`,
and `v7_entropies` to `1e-6` for every matching mean/median key.

- [ ] **Step 6: Run profile and existing metric tests**

Run: `python -m pytest Experiment_2E/tests/test_vertical_profiles.py Experiment_2E/tests/test_metrics.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the profile reducer**

```bash
git add Experiment_2E/experiment_2e/vertical_profiles.py Experiment_2E/tests/test_vertical_profiles.py
git commit -m "feat: add layer-local vertical profiles"
```

### Task 4: Resumable Checkpoint Extractor

**Files:**
- Create: `Experiment_2E/experiment_2e/q3_layer_extract.py`
- Test: `Experiment_2E/tests/test_q3_layer_extract.py`

**Interfaces:**
- Consumes: one generation file, held-out manifest, base or merged HF model path, optional source checkpoint directory, work/output roots, and calibrator path.
- Produces: pooled caches, controls, `layer_profiles_stepNNN.parquet`, `vertical_metrics_stepNNN.parquet`, `model_identity_stepNNN.json`, and `extraction_audit_stepNNN.json`.

- [ ] **Step 1: Write failing tests for atomic output, identity, and audit counts**

Create synthetic pooled caches with 4 hidden-state positions and two
representations. Test these pure helpers:

```python
from experiment_2e.q3_layer_extract import (
    atomic_parquet,
    checkpoint_audit,
    parameter_sample_sha256,
    reduce_checkpoint_caches,
)


def test_atomic_parquet_leaves_no_tmp(tmp_path):
    path = tmp_path / "rows.parquet"
    atomic_parquet(pd.DataFrame({"x": [1, 2]}), path)
    assert pd.read_parquet(path)["x"].tolist() == [1, 2]
    assert not path.with_suffix(".parquet.tmp").exists()


def test_checkpoint_audit_enforces_expected_profile_rows():
    audit = checkpoint_audit(
        profiles=profile_frame,
        controls=control_frame,
        expected_rollouts=2,
        expected_hidden_states=4,
    )
    assert audit["passed"] is True
    assert audit["n_profile_rows"] == 2 * 2 * 4 * 4
```

Also test that changing one model parameter changes
`parameter_sample_sha256`, an existing finalized partition is not rewritten,
and a trained identity equal to the base signature is rejected.

- [ ] **Step 2: Run extractor tests and verify failure**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_extract.py -q`

Expected: collection fails because `q3_layer_extract` does not exist.

- [ ] **Step 3: Implement atomic files and deterministic identity**

```python
def atomic_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def parameter_sample_sha256(model):
    digest = hashlib.sha256()
    named = list(model.named_parameters())
    indices = sorted({0, len(named) // 3, 2 * len(named) // 3, len(named) - 1})
    for index in indices:
        name, parameter = named[index]
        sample = parameter.detach().reshape(-1)[:4096].float().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(sample.tobytes())
    return digest.hexdigest()
```

Hash sorted source `model_world_size_*_rank_*.pt` files and sorted merged
`*.safetensors` files with `common.sha256_file`. Write both manifests, global
step, model type, layer count, hidden size, and parameter sample atomically.

- [ ] **Step 4: Implement forward cache creation and restart behavior**

For each adapted record:

1. derive `cache_name(rollout_id)`;
2. skip only if both cache and controls JSON exist and their stored rollout ID
   matches;
3. call `teacher_forced_pooled_forward`;
4. require 29 hidden states in the real Q3 run;
5. write pooled arrays through `write_pooled_cache` and write stage controls
   through `write_json_atomic`;
6. release outputs, run `gc.collect()`, and empty the CUDA cache.

Use existing `stage_controls` semantics, but pass the forward result's token
log probabilities, policy entropy, hidden norms, response token count, and
trajectory point count. Metadata must include model name, checkpoint, step,
question/rollout/slot, correctness, rewards, difficulty, prompt hash, and token
counts.

- [ ] **Step 5: Implement base calibration and checkpoint reduction**

When `global_step == 0` and the calibrator is absent, call
`fit_base_calibrators(sorted(cache_dir.glob("*.npz")), calibrator_path)`.
For every cache and representation:

```python
common, coordinate_mean, coordinate_sigma = load_calibrators(calibrator_path, representation)
profiles.append(reduce_vertical_profiles(
    cache[representation], cache["progress"], representation=representation,
    metadata=metadata_with_controls, base_common=common,
))
aggregate_rows.extend(reduce_vertical(
    cache[representation], cache["progress"], representation=representation,
    metadata=metadata_with_controls, base_common=common,
    coordinate_common=coordinate_mean, coordinate_sigma=coordinate_sigma,
))
```

Concatenate profiles to one wide DataFrame, aggregate rows to one long
DataFrame, then atomically write both partitions. A finalized passing audit is
the only checkpoint-completion marker.

- [ ] **Step 6: Implement audit gates and CLI**

The audit must require:

- exact unique-rollout count;
- `n_profile_rows = rollouts * 2 representations * 4 stages * 29 layers`;
- exactly 29 layer indices and relative depths from 0 through 1;
- expected structural NaNs only: V1/V3/V7 at layer 0 and V4 at layers 0-1;
- finite V6 at every layer and positive coverage for all defined cells;
- controls for every rollout and stage;
- no file matching `*token_hidden*` under the output root;
- model identity passed and calibrator audit present.

Reset CUDA peak-memory statistics immediately before the first forward. Add
`elapsed_seconds`, `seconds_per_rollout`, and `peak_gpu_memory_bytes` to every
checkpoint audit after reduction completes.

CLI arguments are:

```text
--run-id --generation-file --heldout-file --model-path --tokenizer-path
--model-name --global-step --total-steps --work-dir --output-dir
--calibrator-path [--source-checkpoint-dir] [--base-parameter-sample]
[--question-limit] [--keep-pooled-cache]
```

Print the final audit JSON and exit nonzero when `passed` is false.

- [ ] **Step 7: Run extractor unit tests and adjacent regression tests**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_extract.py Experiment_2E/tests/test_hidden_extract.py Experiment_2E/tests/test_vertical_profiles.py -q`

Expected: all selected tests pass.

- [ ] **Step 8: Commit the extractor**

```bash
git add Experiment_2E/experiment_2e/q3_layer_extract.py Experiment_2E/tests/test_q3_layer_extract.py
git commit -m "feat: extract resumable q3 layer profiles"
```

### Task 5: Question-Equal Analysis And Heatmaps

**Files:**
- Create: `Experiment_2E/experiment_2e/q3_layer_analysis.py`
- Test: `Experiment_2E/tests/test_q3_layer_analysis.py`

**Interfaces:**
- Consumes: finalized layer-profile partitions and source generation files.
- Produces: policy/outcome/AUROC/depth Parquet summaries, PNG figures, `analysis_audit.json`, and smoke approval JSON when requested.

- [ ] **Step 1: Write failing aggregation and depth-summary tests**

Build a synthetic profile table with four questions, four rollouts, two
checkpoints, four layers, both labels, and one metric. Assert:

```python
policy = summarize_policy_profiles(frame, "v1_relative_update_norm")
assert policy["n_questions"].min() == 4

outcome = summarize_outcome_profiles(frame, "v1_relative_update_norm")
assert set(outcome["correct_minus_wrong"]) == {1.0}

auc = summarize_layer_auc(frame, "v1_relative_update_norm")
assert set(auc["question_equal_auc"]) == {1.0}

depth = summarize_depth_change(policy, metric="v1_relative_update_norm", base_step=0)
assert depth.loc[depth.global_step == 0, "depth_mass"].eq(0.0).all()
assert depth.loc[depth.global_step > 0, "peak_depth"].between(0.0, 1.0).all()
```

- [ ] **Step 2: Run analysis tests and verify failure**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_analysis.py -q`

Expected: collection fails because `q3_layer_analysis` does not exist.

- [ ] **Step 3: Implement question-equal summaries**

Policy summaries first average rollouts within
`question_id/checkpoint/stage/representation/layer`, then average questions.
Outcome differences first average correct and wrong rollouts within a question,
drop non-mixed questions, subtract wrong from correct, then average questions.
AUROC uses existing `analysis.within_question_auc_table` separately within each
checkpoint/stage/representation/layer group and averages question AUROCs.

Write exact columns for sample counts: `n_questions`, `n_rollouts`, and
`n_mixed_questions`. Do not pool rollout pairs across questions.

- [ ] **Step 4: Implement depth-change summaries**

For each metric/representation/stage/checkpoint, merge its question-equal mean
profile with step 0 by `layer_index`. Define `change = abs(current - base)` and:

```python
mass = np.nan_to_num(change, nan=0.0)
total = mass.sum()
peak_depth = relative_depth[np.argmax(mass)] if total > 0 else np.nan
depth_center = (relative_depth * mass).sum() / total if total > 0 else np.nan
depth_spread = np.sqrt(((relative_depth - depth_center) ** 2 * mass).sum() / total) if total > 0 else np.nan
early_mass = mass[relative_depth <= 1/3].sum() / total if total > 0 else 0.0
middle_mass = mass[(relative_depth > 1/3) & (relative_depth <= 2/3)].sum() / total if total > 0 else 0.0
late_mass = mass[relative_depth > 2/3].sum() / total if total > 0 else 0.0
```

- [ ] **Step 5: Implement plots and run audit**

For each of the six stored profile columns, two representations, and four
stages, render:

- layer-by-checkpoint policy heatmap;
- base/early/best/final profile curves;
- correct-minus-wrong profile curves;
- layer-wise AUROC heatmap.

Select `best` solely by mean `acc` in the frozen generation files. Use the
ordered checkpoints 0,25,...,250 and retain absent columns as explicit gaps.
Save figures at 160 DPI and close every figure.

The analysis audit records input hashes, checkpoint list, metric list, figure
count, nonblank files (`size > 10_000`), and expected question count. Smoke
approval has `status: passed` only for exact steps `[0, 50, 250]`, 16 questions,
all extraction audits passed, and all expected policy heatmaps nonblank.
It also records measured smoke wall time and
`projected_formal_hours = smoke_wall_seconds * 22528 / 384 / 3600` as the
initial extraction ETA; the report labels this as a linear projection that is
updated from each completed formal checkpoint.

Expose a CLI with `--profiles-dir`, `--generation-dir`, `--output-dir`,
`--expected-steps` as a comma-separated list, `--expected-questions`, and
optional `--approval-output`. The CLI prints `analysis_audit.json` and exits
nonzero when the audit fails.

- [ ] **Step 6: Run analysis tests and existing statistical tests**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_analysis.py Experiment_2E/tests/test_analysis.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the analysis module**

```bash
git add Experiment_2E/experiment_2e/q3_layer_analysis.py Experiment_2E/tests/test_q3_layer_analysis.py
git commit -m "feat: analyze q3 layer profiles"
```

### Task 6: Serial Runner And Tmux Gates

**Files:**
- Create: `Experiment_2E/scripts/run_q3_layer_profiles.sh`
- Create: `Experiment_2E/scripts/launch_q3_layer_profiles_tmux.sh`
- Create: `Experiment_2E/tests/test_q3_layer_profile_runner.py`

**Interfaces:**
- Consumes: `MODE=smoke|formal`, frozen H200 paths, and smoke approval for formal mode.
- Produces: serial merged/extracted checkpoints, analysis outputs, persistent logs, and a detached tmux session.

- [ ] **Step 1: Write failing runner contract tests**

```python
from pathlib import Path


RUNNER = Path("Experiment_2E/scripts/run_q3_layer_profiles.sh")
LAUNCHER = Path("Experiment_2E/scripts/launch_q3_layer_profiles_tmux.sh")


def test_runner_freezes_smoke_and_formal_cohorts():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'STEPS=(0 50 250)' in text
    assert 'STEPS=(0 25 50 75 100 125 150 175 200 225 250)' in text
    assert 'QUESTION_LIMIT=16' in text
    assert 'verl.model_merger merge --backend fsdp' in text
    assert 'rm -rf -- "${MERGED}"' in text
    assert 'smoke_approval.json' in text


def test_formal_launcher_requires_passing_smoke_and_tmux():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'MODE=${MODE:-smoke}' in text
    assert 'q3_layer_profiles_formal' in text
    assert "p.get('status') == 'passed'" in text
    assert 'tmux new-session -d' in text
    assert "experiment_2e.q3_layer_extract" in text
    assert "nvidia-smi --query-compute-apps" in text
```

- [ ] **Step 2: Run runner tests and verify failure**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_profile_runner.py -q`

Expected: both tests fail because the scripts do not exist.

- [ ] **Step 3: Implement the serial runner**

Freeze these defaults:

```bash
PROJECT_ROOT=/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814
VERL_ROOT=/data2/hjk/projects/verl_q3_1p7b_base_20260814
ENV_ROOT=/data2/hjk/envs/verl_qwen3vl_py311
MODEL_PATH=/data2/hjk/models/Qwen3-1.7B-Base
RUN_ROOT=/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814
CKPT_ROOT=/data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814
HELDOUT_FILE=/data2/hjk/data/experiment_2e/q3_1p7b_base_simplerl_strict_v2_seed20260814/simplerl_heldout_256.parquet
```

Smoke writes under `${RUN_ROOT}/layer_profiles_smoke`; formal writes under
`${RUN_ROOT}/layer_profiles_formal`. For each nonzero step, merge
`${CKPT_ROOT}/global_step_${step}/actor` to
`${OUTPUT_ROOT}/work/merged/step${step}`, run the extractor, require its passing
audit, then resolve and validate that the merged path is inside
`${OUTPUT_ROOT}/work/merged` before `rm -rf -- "${MERGED}"`.

Run analysis after all requested checkpoints. Smoke writes
`${OUTPUT_ROOT}/smoke_approval.json`; formal requires that file from the smoke
root before doing any merge.

Use this exact control flow, with the path variables above exported before the
block:

```bash
case "${MODE}" in
  smoke)
    STEPS=(0 50 250)
    QUESTION_LIMIT=16
    OUTPUT_ROOT="${RUN_ROOT}/layer_profiles_smoke"
    ;;
  formal)
    STEPS=(0 25 50 75 100 125 150 175 200 225 250)
    QUESTION_LIMIT=256
    OUTPUT_ROOT="${RUN_ROOT}/layer_profiles_formal"
    SMOKE_APPROVAL="${RUN_ROOT}/layer_profiles_smoke/smoke_approval.json"
    "${ENV_ROOT}/bin/python" - "${SMOKE_APPROVAL}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("status") == "passed", p
PY
    ;;
  *) echo "MODE must be smoke or formal" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_ROOT}/work/merged" "${OUTPUT_ROOT}/metrics"
for step in "${STEPS[@]}"; do
  model="${MODEL_PATH}"
  source_args=()
  base_sample_args=()
  if (( step > 0 )); then
    MERGED="${OUTPUT_ROOT}/work/merged/step${step}"
    if [[ ! -f "${MERGED}/config.json" ]]; then
      cd "${VERL_ROOT}"
      "${ENV_ROOT}/bin/python" -m verl.model_merger merge --backend fsdp \
        --local_dir "${CKPT_ROOT}/global_step_${step}/actor" \
        --target_dir "${MERGED}" --use_cpu_initialization
    fi
    model="${MERGED}"
    source_args=(--source-checkpoint-dir "${CKPT_ROOT}/global_step_${step}/actor")
    base_sample_args=(--base-parameter-sample "${OUTPUT_ROOT}/metrics/model_identity_step000.json")
  fi
  "${ENV_ROOT}/bin/python" -m experiment_2e.q3_layer_extract \
    --run-id "q3_1p7b_native_layer_profiles" \
    --generation-file "${RUN_ROOT}/heldout/generations/${step}.jsonl" \
    --heldout-file "${HELDOUT_FILE}" --model-path "${model}" \
    --tokenizer-path "${MODEL_PATH}" --model-name "Qwen3-1.7B-Base" \
    --global-step "${step}" --total-steps 250 \
    --work-dir "${OUTPUT_ROOT}/work" --output-dir "${OUTPUT_ROOT}/metrics" \
    --calibrator-path "${OUTPUT_ROOT}/metrics/base_calibrators.npz" \
    --question-limit "${QUESTION_LIMIT}" "${source_args[@]}" "${base_sample_args[@]}"
  if (( step > 0 )); then
    merged_root=$(realpath -m "${OUTPUT_ROOT}/work/merged")
    merged_path=$(realpath -m "${MERGED}")
    case "${merged_path}" in
      "${merged_root}"/*) rm -rf -- "${MERGED}" ;;
      *) echo "refusing unsafe merged-model cleanup: ${merged_path}" >&2; exit 4 ;;
    esac
  fi
done

IFS=,; expected_steps="${STEPS[*]}"; unset IFS
analysis_args=()
if [[ "${MODE}" == smoke ]]; then
  analysis_args=(--approval-output "${OUTPUT_ROOT}/smoke_approval.json")
fi
"${ENV_ROOT}/bin/python" -m experiment_2e.q3_layer_analysis \
  --profiles-dir "${OUTPUT_ROOT}/metrics" \
  --generation-dir "${RUN_ROOT}/heldout/generations" \
  --output-dir "${OUTPUT_ROOT}/analysis" \
  --expected-steps "${expected_steps}" --expected-questions "${QUESTION_LIMIT}" \
  "${analysis_args[@]}"
```

- [ ] **Step 4: Implement the guarded tmux launcher**

The launcher chooses sessions `q3_layer_profiles_smoke` and
`q3_layer_profiles_formal`, refuses an existing session, checks active veRL or
extractor processes, and checks H200 compute PIDs. Formal mode parses the smoke
approval in Python and requires `p.get('status') == 'passed'`.

Launch exactly:

```bash
tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_ROOT}' && MODE='${MODE}' bash scripts/run_q3_layer_profiles.sh 2>&1 | tee '${LOG}'"
```

Print the session, log, and attach command after launch.

- [ ] **Step 5: Run runner tests and the full focused suite**

Run: `python -m pytest Experiment_2E/tests/test_q3_layer_profile_runner.py Experiment_2E/tests/test_q3_layer_records.py Experiment_2E/tests/test_q3_layer_forward.py Experiment_2E/tests/test_vertical_profiles.py Experiment_2E/tests/test_q3_layer_extract.py Experiment_2E/tests/test_q3_layer_analysis.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Run neighboring Experiment 2E regression tests**

Run: `python -m pytest Experiment_2E/tests/test_metrics.py Experiment_2E/tests/test_reduction.py Experiment_2E/tests/test_hidden_extract.py Experiment_2E/tests/test_analysis.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit runners and tests**

```bash
git add Experiment_2E/scripts/run_q3_layer_profiles.sh Experiment_2E/scripts/launch_q3_layer_profiles_tmux.sh Experiment_2E/tests/test_q3_layer_profile_runner.py
git commit -m "feat: launch q3 layer profiles in tmux"
```

### Task 7: H200 Smoke, Audit, And Formal Tmux Launch

**Files:**
- Verify only; do not modify source unless a smoke failure is reproduced by a new local test first.

**Interfaces:**
- Consumes: committed implementation, frozen remote inputs, idle H200, and passing local tests.
- Produces: passing remote smoke approval and a live detached formal tmux extraction session.

- [ ] **Step 1: Sync only committed task files to the isolated Q3 project**

Use `scp` for the five Python modules and two scripts. Verify remote SHA256 for
each file equals the local SHA256 before execution. Do not synchronize unrelated
dirty files.

- [ ] **Step 2: Run a remote import and syntax preflight**

Run remotely:

```bash
cd /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814
PYTHONPATH=. /data2/hjk/envs/verl_qwen3vl_py311/bin/python -m py_compile \
  experiment_2e/q3_layer_records.py experiment_2e/q3_layer_forward.py \
  experiment_2e/vertical_profiles.py experiment_2e/q3_layer_extract.py \
  experiment_2e/q3_layer_analysis.py
```

Expected: exit 0 with no output.

- [ ] **Step 3: Launch the smoke in tmux**

Run remotely:

```bash
cd /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814
MODE=smoke bash scripts/launch_q3_layer_profiles_tmux.sh
tmux has-session -t q3_layer_profiles_smoke
```

Expected: launcher prints the persistent log path and `tmux has-session` exits
0.

- [ ] **Step 4: Monitor smoke to a terminal status**

Check the tmux pane, log tail, GPU process, checkpoint audits, disk usage, and
`smoke_approval.json` without stopping the session. A successful smoke must
finish all 384 forwards, produce passing audits for steps 0/50/250, render
nonblank heatmaps, and record an ETA based on measured elapsed time.

If smoke fails, retain artifacts, reproduce the failure in the smallest local
test, commit the correction, resync changed task files, and relaunch with a new
explicit smoke result directory.

- [ ] **Step 5: Inspect smoke artifacts directly**

Read representative profile rows for layers 0, 1, 2, and 28 from both
representations and all stages. Confirm structural NaNs, finite defined values,
relative depths, labels, token controls, checkpoint identity hashes, and source
counts. Open at least one policy heatmap and one AUROC heatmap and verify they
are nonblank and contain columns 0, 50, and 250.

- [ ] **Step 6: Launch the full formal extraction in tmux**

Only after the smoke approval says `status: passed`, run remotely:

```bash
cd /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814
MODE=formal bash scripts/launch_q3_layer_profiles_tmux.sh
tmux has-session -t q3_layer_profiles_formal
```

Expected: the detached session exists, the log records mode `formal`, the
process command contains `experiment_2e.q3_layer_extract`, and the first
checkpoint begins under `layer_profiles_formal`.

- [ ] **Step 7: Report the live formal run**

Report the tmux session name, log path, result root, current checkpoint/rollout,
smoke-measured ETA, GPU memory, and disk headroom. Do not claim the experiment
is complete while the formal tmux session is still active.

---

## Completion Criteria

Implementation is complete when all focused and neighboring tests pass, the
remote three-checkpoint smoke approval is `passed`, representative layer
profiles and figures have been visually/data audited, and the full 11-checkpoint
extraction is demonstrably running in detached tmux on H200 with a persistent
log and result directory.
