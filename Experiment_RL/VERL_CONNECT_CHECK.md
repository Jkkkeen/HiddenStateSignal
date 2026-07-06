# VERL Connect Check

This document records the pre-install checklist for connecting VERL to the H200 long-CoT RL experiments. It is intentionally conservative: do not install packages, pull Docker images, restart services, or modify H200 until each check is reviewed.

## Goal

Run the first RL baseline with VERL:

```text
algorithm: GRPO
actor: FSDP/FSDP2
rollout backend: vLLM
reward: final answer correctness only
rollout.n: 8
formal max_response_length: 16384
smoke max_response_length: 2048 -> 4096 -> 8192 -> 16384
```

The first smoke should prove the data, reward, rollout, and update loop work. It should not try to be a full training run.

## Non-Negotiable Storage Rule

All large files must live on `/data2`, not the H200 system disk.

Allowed locations:

```text
/data2/hjk/envs
/data2/hjk/cache
/data2/hjk/tmp
/data2/hjk/models
/data2/hjk/projects
/data2/hjk/checkpoints
/data2/docker
```

Avoid placing large data in:

```text
/var/lib/docker
/tmp
/root
/home/hjk/.cache
/home/ubuntu/.cache
```

Reason: the PDF guide shows `/` is the smaller system disk, while `/data2` is the large mounted data disk. VERL, vLLM, Docker images, Ray sessions, Hugging Face cache, checkpoints, and temporary files can become large.

## Current H200 Read-Only Findings

Checked without changing H200:

```text
GPU: NVIDIA H200
GPU memory: 143771 MiB
Driver: 580.95.05
CUDA shown by nvidia-smi: 13.0
OS: Ubuntu 22.04.5
/data2 available space: about 6.6T
Existing envs: hs_er, vllm_qwen3vl, fglrl
Docker command: installed
hjk docker permission: currently denied
sudo without password: not available for hjk
nvcc: not found
conda/mamba: not in PATH
```

Interpretation:

```text
1. H200 hardware is suitable.
2. Docker may be possible with admin account, but hjk currently cannot use the Docker daemon directly.
3. Direct Docker use is unsafe until Docker Root Dir is confirmed to be on /data2.
4. A separate /data2 Python environment is the safest non-Docker path.
```

## Route A: Docker Path

Docker is cleanest only if Docker storage is moved to `/data2`.

### A1. Read-Only Checks

Run only after deciding to inspect Docker with admin privileges:

```bash
docker --version
docker info | sed -n '1,120p'
docker info | grep -E 'Docker Root Dir|Runtimes|Default Runtime|nvidia'
nvidia-smi
```

If `docker info` requires sudo:

```bash
sudo docker info | sed -n '1,120p'
sudo docker info | grep -E 'Docker Root Dir|Runtimes|Default Runtime|nvidia'
```

Stop if Docker Root Dir is:

```text
/var/lib/docker
```

Do not pull VERL images before moving Docker storage.

### A2. Required Docker Storage Layout

Target:

```text
Docker Root Dir: /data2/docker
```

Changing this requires system-level edits, so do it only after explicit approval.

Expected config file:

```text
/etc/docker/daemon.json
```

Target content should include:

```json
{
  "data-root": "/data2/docker"
}
```

Then Docker service must be restarted. This is a system change and must not be done casually.

### A3. NVIDIA Container Runtime Check

After Docker Root Dir is safe, check:

```bash
docker info | grep -E 'Runtimes|Default Runtime|nvidia'
command -v nvidia-container-runtime
command -v nvidia-ctk
dpkg -l | grep -E 'nvidia-container|container-toolkit'
```

Only after this is OK, run a GPU container smoke:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

This command may pull an image, so run it only after Docker Root Dir is confirmed on `/data2`.

## Route B: Separate Python Environment Path

This is the safer first route if Docker permission or Docker data-root is not ready.

### B1. Directory Layout

Create everything under `/data2`:

```bash
mkdir -p /data2/hjk/envs
mkdir -p /data2/hjk/cache/pip
mkdir -p /data2/hjk/cache/huggingface
mkdir -p /data2/hjk/cache/torch
mkdir -p /data2/hjk/cache/ray
mkdir -p /data2/hjk/tmp
mkdir -p /data2/hjk/checkpoints/verl_qwen3vl
mkdir -p /data2/hjk/projects
```

Environment path:

```text
/data2/hjk/envs/verl_qwen3vl
```

VERL source path:

```text
/data2/hjk/projects/verl
```

### B2. Required Environment Variables

Set these before installing or running anything:

```bash
export HF_HOME=/data2/hjk/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data2/hjk/cache/huggingface/hub
export TRANSFORMERS_CACHE=/data2/hjk/cache/huggingface
export HF_DATASETS_CACHE=/data2/hjk/cache/huggingface/datasets
export TORCH_HOME=/data2/hjk/cache/torch
export PIP_CACHE_DIR=/data2/hjk/cache/pip
export TMPDIR=/data2/hjk/tmp
export RAY_TMPDIR=/data2/hjk/cache/ray
export WANDB_DIR=/data2/hjk/checkpoints/wandb
export VERL_CHECKPOINT_DIR=/data2/hjk/checkpoints/verl_qwen3vl
```

Also prefer using the existing local model snapshot:

```bash
export QWEN3VL_SNAP=/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b
```

### B3. Pre-Install Checks

Before creating the environment:

```bash
df -h / /data2
python3 --version
git --version
nvidia-smi
```

Expected:

```text
Python 3.10.x
Git available
H200 visible
/data2 has large free space
```

### B4. Install Strategy

Do not install VERL into existing envs:

```text
/data2/hjk/envs/hs_er
/data2/hjk/envs/vllm_qwen3vl
/data2/hjk/envs/fglrl
```

Create a fresh environment:

```bash
python3 -m venv /data2/hjk/envs/verl_qwen3vl
source /data2/hjk/envs/verl_qwen3vl/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Then install VERL according to the selected VERL version and its dependency constraints. Do this only after reviewing the current official install instructions.

## VERL Multimodal GRPO Fit

The target setup follows VERL's multimodal GRPO pattern:

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.n=8
data.image_key=images
custom_reward_function.path=/data2/hjk/projects/AI-HiddenState-ER/scripts/mathverse_answer_reward.py
custom_reward_function.name=compute_score
```

Known caveat:

```text
VERL has public Qwen2.5-VL multimodal GRPO examples. Qwen3-VL-8B-Thinking should be smoke-tested because it may need model/processor compatibility fixes.
```

## MathVerse Data Format

Prepare train/validation parquet under `/data2`:

```text
/data2/hjk/projects/AI-HiddenState-ER/rl_data/mathverse_grpo_train.parquet
/data2/hjk/projects/AI-HiddenState-ER/rl_data/mathverse_grpo_val.parquet
```

Recommended columns:

```text
data_source: "MathVerse"
prompt: VERL chat prompt/messages
images: list of local image paths or image payloads, depending on VERL recipe
reward_model: {"ground_truth": "A" | "B" | "C" | "D"}
extra_info: {"question_id": ..., "answer": ..., "split": ...}
```

Use local image paths under the project/data directory. Do not copy images to the system disk.

## GRPO-Answer Reward Function

First reward should be final answer correctness only:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred = extract_final_choice(solution_str)
    return 1.0 if pred == ground_truth else 0.0
```

Rules:

```text
1. Parse only final A/B/C/D answer.
2. Do not use hidden states.
3. Do not use option-logit margin yet.
4. Log parse failures and option distribution.
```

This isolates the GRPO-answer baseline.

## Smoke Configuration

Keep `rollout.n=8` from the beginning so group structure matches the formal experiment.

### Smoke 1: Minimal Chain Test

```text
rollout.n: 8
max_response_length: 2048
train_batch_size: 1 or 2 prompts
updates: 1
reward: answer correctness
```

Purpose:

```text
data loading -> image loading -> rollout -> answer parsing -> reward -> GRPO update -> checkpoint save
```

### Smoke 2: Longer Response Test

```text
rollout.n: 8
max_response_length: 4096 or 8192
train_batch_size: 2 or 4 prompts
updates: 1-3
reward: answer correctness
```

Purpose:

```text
check long response memory, truncation, parse rate, Ray temp files, checkpoint location
```

### Pilot

```text
rollout.n: 8
max_response_length: 16384
train questions: 200-500
eval questions: fixed 100-300
train_batch_size: increase gradually, starting from 4 or 8 prompts
reward: answer correctness
```

Do not jump directly to a large pilot until Smoke 1 and Smoke 2 pass.

## Monitoring Checklist

During each run, watch:

```text
nvidia-smi
df -h / /data2
du -sh /data2/hjk/cache/ray /data2/hjk/tmp /data2/hjk/checkpoints/verl_qwen3vl
tail -f logs/<run>.log
```

Metrics to record:

```text
train reward
train final-answer accuracy
parse rate
A/B/C/D distribution
mean response tokens
p90 response tokens
truncation rate
checkpoint path
Ray tmp path
system disk usage before/after
```

## Stop Conditions

Stop the run if:

```text
1. `/` system disk usage increases unexpectedly.
2. Ray writes large files outside `/data2`.
3. Docker Root Dir is `/var/lib/docker` and a Docker pull/run is about to happen.
4. vLLM cannot load Qwen3-VL processor/model.
5. parse rate is very low.
6. all rewards in a batch are identical for many consecutive updates.
```

## Next Actions

Recommended next sequence:

```text
1. Decide Docker route or separate Python environment route.
2. If Docker route: first move/confirm Docker Root Dir on /data2.
3. If Python env route: create /data2/hjk/envs/verl_qwen3vl with all cache variables pointing to /data2.
4. Prepare MathVerse VERL parquet.
5. Write MathVerse answer reward function.
6. Run Smoke 1 with rollout.n=8 and max_response_length=2048.
```
