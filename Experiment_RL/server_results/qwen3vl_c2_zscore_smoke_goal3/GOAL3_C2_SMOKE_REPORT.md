# Goal 3: Qwen3-VL C2 Zscore Reward Smoke

## Status

Completed. Stage 2 C2 smoke is engineering-passed with the v1 batch hook enabled.

## Runs

1. `qwen3vl_c2_zscore_smoke_goal3.log`
   - First smoke proved Qwen3-VL option-logit probe model loads and training finishes.
   - It exposed that async v1 agent-loop reward path bypassed the original batch postprocess hook, so C2 zscore did not affect GRPO.

2. `qwen3vl_c2_zscore_smoke_goal3_v1hook.log`
   - Added v1 trainer-side batch hook before advantage computation.
   - Completed 3/3 training steps with exit marker `QWEN3VL_C2_ZSCORE_SMOKE_EXIT_CODE=0`.

## Key Results From v1hook Smoke

| step | nonzero B_gain groups | all-same answer groups with B ranking | score range | actor grad_norm | response mean | gen_s | adv_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.0 | 1.0 to 1.0 | 0.0000 | 1320.5 | 22.76 | 0.140 |
| 2 | 0.5 | 0.5 | 0.8 to 1.2 | 0.0450 | 2955.5 | 34.71 | 0.129 |
| 3 | 0.5 | 0.5 | 0.8 to 1.2 | 0.0552 | 1976.0 | 20.71 | 0.135 |

Interpretation: C2 bonus entered GRPO after the v1 hook. Step 2/3 show group-zscored option gain changed rollout ranking in all-correct groups, producing nonzero advantages and actor gradients.

## Caveats

- Response length is long enough for smoke (`>500`) but below the ideal `0.3 * 16384 ~= 4915` threshold.
- `response_length/clip_ratio=0.5` appears in this tiny smoke; monitor in pilot.
- vLLM logs one `EngineCore_DP0 died unexpectedly` at shutdown after training finished and exit marker was written. GPU and memory returned idle.

## Local Result Directory

`C:\Users\LENOVO\Desktop\A-G实验\AI-HiddenState\Experiment_RL\server_results\qwen3vl_c2_zscore_smoke_goal3`

Pulled files include both logs, reward scripts, runner script, patched `trainer_base.py`, and patched `reward_loop.py`.

## H200 Changes

```text
REMOTE_FILES
-rwxrwxr-x 1 hjk hjk  13K Jul  6 02:56 scripts/mathverse_qwen3vl_option_gain_reward.py
-rwxrwxr-x 1 hjk hjk  12K Jul  6 03:12 scripts/qwen3vl_c2_zscore_reward_manager.py
-rwxrwxr-x 1 hjk hjk 6.0K Jul  6 02:56 scripts/run_verl_qwen3vl_c2_zscore_smoke.sh
VERL_BACKUPS
-rw-rw-r-- 1 hjk hjk 16K Jul  6 02:39 /data2/hjk/projects/verl/verl/experimental/reward_loop/reward_loop.py.bak_c2_20260706_023915
-rw-rw-r-- 1 hjk hjk 75K Jul  6 03:10 /data2/hjk/projects/verl/verl/trainer/ppo/v1/trainer_base.py.bak_c2v1_
-rw-rw-r-- 1 hjk hjk 75K Jul  6 03:11 /data2/hjk/projects/verl/verl/trainer/ppo/v1/trainer_base.py.bak_c2v1_20260706_031135
PROCESSES
1207025       00:00  3752 bash -c cd /data2/hjk/projects/AI-HiddenState-ER && echo 'REMOTE_FILES'; ls -lh scripts/mathverse_qwen3vl_option_gain_reward.py scripts/qwen3vl_c2_zscore_reward_manager.py scripts/run_verl_qwen3vl_c2_zscore_smoke.sh; echo 'VERL_BACKUPS'; ls -lh /data2/hjk/projects/verl/verl/experimental/reward_loop/reward_loop.py.bak_c2_* /data2/hjk/projects/verl/verl/trainer/ppo/v1/trainer_base.py.bak_c2v1_* 2>/dev/null; echo 'PROCESSES'; ps -u hjk -o pid,etime,rss,args --sort=-rss | grep -E 'qwen3vl_c2|verl.trainer.main_ppo|ray::|vLLM|RewardLoopWorker|WorkerDict' | head -20 || true; echo 'GPU'; nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader; echo 'DF'; df -h / /data2
1207029       00:00  2500 grep -E qwen3vl_c2|verl.trainer.main_ppo|ray::|vLLM|RewardLoopWorker|WorkerDict
GPU
0 MiB, 0 %, 42
DF
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda3       434G   99G  313G  25% /
/dev/nvme1n1p1  7.0T  150G  6.5T   3% /data2
```
