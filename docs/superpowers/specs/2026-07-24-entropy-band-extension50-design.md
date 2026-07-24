# Entropy-Band Confirm120 Candidate Extension: First-50 Design

## 1. Purpose

Continue `2Dimension-03` without changing its frozen entropy feature or eligibility gate. The initial frozen 800 new MathVerse questions produced only 84 eligible `2+2` questions, so the next action is to freeze another 500 candidates and generate only the first 50-question batch as a concurrency smoke while `hao.lin`'s idle Jupyter kernel remains untouched.

## 2. Fixed Current State

- Prior Thinking questions excluded: 500.
- Existing new candidates generated: ranks 1-800 under seed `20260725`.
- Existing rollout rows: 6,400, with 8 rollouts per question.
- Eligible primary questions: 84; target: 120; deficit: 36.
- Strict `3+3` sensitivity questions: 44.
- Remaining valid unused MCQ questions: 795.
- Existing candidate order SHA-256: `a6be58071e4cba09297c7d2446d0fc632958f5759e7c0c8ca491d02e8da4eb80`.
- Current unrelated GPU process: user `hao.lin`, PID `696866`, approximately 18.5 GiB, launched by Cursor/Jupyter. It must not be signaled, paused, inspected beyond process metadata, or modified.

## 3. Candidate Extension Freeze

Recompute the complete valid MathVerse candidate order with the original rules and seed `20260725`:

1. exclude all question IDs present in the old Thinking-500 rollout file;
2. require an A/B/C/D ground-truth answer and a non-empty query;
3. sort by `SHA256("20260725:<question_id>")`, breaking ties by question ID;
4. verify that ranks 1-800 exactly match the existing frozen `candidate_ids_all800.txt` before writing anything;
5. freeze ranks 801-1300 as `candidate_ids_extension500.txt`;
6. split the extension into ten immutable 50-question files in rank order.

The extension audit records source hashes, prior candidate hash, full extension hash, batch hashes, rank bounds, seed, and `entropy_used_for_selection=false`. Existing candidate files and rollout files are never rewritten during the freeze step.

## 4. First-50 Generation Run

Only extension ranks 801-850 are generated in this run. The generator appends to the existing raw rollout JSONL with `--resume`, using the unchanged long-response setting:

```text
model                       Qwen3-VL-8B-Thinking
rollouts/question           8
temperature                 0.7
top_p                       0.95
max_tokens                  16384
max_model_len               32768
gpu_memory_utilization      0.65
question batch              extension ranks 801-850 only
```

The run uses a new tmux session and a new log file. It does not attach to or alter any existing tmux session. Before generation, the launcher records the current GPU PIDs and memory usage as provenance, but it does not require the GPU to be empty because the user explicitly approved concurrent execution at the reduced vLLM memory fraction.

## 5. Ten-Minute Safety Observation

After launch, observe for ten minutes without blocking the tmux job. Check at launch, approximately five minutes, and approximately ten minutes:

- vLLM process exists and the log advances;
- no CUDA OOM, engine initialization failure, or repeated request failure;
- total GPU memory remains below 135 GiB;
- `hao.lin` PID `696866` remains alive and unchanged;
- at least one question has begun or completed generation;
- output JSONL grows and contains no duplicate `(question_id, rollout_id)` keys.

If vLLM fails to initialize or memory exceeds the safety bound, terminate only the new experiment tmux/process tree and leave PID `696866` untouched. Do not automatically retry with another memory fraction.

## 6. Post-Batch Audit

After all 50 questions finish:

1. rerun rollout labeling over the combined append-only JSONL;
2. rerun eligibility selection using the frozen order ranks 1-1300;
3. report total generated questions, clean/truncated rollouts, eligible `2+2` count, strict `3+3` count, and remaining deficit to 120;
4. do not run entropy hidden extraction unless the frozen cohort has reached exactly the first 120 eligible questions.

The likely outcome after one 50-question batch is still below 120. That is not a failure; it is the approved concurrency and throughput smoke. Subsequent 50-question batches require a separate go/no-go decision based on this run's stability and observed generation time.

## 7. Success Criteria

The first-50 run succeeds when:

- the original 800-order validation passes;
- extension ranks 801-1300 are frozen before generation;
- exactly the first 50 extension questions receive 8 rollout IDs each;
- no old rollout row or candidate file is overwritten;
- the unrelated Jupyter kernel remains alive;
- labeling and entropy-blind eligibility audit complete;
- no entropy value is computed or inspected before the formal 120-question manifest is ready.

## 8. Non-Goals

- No Experiment 0 multi-resolution refresh.
- No all-layer hidden-state forward.
- No entropy calculation on the current partial 84-question cohort.
- No relaxation of the `2+2` primary eligibility rule.
- No RL training.
- No automatic continuation through all 500 extension questions after the first batch.
