# RL03 Stage A Watcher Design

## Objective

Poll the H200 every ten minutes for completion of the immutable RL03 Stage A0 v2 run. Once A0 finishes, copy every run artifact to the local repository, independently inspect the smoke gate, and start the RL03 Stage A1 full frozen-policy audit only when the engineering smoke gate passes.

This automation must not install packages, stop processes, clean Ray, or alter system state on the H200. It must fail closed: an incomplete, ambiguous, or failed audit never launches A1.

## Fixed Runs

- Remote project: `/data2/hjk/projects/AI-HiddenState-ER`
- A0 run: `rl03_stage_a0_v2_smoke32_seed20260713`
- A0 tmux: `rl03_stage_a0_v2_smoke32`
- Local result root: `Experiment_RL/server_results/rl03_stage_a0_v2_smoke32_seed20260713`
- A1 run: `rl03_stage_a1_v2_full_seed20260713`
- A1 tmux: `rl03_stage_a1_v2_full`

The old RL02 answer-only GRPO scripts also use the name `A1`; they are unrelated and must never be called by this watcher.

## Components

### In-session poller

The active Codex execution session performs one read-only remote check every ten minutes until A0 reaches a terminal state. It does not create a script, register a Windows task, or install a persistent service.

Each poll:

1. Reads the remote A0 `exit_code.txt`, `scores/run_summary.json`, `audit/metric_summary.json`, and process/tmux state over `ssh h200`.
2. Reports a timestamped status line in the active execution session.
3. Returns without modifying the server while A0 is incomplete.
4. Copies the completed A0 run and log into a temporary local directory, then promotes the directory only after required files are present.
5. Re-runs or validates the analysis locally. Failure to execute the local analyzer is terminal and does not fall back to an unverified launch decision.
6. Evaluates the engineering smoke gate.
7. Starts A1 once, only if the smoke gate passes and no A1 session or terminal A1 artifact already exists.
8. Ends the polling loop after A1 is confirmed started or after a terminal A0 failure is recorded.

### A0 completion and smoke gate

A0 is complete only when all of the following exist:

- `exit_code.txt`
- `finished_at.txt`
- `scores/run_summary.json`
- `scores/request_scores.jsonl`
- `audit/metric_summary.json`
- `audit/STAGE_A_AUDIT_REPORT.md`

The engineering smoke gate passes only when:

- shell exit code is zero;
- analyzer `pipeline.passed` is true;
- completion and finite-score rates are at least 0.99;
- score-failure rate is at most 0.01;
- within-question gain variance is nonzero;
- the audit has no context-overflow or pipeline-failure indication;
- expected score and audit artifacts are readable locally.

The 32-question A0 scientific label is reported but does not replace this engineering gate. A formal process-signal decision is deferred to A1 because A0 is an engineering smoke sample.

### Failure handling

If A0 fails, the watcher does not launch A1. It writes a local failure report containing the failed predicates, relevant log tail, resource peaks, and analyzer gate inputs. Any repair is scoped to the identified failure and requires a reviewed A0 rerun; the watcher never edits server code or retries with changed settings automatically.

### Stage A1 launcher

Stage A1 uses the same protocol version, model, source rollouts, interfaces, target normalization, probe matrix, seed, scorer, and analysis code as A0. The only intended change is selection size. Every rollout whose four option contents can be recovered from the frozen prompt or MathVerse metadata is included; excluded question IDs and reasons are recorded in the immutable summary. The 2026-07-14 manifest contains 434 questions and 3,348 rollouts. This is a coverage limitation relative to the 3,858-rollout source, not an outcome-based selection rule.

The launcher:

- writes an immutable full manifest and checksum summary;
- keeps all caches and artifacts under `/data2`;
- refuses to start if the GPU or Ray/vLLM workers are busy;
- refuses to overwrite an existing A1 run;
- starts exactly one detached tmux session;
- records environment, resource, completion, and exit metadata;
- resumes only against an identical run contract.

Because the A1 request matrix is much larger than A0, the launch confirmation reports the exact request count and a runtime estimate derived from A0 instead of claiming a short completion time.

## Verification

- The in-session poll loop is exercised against the live incomplete A0 run before any launch action.
- Existing RL03 tests remain green.
- After conditional launch, verification requires the expected A1 tmux name, command, log path, output path, and a growing progress file.

## Safety Properties

- No package installation or system cleanup on H200.
- No process termination.
- No scheduled task, service, cron entry, or persistent poller installation.
- No automatic scientific-method changes after observing A0.
- No duplicate A1 launch.
- No A1 launch from partial or server-only evidence.
- No modification of unrelated dirty worktree files.
