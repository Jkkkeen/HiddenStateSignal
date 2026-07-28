# Recoverability Candidate Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare letter-token and full-option-content confidence features as within-question predictors of revision recoverability on the existing 64 prefixes, without generating new revisions.

**Architecture:** Pure scoring helpers parse MathVerse choices, extract actual prompt-token log probabilities, and derive gold, commitment, uncertainty, and prompt-calibrated features. A standalone vLLM runner scores the four option contents under prompt-only and trimmed-reasoning contexts on H200. A local analysis script joins those scores with the frozen strict recovery outcomes and existing A/B/C/D probe logits, then reports question-clustered metrics.

**Tech Stack:** Python 3.11, pandas, NumPy, vLLM, Transformers/Qwen3-VL, pytest.

---

### Task 1: Pure Candidate-Scoring Features

**Files:**
- Create: `Experiment_RL/scripts/recoverability_candidate_scoring.py`
- Create: `Experiment_RL/tests/test_recoverability_candidate_scoring.py`

- [ ] Write failing tests for choice parsing, candidate-span log-probability extraction, length-normalized sequence scores, gold margin, selected-wrong commitment margin, top-two gap, entropy, and prompt-only calibration.
- [ ] Run `pytest -q tests/test_recoverability_candidate_scoring.py` and verify failures are caused by the missing module.
- [ ] Implement the minimal pure functions and rerun the focused tests.

### Task 2: vLLM Content-Scoring Runner

**Files:**
- Create: `Experiment_RL/scripts/score_recoverability_candidates_vllm.py`
- Modify: `Experiment_RL/tests/test_recoverability_candidate_scoring.py`
- Create: `Experiment_RL/scripts/run_recoverability_candidate_scoring.sh`
- Create: `Experiment_RL/scripts/launch_recoverability_candidate_scoring_tmux.sh`

- [ ] Write failing tests for prompt-only/reasoning message construction, project-relative image resolution, and deduplication of prompt-only requests by question.
- [ ] Implement a runner that renders Qwen3-VL chat prompts, scores actual tokens belonging to each option content with vLLM `prompt_logprobs`, and writes one JSONL row per prefix.
- [ ] Cache prompt-only scores by question and emit raw per-option scores plus derived content features.
- [ ] Add `/data2`-only shell and tmux launchers with explicit cache, output, model, and log paths.

### Task 3: Letter/Content Comparative Analysis

**Files:**
- Create: `Experiment_RL/scripts/analyze_recoverability_candidate_scores.py`
- Create: `Experiment_RL/tests/test_analyze_recoverability_candidate_scores.py`

- [ ] Write failing tests that merge strict recovery rows, four letter logits, and content scores by `(question_id, rollout_id)`.
- [ ] Test metric orientation: lower commitment gap, lower top-two gap, and higher entropy represent greater editability; higher gold margin represents greater gold support.
- [ ] Implement within-question pairwise concordance, correlations, and question-level bootstrap confidence intervals by reusing the established recoverability analysis helpers.
- [ ] Write CSV outputs and a Markdown report that labels this as discovery analysis and identifies the best exploratory feature without changing the frozen gate.

### Task 4: Verification And H200 Run

**Files:**
- Modify only if verification finds a tested defect in files from Tasks 1-3.
- Create remotely under: `/data2/hjk/projects/AI-HiddenState-ER/rl_recoverability/candidate_scoring_v1/`
- Pull results to: `Experiment_RL/recoverability/candidate_scoring_v1/`

- [ ] Run the focused tests, then `pytest -q tests` from `Experiment_RL`.
- [ ] Sync only the new scripts/tests/launchers to the H200 project under `/data2`.
- [ ] Run a one-prefix preflight and verify finite scores for all four options in both contexts.
- [ ] Launch the 64-prefix scoring run in tmux, confirm the process and GPU utilization, and wait only if completion is short.
- [ ] Pull completed score artifacts, run the local comparative analysis, and record exact AUC/CI results and H200 modifications.
