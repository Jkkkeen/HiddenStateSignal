# Angle: Semantic-Step Correct-Answer Basin Alignment Plan

本文档记录一个新的 hidden-state 方向性实验，用来验证：

```text
在 think / reasoning 过程中，模型是否逐步进入 correct-answer basin？
每个语义 step 的偏转，是否更接近正确答案方向？
```

这个实验的定位不是 test-time verifier，而是 **RL 训练期的 privileged process signal / reward-shaping candidate**。训练时我们知道 ground-truth answer，因此可以定义 correct-answer direction；推理时没有正确答案，所以该信号不能直接作为无监督 verifier 使用。

---

## 0. 背景和动机

现有 ER / path / angular 实验显示：

```text
短 CoT：后期位移小、ER 低、方向平滑 与 correctness 有相关性。
Long-CoT：这些全局/局部几何指标的 AUROC 大多回落到 0.50-0.58。
```

这说明“语义聚集”本身不等于“语义正确”。正确性更可能取决于：

```text
轨迹是否进入正确答案方向附近，
而不是单纯是否聚集、是否平滑、是否短路径。
```

因此本实验把问题从：

```text
trajectory 是否收敛？
```

改成：

```text
trajectory 是否朝 correct-answer direction 演变？
```

如果这个先验成立，correct-answer direction 可以作为 RLVR / GRPO 训练中的 dense reward 或 process reward 的候选特征。

---

## 1. 核心先验

### 1.1 要验证的主先验

```text
正确 rollout 的语义 step 在生成过程中，应该比错误 rollout 更稳定地增加 correct-answer alignment。
```

更具体：

```text
每个 reasoning step 之后，hidden state 是否比 step 之前更靠近 correct-answer direction？
这种 correct-direction gain 在正确 rollout 中是否显著高于错误 rollout？
```

### 1.2 不是要验证的东西

本实验不应该只证明：

```text
最终答对的 rollout 最后更靠近正确答案。
```

这可能是同义反复，尤其在 MCQ 上容易出现标签泄漏。真正有价值的是 prefix / step-level 过程信号：

```text
在最终答案生成之前，thinking trajectory 是否逐步朝 correct-answer basin 组织？
```

### 1.3 非单调先验

不能假设“越早靠近正确答案越好”。正确推理可能先探索错误方向，再排除、转向、收敛。因此更合理的先验是：

```text
early phase: 可以探索，不要求 correct alignment 高
middle/late phase: correct margin 应该上升
late phase: 如果进入 correct basin，应保持相对稳定
rethinking point: 健康转向应带来 correct-margin gain
```

---

## 2. Step 切分：从 token chunk 改为语义 step

本实验不使用固定 token chunk 作为主切分。固定 `64/128/256 token` 本质仍是机械切分，容易把多个 reasoning action 混在一起。

第一版使用 **rule-based semantic step segmentation**，目标是稳定、可复现、足够接近 reasoning unit。

### 2.1 Step 边界规则

按以下优先级切分 think 段：

```text
1. 换行边界
2. 显式 step 标记：Step 1, Step 2, First, Then, Next, Therefore, Thus
3. 句子边界：. ; : ? ! 中文句号、分号、问号、感叹号
4. rethinking trigger：wait, actually, reconsider, however, but, alternatively,
   on second thought, I made a mistake, let me re-check 等
```

### 2.2 Step 清洗

```text
过短 step：合并到相邻 step，例如少于 8-16 tokens。
过长 step：按句子边界继续拆分。
空白 / 纯格式 step：丢弃。
```

### 2.3 每个 step 的 hidden state

对第 `j` 个语义 step，得到 step-level state：

```text
h_j = last-token hidden of semantic step j
```

`last-token hidden` 作为主版本，因为它更接近“完成这一步推理后的状态”。

Robustness 版本：

```text
h_j_mean = mean hidden within semantic step j
```

主分析先用 last-token，mean hidden 作为稳定性检查。

---

## 3. Correct-answer direction 的定义

### 3.1 方向必须外部定义

Correct-answer direction 不能从该 rollout 的最终生成答案中取，否则会产生自证循环。

对于 MCQ，先为每道题构造四个选项方向：

```text
v_A, v_B, v_C, v_D
```

推荐第一版定义：

```text
v_X = normalize( h(prompt + option X) - h(prompt) )
```

其中：

```text
h(prompt)             = prompt 结束位置的 hidden state
h(prompt + option X)  = 追加候选选项 X 后的 hidden state
```

对于 ground-truth option：

```text
v_correct = v_ground_truth
```

错误选项集合：

```text
V_wrong = {v_X | X != ground_truth}
```

### 3.2 为什么这不等于标签泄漏

训练期知道 ground truth，因此可以构造 privileged direction。这个信号的目标是 RL shaping，不是 test-time verifier。

为了避免实验解释混淆，所有 trajectory 指标只使用：

```text
think segment hidden states
```

不使用：

```text
最终 answer segment hidden states
该 rollout 最终生成答案的 hidden state
```

---

## 4. Primary Metrics

第一版只报告少量 primary 指标，避免指标爆炸和多重比较。

### 4.1 State margin

定义当前 step state 相对起点的方向：

```text
s_j = h_j - h_0
```

Correct alignment：

```text
align_correct_state(j) = cos(s_j, v_correct)
```

Wrong alignment：

```text
align_wrong_state(j) = max_{v in V_wrong} cos(s_j, v)
```

State margin：

```text
m_j = align_correct_state(j) - align_wrong_state(j)
```

解释：

```text
m_j > 0: 当前 reasoning state 更接近 correct-answer direction
m_j < 0: 当前 reasoning state 更接近某个 wrong-answer direction
```

### 4.2 Step correct gain

```text
step_correct_gain_j = m_j - m_{j-1}
```

解释：

```text
这个语义 step 是否让 hidden state 更靠近 correct-answer basin？
```

这是第一版最核心的 process signal。

### 4.3 Turn-to-correct

定义 step displacement：

```text
d_j = h_j - h_{j-1}
```

Step movement toward correct direction：

```text
turn_to_correct_j = cos(d_j, v_correct) - max_{v in V_wrong} cos(d_j, v)
```

解释：

```text
这个 step 的移动方向，是更朝正确答案，还是更朝某个错误答案？
```

这是对“语义 step 偏转是否更接近正确答案”的直接度量。

### 4.4 Productive turn

相邻 step 位移的转向角：

```text
turn_angle_j = arccos( cos(d_j, d_{j-1}) )
```

productive turn：

```text
productive_turn_j = turn_angle_j * max(0, step_correct_gain_j)
```

解释：

```text
大的语义转向是否带来了 correct-margin gain？
```

这对 long-CoT / rethinking 特别重要。健康反思应该表现为：

```text
large turn_angle + positive step_correct_gain
```

无效震荡则可能是：

```text
large turn_angle + zero/negative step_correct_gain
```

### 4.5 Late margin

定义 late semantic steps：

```text
late = final 25% semantic steps
```

```text
late_margin = mean(m_j over late steps)
late_margin_std = std(m_j over late steps)
```

解释：

```text
推理后期是否稳定靠近 correct-answer basin？
```

---

## 5. Rollout-level Summary Features

从 step-level 序列聚合成 rollout-level 特征，用于 within-question AUROC 和统计检验。

Primary rollout features：

```text
mean_step_correct_gain       = mean(step_correct_gain_j)
positive_gain_rate           = fraction(step_correct_gain_j > 0)
late_margin_mean             = mean(m_j in late steps)
early_to_late_margin_gain    = late_margin_mean - early_margin_mean
mean_turn_to_correct         = mean(turn_to_correct_j)
late_turn_to_correct         = mean(turn_to_correct_j in late steps)
productive_turn_p90          = p90(productive_turn_j)
productive_turn_mean         = mean(productive_turn_j)
```

Basin entry / stability features：

```text
first_cross_pos = first relative step index where m_j > 0
final_margin    = m_last
late_margin_std
stable_correct_basin = late_margin_mean > 0 and late_margin_std small
```

Wrong-basin lock features：

```text
chosen_wrong_margin = align(chosen_wrong_option) - align(correct_option)
wrong_lock_score    = mean(chosen_wrong_margin in late steps), only for incorrect rollouts
```

---

## 6. Hypotheses and Success Criteria

### H1: Correct rollout has stronger correct-direction gain

```text
Correct rollouts should have higher mean_step_correct_gain and positive_gain_rate than incorrect rollouts.
```

If H1 fails, correct-answer basin alignment is weak as a dense process reward.

### H2: Correct rollout has higher late correct margin

```text
Correct rollouts should have higher late_margin_mean than incorrect rollouts.
```

If only H2 holds but H1 fails, the signal is more like outcome/probe signal than process reward.

### H3: Rethinking steps are productive in correct rollouts

For steps matching rethinking triggers:

```text
Correct rollouts should show higher turn_to_correct and step_correct_gain after rethink steps.
```

This tests whether long-CoT rethinking is geometrically meaningful.

### H4: Incorrect rollout converges to chosen-wrong basin

Among incorrect rollouts:

```text
late chosen_wrong_margin should often exceed correct margin.
```

This would support the wrong-basin convergence story.

---

## 7. Evaluation Protocol

### 7.1 Data

Start with existing long-CoT Thinking data:

```text
Qwen3-VL-8B-Thinking
MathVerse MCQ
smoke500 clean subset
think segment only
```

Then optionally compare to short-CoT Instruct data.

### 7.2 Layers

Use the same layers as ER.md / ER_long.md:

```text
layer 24
layer 36
```

### 7.3 Main evaluation

Within-question AUROC：

```text
For each question with both correct and incorrect rollouts,
rank rollouts by each basin feature.
```

Report：

```text
mean AUROC by feature
bootstrap CI by question
AUROC(+feature) and AUROC(-feature)
```

Primary features for AUROC：

```text
late_margin_mean
early_to_late_margin_gain
mean_step_correct_gain
late_turn_to_correct
productive_turn_p90
```

### 7.4 Step-level analysis

Step-level plots：

```text
x-axis: relative semantic step position
y-axis: m_j / turn_to_correct_j / step_correct_gain_j
lines: correct vs incorrect rollout
```

Rethink event analysis：

```text
Compare margin before and after rethink-trigger steps.
```

---

## 8. Required Controls

### 8.1 Chosen-answer basin control

For each rollout, define chosen option from `pred_answer`:

```text
v_chosen = v_pred_answer
```

Compute chosen margin:

```text
chosen_margin_j = cos(h_j - h_0, v_chosen) - max_{X != chosen} cos(h_j - h_0, v_X)
```

Purpose：

```text
Check whether the model is simply moving toward whatever answer it will eventually choose.
```

### 8.2 Label permutation control

Randomly permute correct labels within questions or across questions and recompute:

```text
permuted_correct_margin
```

Expected result：

```text
AUROC should fall to ~0.5.
```

This checks whether the signal is real rather than an artifact of option formatting or length.

### 8.3 Prefix-only control

All primary metrics must exclude final answer segment. Only think-step hidden states are used.

### 8.4 Step segmentation robustness

Compare at least two step-state definitions:

```text
last-token hidden of semantic step
mean hidden of semantic step
```

If results only appear in one unstable definition, interpret cautiously.

---

## 9. Expected Interpretations

### Positive result

```text
Correct rollouts show increasing correct-answer margin over semantic steps.
Rethinking steps in correct rollouts produce positive correct-margin gain.
Incorrect rollouts often enter and stabilize near a chosen-wrong basin.
```

Interpretation：

```text
Correct-answer basin alignment can be used as a training-time dense process signal.
```

Possible RL shaping：

```text
r_j = step_correct_gain_j
r_j = turn_to_correct_j
r_late = late_margin_mean
penalty = wrong_lock_score
```

### Partial result

```text
Only late_margin_mean separates correct vs incorrect, but step_correct_gain does not.
```

Interpretation：

```text
The signal behaves like an outcome probe, not a good step-level reward.
```

### Negative result

```text
Correct-answer direction does not separate correct and incorrect trajectories,
or signal disappears under label permutation / chosen-answer control.
```

Interpretation：

```text
The operational answer-basin definition is not reliable.
Need better option direction construction, SAE features, or learned probes.
```

---

## 10. First Experiment to Run

The first experiment should be:

```text
Semantic-Step Correct-Direction Gain on long-CoT smoke500 clean subset.
```

Minimum outputs：

```text
1. Step segmentation report
   - number of semantic steps per rollout
   - step length distribution
   - rethink trigger count

2. Correct margin curves
   - m_j vs relative semantic step position
   - correct vs incorrect rollout

3. Step gain curves
   - step_correct_gain_j vs relative semantic step position

4. Rethink event table
   - margin before/after rethink trigger
   - separated by final correctness

5. Within-question AUROC table
   - late_margin_mean
   - early_to_late_margin_gain
   - mean_step_correct_gain
   - late_turn_to_correct
   - productive_turn_p90

6. Controls
   - chosen-answer basin comparison
   - label permutation AUROC
```

Pass condition for moving toward RL shaping：

```text
H1 or H3 must hold robustly.
```

If only H2 holds, the signal may still be useful for analysis, but is weaker as a process reward.

---

## 11. Notes

- This experiment is intentionally different from ER/path experiments: it asks where the trajectory moves, not only how much it moves or how concentrated it is.
- The correct-answer direction is privileged information and should be framed as training-time shaping, not test-time inference.
- Semantic step segmentation is part of the hypothesis. If rule-based steps are too noisy, the next version can use LLM-labeled reasoning steps or parser-based discourse segmentation.

---

## 12. Follow-up: Option Logit Trajectory

Hidden-direction basin alignment produced useful diagnostics, but the strongest long-CoT signal is still modest:

```text
think_final_margin @ L24: AUROC ~= 0.62
answer_mean_gain / answer_lock_score: AUROC ~= 0.56-0.57
```

The limitation is that hidden-direction metrics use an externally constructed direction:

```text
v_correct = h(prompt + correct option) - h(prompt)
```

This direction may not match the model's actual answer preference at a partial reasoning prefix. The next experiment therefore asks a more direct MCQ question:

```text
At each reasoning progress point, if we ask the model for A/B/C/D now,
does the correct option have higher logit margin?
```

### 12.1 Core Probe

For a rollout prefix ending at probe point `j`, append a fixed answer interface:

```text
Given the reasoning so far, the answer is (
```

Then forward this prompted prefix and read the logits for the option labels:

```text
logit_A(j), logit_B(j), logit_C(j), logit_D(j)
```

For correct answer `B`, compute two margins:

```text
logit_margin_max(j)
  = logit_B(j) - max(logit_A(j), logit_C(j), logit_D(j))

logit_margin_mean(j)
  = logit_B(j) - mean(logit_A(j), logit_C(j), logit_D(j))
```

Interpretation:

```text
max margin  : strict lock-in against the strongest wrong option
mean margin : smoother overall preference for the correct option
```

Both should be saved. If max works but mean does not, correctness depends on beating the strongest distractor. If mean works but max does not, the model is broadly moving correct but one wrong option remains competitive.

### 12.2 Probe Positions

The first version should not probe every semantic step. Current semantic-step splitting is sentence-like and too fine for long thinking outputs:

```text
long thinking often contains many "Wait / Actually / But" fragments
some rollouts have 70-180+ semantic steps
probing every step would be expensive and noisy
```

Use fixed token-relative probe positions instead:

```text
thinking primary: 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95%, 100%
answer exploratory: 0%, 25%, 50%, 75%
```

This gives approximately 16 probes per rollout and keeps all rollouts comparable.

Notes:

```text
0% thinking is often empty-prefix / boilerplate dominated, so it is diagnostic rather than primary.
100% answer should not be primary because it can leak the generated final answer.
answer probes are exploratory; thinking probes are the main process signal.
```

`wait / actually / however / but` should not be first-version step boundaries. For now they are event markers only:

```text
rethink_count
rethink_density
first_rethink_pos
last_rethink_pos
```

Event-aware probes can be a second version if the fixed-grid result is promising.

### 12.3 Why Not Use Raw Next-Token A/B/C/D Logits?

At an arbitrary middle-of-thinking token, the model is usually not about to output an answer label. Directly reading A/B/C/D logits at that position can be poorly calibrated.

The fixed probe prompt makes every position comparable:

```text
same answer interface
same option tokens
same decision question
```

This is closer to the MCQ correctness mechanism than hidden-direction cosine.

### 12.4 Current Smoke Finding: What The Probe Measures

The first debug smoke showed an important behavior:

```text
qid=23, ground truth = A, rollout final answer = D
probe top option = A at almost every thinking/answer probe
```

Even when the prefix tail already contained text like "the answer is D", the forced probe still preferred the ground-truth option `A`.

This should not be treated as a simple bug. It means the fixed answer interface is not measuring whether the rollout will faithfully repeat its final generated answer. Instead, it measures:

```text
Given the prompt and the reasoning prefix so far,
what option does the model currently prefer when queried through a shared A/B/C/D interface?
```

So this experiment should be framed as a **forced-answer belief probe** or **option-preference probe**, not as a rollout self-consistency probe.

Implication:

```text
top_option is diagnostic only.
The primary signal should be the continuous correct-option margin trajectory.
```

In the smoke result, top option stayed `A`, but the correct margin weakened across late thinking probes in incorrect rollouts. That weakening is meaningful because it suggests wrong reasoning can damage the correct-answer preference even if it does not fully flip the top option.

### 12.5 First-Version Metrics

For both `logit_margin_max` and `logit_margin_mean`, compute:

```text
prompt_only_margin
early_margin
final_margin
late_margin_mean
early_to_late_gain
late_margin_drop
relative_late_margin
relative_final_margin
mean_step_gain
positive_margin_rate
first_cross_pos
max_margin
min_margin
margin_std
late_flip_rate
top_option_switch_count
```

Definitions:

```text
prompt_only_margin = margin from prompt + fixed answer interface, with no rollout prefix
early_margin = margin at first thinking probe, e.g. 5% or 10%
relative_margin_j = margin_j - prompt_only_margin
relative_late_margin = late_margin_mean - prompt_only_margin
relative_final_margin = final_margin - prompt_only_margin
early_to_late_gain = late_margin_mean - early_margin
late_margin_drop = early_margin - late_margin_mean
step_gain_j = margin_j - margin_{j-1}
first_cross_pos = first relative probe position where margin_j > 0
late_flip_rate = fraction of late probes where margin_j < 0
```

Primary features:

```text
late_margin_mean_max
early_to_late_gain_max
late_margin_drop_max
relative_late_margin_max
relative_final_margin_max
first_cross_pos_max
```

Secondary features:

```text
mean-margin versions
stability / flip features
answer-stage margins
rethink marker counts
top_option / top_option_switch_count
```

Why include `prompt_only_margin`:

```text
Some MCQ items are easy enough that the model already prefers the correct option before reading any rollout reasoning.
Without a prompt-only baseline, high late margin can be mistaken for a process signal.
```

The more RL-relevant question is therefore:

```text
Does the generated thinking improve, preserve, or damage the prompt-only correct-option preference?
```

### 12.6 Evaluation

Same as other experiments:

```text
within-question AUROC
bootstrap CI over questions
AUROC(+feature) and AUROC(-feature)
```

Use the clean long-CoT smoke500 set first:

```text
data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl
clean rollouts: 3858
mixed clean questions: 206
```

Recommended run order:

```text
1. Inspect probe positions on 5-10 rollouts, no model forward
2. Debug 3-24 rollouts with probe logits and inspect prefix tails / top options / margins
3. Smoke 100-200 rollouts if output format and logit margins look sane
4. Full clean smoke500 only if AUROC from smoke is promising
```

### 12.7 Expected Outcomes

Positive:

```text
correct rollouts preserve or increase correct logit margin over thinking progress
incorrect rollouts show margin drop, entropy increase, or late wrong-option competition
relative_late_margin / late_margin_drop reaches useful AUROC
wrong rollouts show late flip or lock into wrong option
AUROC approaches or exceeds 0.65
```

Partial:

```text
absolute late margin works but relative gain/drop does not
```

This suggests item difficulty or prompt-only knowledge explains much of the signal. It is still useful as a prefix-level verifier, but weaker as dense RL shaping.

Negative:

```text
logit margins and relative margin changes remain near AUROC 0.5
```

Then MCQ correctness is not captured well by option preference at these fixed probe points, and stronger probes or learned process classifiers are needed.

### 12.8 Interpretation Guardrails

Do not interpret this probe as:

```text
"What answer will this rollout eventually output?"
```

Interpret it as:

```text
"If the model is forced to answer now through the same A/B/C/D interface,
 how much does the current prefix support the correct option?"
```

Therefore:

```text
top_option can stay constant while margin still changes meaningfully.
wrong rollouts can still have positive correct margin if the prompt itself makes the answer easy.
the strongest process signal may be margin degradation, not top-option flip.
```
