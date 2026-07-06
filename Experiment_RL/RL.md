# RL Experiment Plan: Hidden/Logit Process Rewards for Long-CoT VLM Reasoning

本文档记录下一步想做的三个 GRPO/RL 实验。目标不是立刻证明新 reward 一定提升最终准确率，而是先验证：

```text
option-logit margin gain 或 hidden final margin 这类过程信号，
能否作为 answer correctness 之外的辅助 reward，
让 long-CoT VLM reasoning 更稳定地朝正确答案偏好演化。
```

当前最重要的已有证据：

```text
Qwen3-VL-8B-Thinking long-CoT smoke500:
accuracy_full  ~= 0.7728
accuracy_clean ~= 0.7875

hidden-direction think_final_margin L24:
AUROC ~= 0.6211

option-logit margin gain:
think_mean_step_gain_mean AUROC ~= 0.7763
think_mean_step_gain_max  AUROC ~= 0.7737

trimmed option-logit margin:
trimmed_margin_mean AUROC ~= 0.7967
trimmed_margin_max  AUROC ~= 0.7892
```

其中 `trimmed` 实验已经去掉 thinking 末尾的显式结论句，说明 logit margin 信号不只是最后一句 `therefore / answer is X` 的泄漏。

---

## 1. Core Setup

### 1.1 Base Model

第一版沿用当前 long-CoT 实验模型：

```text
Qwen/Qwen3-VL-8B-Thinking
```

或服务器本地 snapshot：

```text
/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/...
```

### 1.2 Task

第一版只做 MCQ 多模态数学：

```text
MathVerse mixed MCQ
```

理由：

```text
1. 已经有 rollout / labeling / logit-probe pipeline。
2. 选项 A/B/C/D 可以直接定义 correct-option logit margin。
3. 当前 baseline accuracy 已经明确，方便看 RL 是否带来增益或副作用。
```

### 1.3 GRPO Reward Form

所有实验都使用同一个 GRPO 框架，只换 reward：

```text
R_total_i = R_answer_i + lambda * B_i
```

其中：

```text
R_answer_i = 1 if final answer is correct else 0
B_i        = normalized bonus reward
```

GRPO advantage 仍然做 group-relative normalization：

```text
A_i = (R_total_i - mean_group(R_total)) / std_group(R_total)
```

bonus 不直接用 raw value，先做稳定化：

```text
B_i = clip(zscore_group(raw_bonus_i), -2, 2)
```

第一版推荐：

```text
lambda = 0.1
```

如果训练不稳定或 length 明显膨胀，再降到：

```text
lambda = 0.05
```

---

## 2. Experiment A: GRPO-Answer Baseline

### 2.1 Purpose

这是必须有的 baseline，用来回答：

```text
只用最终答案 correctness 做 GRPO，能带来多少提升？
```

### 2.2 Reward

```text
R_total = R_answer
```

其中：

```text
R_answer = 1 if pred_answer == ground_truth else 0
R_answer = 0 otherwise
```

### 2.3 Expected Role

这个实验不验证 hidden/logit 指标，只提供对照：

```text
如果后续 bonus reward 没有超过这个 baseline，
说明过程 reward 至少第一版没有实际训练价值。
```

### 2.4 Metrics

训练中记录：

```text
train reward
train accuracy
response length
think length
truncation rate
format validity
option distribution A/B/C/D
```

评估集记录：

```text
accuracy
clean accuracy
pass@k / majority vote if available
mean response tokens
truncation rate
```

---

## 3. Experiment B: GRPO + Hidden Final Margin L24

### 3.1 Purpose

测试 hidden-space correct-answer basin signal 能不能作为 reward shaping。

这是一个对照性质较强的实验，因为该指标在离线 AUROC 上只有：

```text
think_final_margin L24 AUROC ~= 0.6211
```

它的优势是更接近 hidden geometry；劣势是信号弱于 option-logit margin。

### 3.2 Hidden Direction Definition

对每道 MCQ 构造四个选项方向：

```text
v_A = normalize(h(prompt + option A) - h(prompt))
v_B = normalize(h(prompt + option B) - h(prompt))
v_C = normalize(h(prompt + option C) - h(prompt))
v_D = normalize(h(prompt + option D) - h(prompt))
```

如果正确答案是 C：

```text
v_correct = v_C
V_wrong = {v_A, v_B, v_D}
```

### 3.3 Reward Bonus

取 layer 24 的 thinking final hidden state：

```text
s_final = h_final - h_0
```

hidden margin：

```text
R_hidden_raw =
cos(s_final, v_correct)
-
max_{v in V_wrong} cos(s_final, v)
```

进入 GRPO 前做 group normalization：

```text
B_hidden = clip(zscore_group(R_hidden_raw), -2, 2)

R_total = R_answer + lambda_hidden * B_hidden
```

推荐第一版：

```text
lambda_hidden = 0.1
```

### 3.4 Interpretation

如果该实验优于 answer baseline，说明 hidden correct-answer basin alignment 不只是离线 diagnostic，也可能作为训练期 shaping。

如果不优于 baseline，也不意外：

```text
1. hidden AUROC 只有约 0.62。
2. hidden geometry 跨训练 step 会漂移。
3. reward 计算需要额外 forward hidden states，成本较高。
```

---

## 4. Experiment C: GRPO + Option-Logit Margin Gain

### 4.1 Purpose

这是当前最有希望的主实验。

离线结果显示：

```text
think_mean_step_gain_mean AUROC ~= 0.7763
think_mean_step_gain_max  AUROC ~= 0.7737
```

这比 hidden-direction reward 更强，而且更直接接近 MCQ answer preference。

### 4.2 Correct-Option Logit Margin

对每个 probe prefix 接固定回答接口：

```text
Given the reasoning so far, the answer is (
```

读取最后位置的 A/B/C/D logits。

如果正确答案是 C：

```text
margin_max_j =
logit_C_j - max(logit_A_j, logit_B_j, logit_D_j)

margin_mean_j =
logit_C_j - mean(logit_A_j, logit_B_j, logit_D_j)
```

由于 raw `top_option` 有强烈 A 偏置，reward 不使用 `top_option`，只使用 continuous margin。

### 4.3 Sparse Probe Points

训练时不要使用离线实验里的完整 12 个 thinking probe，因为太贵。

第一版使用稀疏 probe：

```text
25%, 50%, 75%, trimmed-final
```

其中 `trimmed-final` 是去掉 thinking 末尾显式结论句后的 prefix。

### 4.4 Reward Bonus

定义相邻 gain：

```text
gain_50 = margin_50 - margin_25
gain_75 = margin_75 - margin_50
gain_T  = margin_trimmed_final - margin_75
```

主 reward：

```text
R_gain_raw =
mean([
  clip(gain_50, -c, c),
  clip(gain_75, -c, c),
  clip(gain_T,  -c, c)
])
```

推荐：

```text
c = 5.0
```

进入 GRPO 前再做 group normalization：

```text
B_gain = clip(zscore_group(R_gain_raw), -2, 2)

R_total = R_answer + lambda_gain * B_gain
```

推荐第一版：

```text
lambda_gain = 0.1
```

如果发现输出变长或 reward hacking：

```text
lambda_gain = 0.05
```

### 4.5 Why Gain Instead Of Final Margin

不建议第一版直接用：

```text
R = final_margin
```

因为 final margin 更像 pre-answer outcome probe。

更推荐：

```text
R_gain = margin_j - margin_{j-1}
```

它奖励的是推理过程是否逐步提高 correct-option preference，更像 process reward。

---

## 5. Recommended Run Order

不要三个长训同时开。建议先做小规模 pilot：

```text
train prompts: 200-500 questions
rollouts per prompt: 4 or 8
training steps: 50-100
eval prompts: fixed held-out 100-300 questions
```

推荐顺序：

```text
1. Experiment A: GRPO-answer
2. Experiment C: GRPO + option-logit margin gain
3. Experiment B: GRPO + hidden final margin L24
```

原因：

```text
1. A 是必须 baseline。
2. C 离线 AUROC 最强，最值得优先验证。
3. B 信号较弱、成本较高，更适合作为 hidden-reward 对照。
```

---

## 6. Estimated Cost

粗略估计：

```text
GRPO-answer:
只需要 rollout + answer labeling，最快。

GRPO-hidden-final:
需要额外 hidden forward 和 option-direction hidden forward，
大约比 answer baseline 慢 30%-80%。

GRPO-logit-gain:
每条 rollout 需要 4 个 probe forward，
大约比 answer baseline 慢 50%-150%，取决于 prefix 长度和是否 batching。
```

如果做 200-500 questions 的 pilot：

```text
GRPO-answer: 约数小时到半天
GRPO-logit-gain: 约半天到一天
GRPO-hidden-final: 约半天到一天
```

实际时间取决于 VERL/GRPO 配置、max tokens、rollouts per prompt、是否使用 vLLM rollout。

---

## 7. Evaluation Protocol

所有实验使用同一组 eval prompts 和 decoding 参数。

### 7.1 Main Evaluation

```text
accuracy
clean accuracy
response length
think length
truncation rate
format validity
option distribution A/B/C/D
```

### 7.2 Diagnostic Evaluation

训练前后都跑离线 diagnostic：

```text
option-logit margin AUROC
trimmed margin AUROC
hidden final margin AUROC
response length vs reward correlation
option distribution shift
```

重点看：

```text
1. accuracy 是否提升。
2. process reward 是否导致 thinking 变长。
3. A/B/C/D 分布是否异常偏移。
4. 模型是否学会在 thinking 中提前写答案来刷 margin。
```

---

## 8. Reward Hacking Risks

### 8.1 A/B/C/D Prior

已有实验显示 raw top option 有强烈 A 偏置：

```text
Top option counts mostly A
```

因此：

```text
不能奖励 top_option。
只能奖励 correct-option margin。
```

### 8.2 Final Answer Leakage

如果奖励 final margin，模型可能学会尽早写：

```text
Therefore, the answer is C.
```

来提高 probe margin。

缓解：

```text
1. 使用 trimmed-final。
2. 使用 margin gain，而不是 final margin。
3. 监控 thinking 中是否过早出现 answer phrase。
```

### 8.3 Length Inflation

process reward 可能鼓励更长 thinking。

监控：

```text
mean think tokens
p90 think tokens
truncation rate
accuracy per length bin
```

必要时加入：

```text
R_total = R_answer + lambda * B_gain - alpha * length_penalty
```

---

## 9. First Pilot Recommendation

第一版最小可解释实验：

```text
Experiment A:
R = R_answer

Experiment C:
R = R_answer + 0.1 * zclip(R_logit_gain)
```

先不跑 hidden-final，等 A 和 C 有结果后再决定。

如果 C 相比 A：

```text
accuracy higher
length not much longer
truncation not worse
option distribution normal
held-out performance stable
```

再跑：

```text
Experiment B:
R = R_answer + 0.1 * zclip(R_hidden_L24)
```

作为 hidden geometry reward 的对照。

---

## 10. Current Open Questions

1. 训练集用 MathVerse 哪个 split / 哪些题？
2. 每题 rollout 数用 4 还是 8？
3. max response tokens 是否继续用 16384？
4. GRPO 使用哪个现成 pipeline：VERL / 自己轻量实现 / 现有服务器训练脚本？
5. option-logit probe reward 是否和 rollout 生成模型共享同一个模型权重实时计算？
6. reward probe 是否需要停止梯度，只作为 scalar reward 使用？

第一版建议先回答 1-4，然后再写代码。
