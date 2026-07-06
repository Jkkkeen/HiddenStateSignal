# Hidden-State Uncertainty Signal: Research Plan

## 1. 核心问题

目标不是简单训练一个 hidden-state PRM，而是从推理过程的 hidden states / hidden-state trajectory 中提炼一个类似熵或不确定性的标量信号：

```text
u_k = f(H_k)
H_k = hidden states of prefix p_k at reasoning step k
```

希望它表达：

```text
u_k 高 -> 当前推理状态不稳定、分歧大、未来失败风险高
u_k 低 -> 当前推理状态更收敛、更稳定、未来成功概率高
```

更接近信用分配的定义是：

```text
V_k = P(final answer correct | prefix p_k)
```

所以真正要证明的是：

```text
u_k 可以预测 V_k
Delta u_k 可以反映当前 step 是否改善了推理状态
```

也就是说，这个信号最好不是“当前句子真不真”，而是“当前 reasoning state 还有没有走向正确答案的潜力”。

## 2. 和已有工作的区别

### 2.1 不只是 Hidden-State PRM

Hidden-state PRM 通常是：

```text
hidden state / prefix -> value head / classifier -> V_k
```

这里更想做的是：

```text
hidden-state trajectory -> interpretable scalar u_k
```

如果 `u_k` 足够稳定，它可以直接作为：

- 推理阶段的风险诊断信号
- RL 阶段的 process reward / reward shaping
- TTT / TTA 阶段的 update gate
- PRM 的辅助特征

### 2.2 和 ST-BoN 的区别

ST-BoN 主要解决横向选择问题：

```text
给 N 条候选，哪一条最可能最终最好？
```

这里更希望解决纵向诊断问题：

```text
给一条正在生成的 reasoning trajectory，第 k 步是不是进入高风险状态？
```

因此不要把主贡献写成：

```text
hidden-state score for reranking
```

更好的定位是：

```text
hidden-state uncertainty for step-level reasoning diagnosis and RL credit assignment
```

### 2.3 和 TTT / TENT / MEMO 的关系

TTT 的本质是：测试时用无标签目标更新模型参数。

```text
theta' = theta - eta * grad_theta L_TTT(x)
```

TENT / MEMO 用 output-space entropy 构造 test-time objective。这里可以把 `u_k` 看成 reasoning-space / hidden-state-space 的 uncertainty objective。

但是如果直接最小化 `u_k`，容易出现 representation hacking。因此更稳的用法是：

```text
u_k 决定是否 update / 是否分支 / 是否回滚
```

而不是一开始就：

```text
minimize u_k directly
```

## 3. 可以尝试的 hidden-state 信号

### 3.1 Layer Trajectory Instability

把同一个 prefix 在不同层的 hidden state 看成一条层间轨迹：

```text
h_0 -> h_1 -> ... -> h_L
Delta h_l = h_{l+1} - h_l
```

候选指标：

- path length: `sum_l ||Delta h_l||`
- normalized path length
- angle change between adjacent displacements
- curvature / turning rate
- displacement direction variance
- late-layer trajectory stability

直觉：正确推理可能有更稳定、更有结构的层间变化；错误推理可能方向变化更混乱，或者过早塌缩。

### 3.2 Effective Rank / Covariance Entropy

对多层 hidden states 或 displacements 做协方差分解：

```text
C = Cov({h_l}) 或 Cov({Delta h_l})
p_i = lambda_i / sum_j lambda_j
u = -sum_i p_i log p_i
```

可以得到类似“表示分散程度”的 entropy。

需要注意：高 entropy 到底表示探索充分，还是表示混乱，不能先验决定，必须通过 correctness / promise 做校准。

### 3.3 Layer Disagreement

给不同层训练轻量 probe，或者用同一个 probe 读不同层：

```text
q_l = probe_l(h_l)
u_k = Var_l(q_l)
```

直觉：如果不同层对当前 prefix 是否可靠的判断差异很大，说明模型内部还没有达成稳定判断。

这个方向比单点 probe 更像 uncertainty。

### 3.4 Temporal Drift Across Reasoning Steps

看 step 与 step 之间 hidden state 的变化：

```text
d_k = ||h_k - h_{k-1}||
a_k = ||h_k - 2h_{k-1} + h_{k-2}||
```

候选信号：

- sudden drift
- acceleration spike
- direction reversal
- long plateau followed by sharp jump

直觉：错误步骤、逻辑跳跃、反悔、编造，可能对应 hidden trajectory 的突变。

### 3.5 Success / Failure Prototype Distance

先收集正确和错误轨迹，构造 prototype：

```text
mu_success_k
mu_failure_k
u_k = dist(H_k, mu_success_k) - dist(H_k, mu_failure_k)
```

这可以作为非参数 verifier，也可以作为 probe 之前的强 baseline。

### 3.6 Perturbation / Rollout Consistency

对同一问题或同一 prefix 做轻微扰动：

- prompt paraphrase
- sampling multiple continuations
- dropout / noise perturbation
- small formatting changes

看 hidden states 是否稳定：

```text
u_k = disagreement({H_k^{(m)}})
```

这个方向成本更高，但能强调 `u_k` 不只是 token entropy，而是 reasoning state consistency。

## 4. 一般性验证实验

第一阶段不要训练模型，只做信号发现和诊断。

### 4.1 数据构造

对每道题生成 reasoning trace：

```text
question x
reasoning steps s_1, ..., s_K
final answer correct / wrong
prefix p_k = x + s_1 + ... + s_k
hidden states H_k
```

推荐任务：

- GSM8K / MATH 子集
- logic reasoning
- code reasoning
- symbolic manipulation
- 自造可控错误数据

### 4.2 Final Correctness Prediction

问题：

```text
u_k 能否区分最终正确和最终错误的轨迹？
```

指标：

- AUC
- accuracy / F1
- Spearman correlation
- calibration curve
- correct vs wrong 的均值差异和 effect size

关键点：要看不同 step ratio 的表现，例如 20%、40%、60%、80%。

### 4.3 Prefix Promise Prediction

这是最关键实验。

对每个 prefix `p_k` 做多次 rollout：

```text
V_k = correct_count / N
```

然后验证：

```text
Corr(u_k, V_k) < 0
Corr(Delta u_k, Delta V_k) < 0
```

其中：

```text
Delta u_k = u_k - u_{k-1}
Delta V_k = V_k - V_{k-1}
```

如果 `u_k` 能预测 `V_k`，说明它不是简单的最终答案置信度，而是 step-level promise signal。

### 4.4 Early-Warning Ability

只看前半段推理，判断最终是否会错。

对比：

- output entropy
- token logprob
- logit margin
- response length
- CoT length
- self-consistency vote
- text PRM
- standard hidden-state probe
- CoE / trajectory geometry
- ST-BoN-style cross-sample consistency

要证明：

```text
hidden-state u_k 比 output entropy 更早、更稳地发现失败风险
```

### 4.5 Error Onset Localization

标注或构造第一个错误步骤 `k*`。

验证：

```text
u_k 是否在 k* 附近上升？
u_k 是否能定位错误开始的位置？
```

指标：

- error onset localization accuracy
- mean distance to first wrong step
- spike-before-error rate
- top-k step detection

这个实验能说明 `u_k` 是纵向风险诊断信号，而不是只会给整条答案打分。

### 4.6 OOD / Cross-Model Stability

验证泛化能力：

- 数学题上调阈值，逻辑题上测试
- short CoT 上调阈值，long CoT 上测试
- 模型 A 上拟合，模型 B 上测试
- easy subset 上拟合，hard subset 上测试

如果只在同模型同数据集有效，文章价值会弱；如果 OOD 仍有效，信号的意义会强很多。

## 5. 如果用于推理阶段

推理阶段不要主打“rerank”，否则容易像 ST-BoN。更好的用法是 adaptive reasoning control。

### 5.1 Single-Trajectory Failure Warning

只生成一条 CoT，不开 Best-of-N：

```text
p_1 -> u_1
p_2 -> u_2
...
p_K -> u_K
```

验证：

```text
一条轨迹内部，u_k 能否提前发现最终失败？
```

这个设置下 ST-BoN 不自然适用，因为它需要多条候选横向比较。

### 5.2 Adaptive Compute Allocation

规则例子：

```text
if u_k low:
    continue / early stop
if u_k high:
    branch / rollback / resample / call verifier / use tool
```

指标：

- accuracy
- token cost
- accuracy per token
- number of branches
- failure recovery rate

### 5.3 Rollback / Branching

如果 `u_k` 在第 k 步突然升高：

```text
rollback to p_{k-1} or last low-u prefix
resample from that prefix
```

这个应用比简单 rerank 更能体现 step-level diagnosis。

### 5.4 与 ST-BoN 的公平比较

对照组：

- greedy
- self-consistency
- fixed BoN
- ST-BoN
- output entropy gating
- text PRM gating
- u-based adaptive control

重点不要说全面超过 ST-BoN，而是说：

```text
ST-BoN is a horizontal selector over multiple samples.
Our u_k is a vertical monitor over one reasoning trajectory.
```

## 6. 如果用于 RL 阶段

更推荐把 RL 作为主线，因为推理期 hidden-state scoring 已经有不少类似工作。

核心叙事：

```text
hidden-state uncertainty dynamics -> process reward -> better credit assignment in reasoning RL
```

### 6.1 不建议直接奖励绝对低 u

风险写法：

```text
reward = final_reward - lambda * u_k
```

问题：模型可能学会把 `u_k` 做好看，而不是真的推理更好。

更稳的写法：

```text
r_k^u = alpha * (u_{k-1} - u_k)
```

也就是奖励“不确定性下降”：

```text
当前步骤是否让推理状态更收敛？
```

### 6.2 Frozen Shadow Scorer

为了避免 representation hacking / representation drift，建议不要从正在训练的 policy 自己的 hidden state 直接算 `u_k`。

更稳的框架：

```text
policy model 生成 prefix p_k
frozen shadow model 读取 p_k
shadow hidden states -> u_k
Delta u_k -> process reward
```

这样 policy 不能直接修改 scorer 的 hidden geometry，只能通过生成更好的文本 prefix 间接影响 `u_k`。

### 6.3 RL Reward 形式

总奖励可以写成：

```text
R = R_outcome
  + beta * sum_k (u_{k-1} - u_k)
  - gamma * KL(policy || reference)
  - length_penalty
```

或者 step reward：

```text
r_k = beta * (u_{k-1} - u_k)
r_final = outcome reward
```

用于 PPO / GRPO / REINFORCE-style training。

### 6.4 RL 实验列表

#### Experiment R1: Offline Reward Validation

在已有 trajectories 上验证：

```text
u_{k-1} - u_k > 0 的步骤
是否更常出现在最终正确轨迹中？
```

以及：

```text
Delta u_k 是否和 Delta V_k 对齐？
```

这是 RL 前的必要 sanity check。

#### Experiment R2: RL Sample Efficiency

对比：

- outcome-only PPO / GRPO
- outcome + output entropy penalty
- outcome + text PRM
- outcome + CoE-style reward
- outcome + direct `-u_k`
- outcome + `Delta u_k`
- outcome + shadow-model `Delta u_k`

指标：

- pass@1
- final accuracy
- sample efficiency
- reward variance
- KL from reference model
- average reasoning length
- training stability

重点看：

```text
是否更快达到同等 accuracy
是否减少高方差 outcome-only reward 的训练不稳定
```

#### Experiment R3: Credit Assignment

验证奖励是不是真的分到了关键步骤。

可做：

- 人工插入错误步骤
- 对比错误发生前后的 `u_k`
- 观察 reward spike 是否靠近关键推理转折点
- 对错误步骤做 counterfactual replacement，看 `u_k` 是否下降

#### Experiment R4: Reward Hacking Check

必须做。

检查：

- `u_k` 越来越好看，但 accuracy 有没有同步上升？
- external verifier score 是否同步上升？
- OOD accuracy 是否崩？
- 是否出现模板化、重复、过短、过度自信？
- reasoning length 是否被异常压短？
- policy hidden-state scorer 与 shadow scorer 是否出现分歧？

如果 direct `-u_k` 出现 hacking，而 shadow `Delta u_k` 更稳，这反而是很好的实验结果。

#### Experiment R5: Ablation

消融：

- own hidden states vs frozen shadow hidden states
- absolute `u_k` vs `Delta u_k`
- layer choices
- last-token hidden state vs full-step pooled hidden states
- formula-based signal vs trained probe signal
- no KL vs KL
- with / without length control

#### Experiment R6: Generalization

在一个任务上训练，在另一个任务上测试：

- GSM8K -> MATH
- arithmetic -> symbolic reasoning
- short reasoning -> long reasoning
- in-domain prompts -> adversarial / noisy prompts

## 7. 推荐论文主线

如果主打 RL：

```text
Title direction:
Latent Uncertainty Dynamics as Process Reward for Reasoning RL
```

核心贡献可以写成：

1. 提出从 hidden-state trajectory 中提取 step-level uncertainty signal `u_k`。
2. 证明 `u_k` 能预测 prefix promise `V_k`，并且早于 output entropy / logprob 发现失败风险。
3. 提出 `Delta u_k` 作为 reasoning RL 的 process reward，用 frozen shadow scorer 缓解 reward hacking 和 representation drift。
4. 在 GRPO / PPO 中验证它提升 sample efficiency、final accuracy 和 step-level credit assignment。

推理阶段应用可以作为副实验：

```text
u_k also enables adaptive inference, but the main contribution is RL credit assignment.
```

## 8. 文章结构草案

```text
1. Introduction
   - reasoning RL 缺少可靠 step-level credit assignment
   - outcome reward 稀疏且高方差
   - hidden states 可能包含内部 uncertainty / promise 信号

2. Hidden-State Uncertainty Signal
   - 定义 H_k, u_k
   - 几类 trajectory / entropy / disagreement 指标
   - 不把它等同于 PRM

3. Signal Validation
   - final correctness prediction
   - prefix promise V_k
   - early warning
   - error onset localization
   - OOD / cross-model

4. RL with Latent Uncertainty Dynamics
   - Delta u_k process reward
   - frozen shadow scorer
   - integration with PPO / GRPO

5. Experiments
   - RL sample efficiency
   - comparison with baselines
   - reward hacking analysis
   - ablations

6. Adaptive Inference as Secondary Application
   - single trajectory warning
   - rollback / branching
   - comparison with ST-BoN

7. Limitations
   - cost of hidden-state extraction
   - scorer dependence
   - possible residual hacking
   - model / dataset specificity
```

## 9. 最重要的证明链条

整篇文章最核心的证据链应该是：

```text
u_k correlates with V_k
Delta u_k correlates with Delta V_k
Delta u_k works as process reward
shadow scorer reduces hacking
RL improves more efficiently than outcome-only training
```

如果这条链条成立，文章就不只是“hidden state 里有信息”，而是：

```text
hidden-state uncertainty dynamics can be used for credit assignment in reasoning RL
```

