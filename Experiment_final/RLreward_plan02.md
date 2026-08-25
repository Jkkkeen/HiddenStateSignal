# Query-Relative H2/H10 Behavior Advantage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在同一个 query 的 rollout group 内，用 H2 判断当前轨迹更需要聚焦还是探索，再用 H10-P、H10-O 和 H2' 构造不使用 correctness/reward 标签的行为 advantage，并接入 Qwen3-1.7B GRPO canary。

**Architecture:** 每条完整 response 先计算一个 rollout-level 几何状态和两个行为质量分数。H2 只负责状态权重，前向占比/净前向占比组成聚焦分数，H10-O 与 H2' 组成探索速度分数；所有比较都在同一 query 的 rollout group 内完成。最后将中心化后的行为分数作为 response-level advantage，加到原始 GRPO advantage 上。

**Tech Stack:** Python, NumPy, pandas/Parquet, PyTorch hidden forward, veRL/GRPO, H200, tmux。

## Global Constraints

- 几何 bonus 不读取 `is_correct`、`answer_reward`、pass@k 或其他 correctness 标签；这些只用于离线评估。
- 当前 Qwen3-1.7B formal cohort 每个 query 有 8 条 rollout，组内比较的分母必须是 `n_q - 1 = 7`，不能固定写 15。
- 不使用 B1-B4 作为 reward gate，不使用 sigmoid/soft gate，不使用跨模型的绝对 H2 阈值。
- H2/H10 使用完整 thinking+answer response；terminal direction 由完整 response 的首尾状态定义。
- H10 是完整 response 结束后才能计算的 response-level 信号，不作为 token 生成过程中的即时方向标签。
- 首个较强 canary 使用 `lambda = 0.2`；`lambda = 0.1` 作为保守对照，`lambda = 0.3` 作为敏感性对照。正式扩大训练前必须比较三者。
- 正式训练和 canary 必须在 H200 的命名 tmux session 中启动，并保留日志、checkpoint 和 audit。

---

## 1. Rollout-Level Definitions

对同一个 query 的第 (i) 条 rollout，完整 response 的 pooled chunk states 为：

\[
h_{i,0},h_{i,1},\ldots,h_{i,K_i}
\]

相邻位移为：

\[
d_{i,k}=h_{i,k}-h_{i,k-1}
\]

完整 response 的 terminal direction 为：

\[
\hat g_i=
\frac{h_{i,K_i}-h_{i,0}}
{\|h_{i,K_i}-h_{i,0}\|+\epsilon}
\]

对每一步位移：

\[
a_{i,k}=d_{i,k}^{\top}\hat g_i
\]

\[
f_{i,k}=\max(a_{i,k},0),
\qquad
b_{i,k}=\max(-a_{i,k},0)
\]

\[
o_{i,k}=\sqrt{\max(\|d_{i,k}\|^2-a_{i,k}^2,0)}
\]

其中 (f) 是沿 terminal direction 的前向位移，(b) 是反向位移，(o) 是垂直位移。

### H2 state

完整 response 的累计 H2 为：

\[
H_i=H2_i=
\frac{\left\|\sum_{k=1}^{K_i}d_{i,k}\right\|}
{\sum_{k=1}^{K_i}\|d_{i,k}\|+\epsilon}
\]

- (H_i) 高：轨迹更直接、更少发生路径绕行；
- (H_i) 低：轨迹更曲折，侧向或抵消成分更多。

H2 不是“探索质量分数”，只用来决定当前更偏向聚焦还是探索。

### H10-P, H10-O and the focus score

\[
F_i=\sum_k f_{i,k},
\qquad
B_i=\sum_k b_{i,k},
\qquad
D_i=\sum_k o_{i,k},
\qquad
L_i=\sum_k\|d_{i,k}\|
\]

轴向前进/回退比例：

\[
Q_i=\frac{F_i}{F_i+B_i+\epsilon}
\]

这里的 (Q_i) 才是“轴向运动中向前的比例”。它不能单独作为聚焦分数：当 (F_i+B_i) 很小时，即使总运动几乎都是垂直的，(Q_i) 也可能接近 1。

轴向活动量和真正的前向占比定义为：

\[
A_i=\frac{F_i+B_i}{L_i+\epsilon},
\qquad
C_i^{\mathrm{focus}}=A_iQ_i=\frac{F_i}{L_i+\epsilon}
\]

如果希望同时惩罚沿 terminal direction 的回退，可以使用净前向比例：

\[
P_i^{\mathrm{net}}=\frac{F_i-B_i}{L_i+\epsilon}
=A_i(2Q_i-1)
\]

垂直比例：

\[
O_i=\frac{D_i}{L_i+\epsilon}
\]

低 H2 时的聚焦项使用 (C_i^{\mathrm{focus}}) 或 (P_i^{\mathrm{net}})，而不是 (Q_i-O_i)。这样会提高前向推进要求，但不会把垂直探索显式当成负奖励。高 H2 时再通过探索项提高对 (O_i) 和 H2' 的偏好。

### H2' exploration speed

为了避免把 B1-B4 固定 stage 写进 reward，建议从完整 response 的累计轨迹 (H_{i,k}) 对归一化进度 (u_{i,k}) 估计整体斜率：

\[
u_{i,k}=\frac{k}{K_i}
\]

\[
V_i=H2'_i=
\frac{\sum_k(u_{i,k}-\bar u_i)(H_{i,k}-\bar H_i)}
{\sum_k(u_{i,k}-\bar u_i)^2+\epsilon}
\]

- (V_i<0)：完整回答过程中 straightness 在下降，探索速度更高；
- (V_i>0)：轨迹整体在变直，开始收束。

现有 `delta_h2` 是 stage 间累计 H2 差分，可继续用于离线诊断，但不能不加说明地当作无 stage 的 rollout-level (V_i)。

---

## 2. Within-Query Relative Ranks

对同一个 query 的 rollout group，设 rollout 数为 (n_q)。当前 formal 数据中 (n_q=8)。

### H2 state rank

\[
R_H(i)=
\frac{1}{n_q-1}
\sum_{j\ne i}
\left[
\mathbf 1(H_j<H_i)
+\frac12\mathbf 1(H_j=H_i)
\right]
\]

- (R_H=0)：组内 H2 最低，轨迹最曲折；
- (R_H=1)：组内 H2 最高，轨迹最直接。

因此不需要 reference-policy 的 (μ) 或 (s)，也不需要跨模型共享 H2 阈值。

### Focus win score

\[
W_F(i)=
\frac{1}{n_q-1}
\sum_{j\ne i}\operatorname{sgn}(C_i^{\mathrm{focus}}-C_j^{\mathrm{focus}})
\]

第一版建议使用 (C_i^{\mathrm{focus}}=F_i/L_i)；如果需要明确抑制回退，则改用 (P_i^{\mathrm{net}})。两者都不直接减去 (O_i)。

### Exploration win score

因为 (V_i) 越负越好，定义：

\[
W_E(i)=
\frac{1}{n_q-1}
\sum_{j\ne i}\operatorname{sgn}(V_j-V_i)
\]

并列时 `sgn(0)=0`，不产生胜负。

这两个分数都在 ([-1,1]) 内，天然可以直接比较。

---

## 3. State-Conditioned Behavior Advantage

状态条件行为效用为：

\[
U_i=(1-R_H(i))W_F(i)+R_H(i)W_E(i)
\]

含义：

- H2 较低时，权重偏向 (W_F)：已经很曲折，提高前向推进要求，但不显式惩罚垂直探索；
- H2 较高时，权重偏向 (W_E)：轨迹过于直接，鼓励增加探索速度；
- 中间 H2 自动混合两种要求。

组内中心化：

\[
A_i^{\mathrm{beh}}=
U_i-\frac1{n_q}\sum_{j=1}^{n_q}U_j
\]

最终 GRPO advantage：

\[
\boxed{
A_i^{\mathrm{total}}=
A_i^{\mathrm{GRPO}}+\lambda A_i^{\mathrm{beh}}
}
\]

第一版默认：

\[
\lambda=0.2
\]

由于 (W_F,W_E,U) 已经是组内相对分数，(lambda=0.2) 会提供明显但不压过原始 reward 的行为偏好。训练日志必须同时记录：

- `std(A_grpo)`；
- `std(A_beh)`；
- `std(lambda * A_beh)`；
- 行为项占 total advantage 的比例。

如果行为项标准差超过原始 GRPO advantage 的 50%，需要降低 (lambda) 或做固定比例缩放，不能继续盲目增大。

---

## 4. Worked Example

用 4 条 rollout 展示计算过程。实际 8-rollout group 只需把分母从 3 改为 7。

| rollout | (H_i) | (V_i) | (Q_i=F_i/(F_i+B_i)) | (A_i=(F_i+B_i)/L_i) | (C_i^{focus}=F_i/L_i) |
|---|---:|---:|---:|---:|---:|
| A | 0.10 | -0.05 | 0.50 | 0.40 | 0.20 |
| B | 0.25 | -0.10 | 0.70 | 0.70 | 0.49 |
| C | 0.60 | -0.30 | 0.45 | 0.60 | 0.27 |
| D | 0.90 | -0.08 | 0.60 | 0.50 | 0.30 |

H2 从低到高为 (A<B<C<D)，所以：

\[
R_H=(0,1/3,2/3,1)
\]

按 (C^{focus}) 排序得到：

\[
W_F=(-1,1,-1/3,1/3)
\]

按 (V) 越负越好排序得到：

\[
W_E=(-1,1/3,1,-1/3)
\]

代入：

\[
U_A=-1
\]

\[
U_B=\frac23(1)+\frac13\left(\frac13\right)=\frac79\approx0.778
\]

\[
U_C=\frac13\left(-\frac13\right)+\frac23(1)=\frac59\approx0.556
\]

\[
U_D=-\frac13
\]

该例的行为排序为：

\[
B>C>D>A
\]

- A：H2 最低但前向推进最差，应该被压低；
- B：H2 较低但前向推进最好，得到最高 bonus；
- C：H2 较高且探索速度最快，得到探索 bonus；
- D：H2 最高但没有继续探索，得到负 bonus。

---

## 5. Offline Validation Before Training

- [ ] 审计每个 query 是否恰有 8 条 rollout；缺失或重复 group 不进入 bonus。
- [ ] 审计 (H,V,Q,A,C^{\mathrm{focus}},P^{\mathrm{net}},O,R_H,W_F,W_E,U,A^{\mathrm{beh}}) 的 NaN、范围和并列比例。
- [ ] 使用 H200 已保存 rollout，比较高 (A^{\mathrm{beh}}) 与低 (A^{\mathrm{beh}}) rollout 的 response length、pass@k 和 hidden geometry。
- [ ] 做 permutation control：在 query 内随机打乱 H2/H2'/H10 配对，确认 advantage 效果消失。
- [ ] 做 ablation：H2-only、固定聚焦、固定探索、query-relative adaptive 四个版本。
- [ ] 记录 `lambda=0.1/0.2/0.3` 的 advantage scale、训练稳定性和 held-out pass@1/pass@4/pass@8。
- [ ] 检查高行为分数是否只是更长 response 的替代变量。

## 6. RL Canary and Formal Run

### Canary

- [ ] 先在固定 query group、短训练预算上运行 `lambda=0.2` canary。
- [ ] 同时保存原始 GRPO、行为 bonus、total advantage、response length、KL、clip fraction 和 pass@k。
- [ ] 如果行为项主导 total advantage、KL 明显异常或 pass@k 下降，先回到 `lambda=0.1`，不修改几何公式。

### Formal

- [ ] 冻结 canary 通过的公式、group size 和 lambda。
- [ ] 在 H200 命名 tmux session 中启动正式训练。
- [ ] 保存 manifest、代码版本、配置、日志、checkpoint 和训练后 held-out rollout。
- [ ] 用独立 held-out cohort 验证行为 advantage 是否带来 pass@k 提升，而不是只看训练 reward。

## 7. Implementation File Map

当前计划只写设计，不修改训练代码。后续实现预计涉及：

- Modify: `Experiment_2E/experiment_2e/h2_h10.py`，补充完整 response H2/H2' 与 rollout-level P/O 导出；
- Modify: `Experiment_2E/experiment_2e/online_metrics.py`，记录行为项和 advantage scale；
- Modify: `Experiment_2E/experiment_2e/grpo3b.py` 或对应 veRL reward/advantage hook，加入 `lambda * A_beh`；
- Add tests: `Experiment_2E/tests/test_h2_h10.py`、`Experiment_2E/tests/test_behavior_advantage.py`；
- Run offline analysis against H200 rollout directory before any formal training.

本方案的科学表述是：H2 不是被直接优化的目标，而是控制“聚焦”和“探索”两种行为偏好的状态变量；H10-P/O 和 H2' 决定在该状态下哪一种行为更值得被相对奖励。
