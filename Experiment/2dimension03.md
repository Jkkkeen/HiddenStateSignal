# 2Dimension-03: Raw Entropy Band Confirmatory Replication

## 1. 目标

在完全未参与 Experiment 0/02 的 MathVerse 题目上，确认 `raw activation entropy` 的 `L14-L19 / progress bin6` 层平均能否区分正确与错误 long-response rollout，并判断它是否提供超出回答长度的增量信息。本轮不扫描其他 layer/bin，不修改冻结特征，不进行 RL 训练。

## 2. 数据

```text
model                 Qwen/Qwen3-VL-8B-Thinking
dataset               MathVerse testmini, answer in A/B/C/D
rollouts/question     8
temperature           0.7
top_p                 0.95
max_tokens             16384
analysis segment       response start 至完整 </think> 前
primary cohort         120 个全新问题，每题 >=2 correct 且 >=2 wrong
sensitivity cohort     其中 >=3 correct 且 >=3 wrong 的问题
```

先排除旧 Thinking 500 题，再以固定 seed `20260725` 对候选题排序。预先冻结 600 道主批次和 200 道备用题；若主批次不足 120 道合格题，则按预先冻结的 50 题批次顺序补充。正式 cohort 始终取冻结顺序中最先满足条件的 120 题，选择时不计算或查看 entropy。

## 3. 冻结特征

对 rollout `i`、thinking token `t`、层 `l` 的 hidden state `h_{i,t,l}`，先在该 token 的 hidden coordinates 内去均值：

\[
\widetilde h_{i,t,l,j}=h_{i,t,l,j}-\frac{1}{D}\sum_m h_{i,t,l,m}
\]

再将平方能量归一化并计算 normalized coordinate-energy entropy：

\[
p_{i,t,l,j}=\frac{\widetilde h_{i,t,l,j}^2}
{\sum_m\widetilde h_{i,t,l,m}^2},\qquad
H_{i,t,l}=-\frac{\sum_j p_{i,t,l,j}\log(p_{i,t,l,j}+10^{-12})}{\log D}
\]

将完整 thinking 段按相对 token 位置均分为 10 个 bin。冻结分数为：

\[
E_i=\frac{1}{6}\sum_{l=14}^{19}
\left(\frac{1}{|T_{i,6}|}\sum_{t\in T_{i,6}}H_{i,t,l}\right)
\]

其中 `bin6` 是相对进度 `[0.6, 0.7)`。方向也冻结为 entropy 越高越预测 correct。抽取只保存六个层均值、层平均 `E_i` 和实际 `think_length`，不保存完整 hidden vectors。

## 4. 两级判据

1. **信号复现**：逐题计算 correct-vs-wrong pairwise AUC，再对 120 题等权平均；4,000 次 question bootstrap 的 95% CI 下界必须 `>0.5`。
2. **RL 候选**：5-fold question-grouped OOF logistic regression 中，`entropy+think_length` 相对 `think_length-only` 的逐题 AUC 增量，经配对 question bootstrap 后 95% CI 下界必须 `>0`。

两级都通过才进入 RL credit-assignment 实验；只通过第一级则视为内部相关信号，但暂不作为 RL 指标；第一级未通过则判为未复现。`3+3` cohort 只作预先声明的敏感性分析，不替代主结论。

## 5. 输出与资源

- 冻结候选清单、生成/eligibility audit、正式 manifest 与 SHA256；
- rollout-level parquet、question-level AUC、两级 gate 表、OOF 分数和 Markdown 报告；
- 一张 correct/wrong 分布图和一张逐题 AUC 图；不输出 layer/bin 搜索热图；
- H200 单卡，vLLM `gpu_memory_utilization=0.75`，tmux 断点续跑；
- 预计新 rollout 生成约 `9-17` 小时，标注与 cohort 冻结约 `5-10` 分钟，精简 hidden forward 约 `20-40` 分钟，统计约 `5-10` 分钟，总计约 `10-18` 小时；实际时间主要由新题的 thinking 长度决定。
