# 2Dimension-02：Long-Response Hidden-State 复现与轨迹几何实验

## 1. 目标

在新的 MathVerse long-response 题目上复现两个已冻结的 hidden-state 信号，同时探索原始 activation entropy 与路径直线度/绕路比。当前阶段只做离线发现与确认，不修改 RL 训练。

## 2. 数据与固定设置

```text
model                 Qwen/Qwen3-VL-8B-Thinking
server root            /data2/hjk/projects/AI-HiddenState-ER
source                 data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl
run directory          experiment0_hidden_dynamics/experiment02_replication96_20260723
questions              500
rollouts/question      8
max_tokens             16384
analysis segment       response 起点至完整 </think> 之前
primary cohort         96 个新问题，每题至少 2 correct + 2 wrong
excluded               现有 FormalDiscovery24 的 24 个问题
```

只使用 non-truncated、具有可靠 think closing boundary 的 rollout，不重新生成回答。源数据在排除旧 24 题后有 101 题满足 `2+2`，其中 48 题满足 `3+3`；后者作为严格敏感性子集。正式运行前重新生成 eligibility audit 和冻结 manifest。

## 3. Confirmatory 指标

所有 progress bin 0–9 都计算并保存，但只有以下两个 cell 进入主确认判据，不根据新数据重新选择 layer、bin 或 representation。

### 3.1 横向 movement

主规格为 `mean_w128_s64 / L24`。对 span pooled hidden state (g_{i,k}^{24})：

\[
M_{i,b}=\operatorname{median}_{k:\operatorname{bin}(k)=b}
\left\|g_{i,k}^{24}-g_{i,k-1}^{24}\right\|_2
\]

确认 cell 为 `bin5`。同时保留每条 rollout 的 \((M_{i,0},\ldots,M_{i,9})\) 完整曲线。

### 3.2 纵向 activity burst

主规格为 `token / L15`：

\[
A_{i,b}=Q_{0.9}\left\{
\left\|h_{i,t}^{15}-h_{i,t}^{14}\right\|_2:
\operatorname{bin}(t)=b
\right\}
\]

确认 cell 为 `bin9`。同时保留每条 rollout 的 \((A_{i,0},\ldots,A_{i,9})\) 完整曲线。

### 3.3 主统计

- 每题内部 correct-vs-wrong AUC，再对问题等权平均；
- 4,000 次 question bootstrap 95% CI，固定 seed `20260724`；
- 每题 AUC \(>0.5\) 的 sign fraction；
- feature-only、think-length-only、feature+length 的 question-grouped OOF AUC；
- 96 题主结果与严格 `3+3` 子集结果分开报告。

若主 AUC 的 95% CI 下界高于 0.5，判为复现；CI 跨 0.5 判为不确定，不判为证否。两个主 cell 之外的 bin 只作描述性分析。

## 4. Exploratory 指标

### 4.1 原始与轨迹标准化 activation entropy

现有 update entropy 使用 (h_t^l-h_t^{l-1})。本轮同时计算两种原始层 activation 的 coordinate-energy entropy，并列保存，不相互替代：

#### 4.1.1 `H_act-raw`：per-token 去中心版

\[
\widetilde h_{t,l,j}=h_{t,l,j}-\frac{1}{D}\sum_m h_{t,l,m}
\]

\[
p_{t,l,j}=\frac{\widetilde h_{t,l,j}^2}{\sum_m\widetilde h_{t,l,m}^2},\qquad
H_{\mathrm{act,raw}}(t,l)=-\frac{\sum_jp_{t,l,j}\log(p_{t,l,j}+\epsilon)}{\log D}
\]

该版本回答“当前 token 的原始激活能量分散在多少个坐标上”，保留坐标之间的绝对尺度差异。

#### 4.1.2 `H_act-z`：trajectory-coordinate z-score 版

对每条 rollout (i)、每个 layer (l) 和每个坐标 (j)，只在该 rollout 的有效 thinking token 上计算时间均值和标准差：

\[
\mu_{i,l,j}=\frac{1}{T_i}\sum_{t=1}^{T_i}h_{i,t,l,j}
\]

\[
\sigma_{i,l,j}=\sqrt{\frac{1}{T_i-1}\sum_{t=1}^{T_i}
(h_{i,t,l,j}-\mu_{i,l,j})^2}
\]

\[
z_{i,t,l,j}=\frac{h_{i,t,l,j}-\mu_{i,l,j}}
{\max(\sigma_{i,l,j},\sigma_{\min})}
\]

其中
`sigma_min = 1e-4 × median_j(sigma_{i,l,j})`，并将 `z` 裁剪到 `[-8, 8]`，避免近乎不变坐标或极端离群值造成数值爆炸。随后直接使用 z-score 后的坐标能量计算：

\[
p^{z}_{i,t,l,j}=\frac{z_{i,t,l,j}^2}
{\sum_m z_{i,t,l,m}^2+\epsilon}
\]

\[
H_{\mathrm{act,z}}(i,t,l)=-\frac{\sum_jp^{z}_{i,t,l,j}\log(p^{z}_{i,t,l,j}+\epsilon)}{\log D}
\]

该版本回答“当前 token 上，有多少坐标相对于自己在整条 rollout 中的常态发生了异常变化”。z-score 后不再进行第二次 per-token 去中心，以便单独检验轨迹标准化的作用。

对 `H_act-raw` 和 `H_act-z` 都保存所有 hidden-state layer、所有 progress bin 的 mean、median、P90、effective dimensions；同时保存每层的 `median_sigma` 和低方差坐标比例，并与原有 update entropy 对照。两者均属于 exploratory family，不进入 confirmatory 判据。由于 `H_act-z` 使用完整 rollout 的均值和标准差，它只用于本轮离线发现；若未来用于在线 token-level RL credit，需要改用 prefix/running statistics 或冻结的校准统计量。

### 4.2 路径直线度与绕路比

主规格为 `mean_w128_s64`，固定 L24/L36；`mean_w128_s128` 作为无重叠窗口敏感性分析。令：

\[
d_{i,k,l}=g_{i,k,l}-g_{i,k-1,l}
\]

对整条轨迹或 progress block (B)：

\[
L_B=\sum_{k\in B}\|d_{i,k,l}\|_2,\qquad
N_B=\left\|\sum_{k\in B}d_{i,k,l}\right\|_2
\]

\[
S_B=\frac{N_B}{L_B+\epsilon},\qquad
D_B=\log(L_B+\epsilon)-\log(N_B+\epsilon)
\]

固定 `epsilon=1e-12`。同时报告 path length (L_B)、net displacement (N_B)、straightness (S_B) 和 log-detour (D_B)。计算 whole-trajectory 与 10 个 progress blocks；block 至少包含 3 个有效 displacement。控制 think length、span count 和 movement amplitude。

## 5. 图与结果文件

1. Movement 与 activity 的全部 rollout spaghetti curves，叠加 question-weighted correct/wrong 均值和 bootstrap CI；分别标记 bin5/bin9。
2. 两个指标的 bin0–9 within-Q AUC 曲线，0.5 为随机基线；主 cell 实心标记，其他 bin 空心标记。
3. 旧 Discovery24 与新 Replication96 的 correct-minus-wrong progress 曲线，分开显示、不合并统计。
4. `H_act-raw` 与 `H_act-z` 的 layer × progress 热图及 correct/wrong 趋势；比较两者的动态范围、within-Q AUC 和长度控制后的增量信息。
5. Path、net、straightness、log-detour 的 whole 与 progress-block 结果。
6. 输出冻结 manifest、eligibility audit、rollout-level parquet、question-level summary、分析报告和运行元数据。

## 6. 执行与资源

```text
GPU                    1 × H200
new generation         无
full hidden states     不落盘
saved vectors          仅 L24/L36 span displacement，float16
estimated vector disk  约 1–1.5 GB
execution              tmux，按问题断点续跑；GPU 非空闲时不启动
```

执行顺序：

1. 生成 audit 和冻结 96 题 manifest；
2. 先跑 4 题/约 32 rollout smoke，验证边界、公式、缺失值和文件恢复；
3. smoke 通过后在 tmux 启动 96 题正式抽取；
4. 完成 question-level bootstrap、长度控制、图和报告；
5. 检查主 cell 未被改动、全部 bin 曲线非空、旧/新数据未混合。

预计耗时：smoke 约 10–15 分钟，正式 forward 与在线聚合约 60–90 分钟，统计和画图约 15–25 分钟；总计算时间约 1.5–2 小时，保守预留 2.5 小时。不包含 Instruct 对照和 RL 训练。
