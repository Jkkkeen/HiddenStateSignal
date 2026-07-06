# ER 语义轨迹实验计划

本文档记录两类围绕 Effective Rank (ER) 的 hidden-state 实验设计：

1. **Per-rollout local ER**：观察单条 rollout 在不同 chunk 位置上的语义展开/收敛轨迹，用于寻找可作为 `u_k` 的单条轨迹风险信号。
2. **Cross-rollout ER**：观察同一道题的多条 rollout 在同一生成阶段的语义分歧程度，用于理解题目级 uncertainty、consensus、ST-BoN/BoN 为什么有效或失效。

核心思想是：不要把 hidden state 只压成一个全局 scalar，而是把 response 看成一条语义轨迹，研究它在生成过程中的局部结构、阶段变化和同题分歧。

---

## 0. 背景和动机

VERL / Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning 用 last-layer hidden states 的 Effective Rank 来刻画 semantic exploration，并进一步定义 ERV / ERA 作为语义轨迹的速度和加速度。它的优势在于指标具有较清楚的物理意义：

```text
hidden-state trajectory -> semantic spectrum -> effective rank -> velocity / acceleration
```

相比之下，CoE 这类指标对 long-CoT 的完整 response 做 mean pooling，再跨层计算几何量。它可以作为 hidden geometry diagnostic，但直接做 raw delta 的物理意义较弱：

```text
raw Delta CoE = CoE(prefix_{k+1}) - CoE(prefix_k)
```

这个差值混入了 prefix 长度变化、mean pooling 尺度漂移、旧 token 被重新平均后的权重变化。因此下一步更合理的方向是转向 chunk-level 的 semantic trajectory diagnostic。

---

## 1. 基本定义

给定一道题 `q` 的第 `i` 条 rollout：

```text
y_{q,i} = [y_1, y_2, ..., y_T]
```

将 response 按 token 切成 `K` 个 chunk：

```text
C_{q,i,1}, C_{q,i,2}, ..., C_{q,i,K}
```

例如：

```text
chunk size = 256 / 512 / 1024 tokens
stride = chunk size 或 chunk size / 2
```

对每个 chunk 抽取某一层 hidden states：

```text
H_{q,i,k}^{(l)} in R^{m x d}
```

其中：

```text
m = 当前 chunk 的 token 数
d = hidden dimension
l = layer index
```

候选层：

```text
last layer：更接近 VERL 的 semantic trajectory 设定
middle-late layer：例如 layer 29，可能更接近 correctness-readable representation
```

对 hidden-state 矩阵做 SVD：

```text
H = U Sigma V^T
singular values = sigma_1, sigma_2, ...
```

归一化奇异值：

```text
p_j = sigma_j / sum_r sigma_r
```

计算 Effective Rank：

```text
ER(H) = exp(- sum_j p_j log p_j)
```

直觉：

```text
ER 低：hidden trajectory 主要沿少数语义方向走，语义状态较集中
ER 高：hidden trajectory 覆盖更多语义方向，语义展开更宽
```

注意：ER 不应该被简单解释成越高越好或越低越好。更合理的是结合生成位置看：

```text
早期较高 ER：可能表示有效探索
后期持续高 ER：可能表示发散、犹豫或过度思考
过低 ER：可能表示过早坍缩或模板化
```

---

## 2. 实验 A：Per-rollout Local ER

### 2.1 目标

实验 A 的目标是验证：

```text
单条 rollout 的局部 hidden-state ER 曲线，是否能区分最终正确和错误的推理轨迹？
```

更具体地说，想回答：

1. 正确 rollout 和错误 rollout 的 `ER_k` 曲线是否存在系统性差异？
2. 正确 rollout 是否表现出“先探索，后收敛”的模式？
3. 错误 rollout 是否表现出“持续发散、后期反弹、高波动”的模式？
4. `ER_k` 或其校准版本能否作为 `u_k` 的组成部分？

### 2.2 计算对象

对每道题 `q`、每条 rollout `i`、每个 chunk `k`：

```text
ER_local(q, i, k) = ER(H_{q,i,k})
```

这里的 `H_{q,i,k}` 是单条 rollout 在第 `k` 个 chunk 内的 token-level hidden states。

这是一种 intrinsic score，因为它只依赖当前 rollout 自己：

```text
s_i = f(q, y_i)
```

这种指标更适合后续做 RL reward / advantage shaping。

### 2.3 数据需求

最低实验规模：

```text
题目数：200-500
每题 rollout 数：4-8
max_new_tokens：512-1536
模型：1.5B / 3B / 7B
层：last layer + 一个中后层
```

如果使用已有 long-CoT 多模态 rollouts：

```text
只抽 1-2 层
按 chunk 在线计算 ER
不保存完整 token-level hidden states
```

建议保存：

```text
question_id
rollout_id
chunk_id
chunk_start
chunk_end
is_correct
response_length
layer
ER_local
token_logprob_mean
token_entropy_mean
```

### 2.4 主要图表

#### 图 A1：Correct vs Incorrect 的 ER 曲线

```text
x-axis: chunk index 或 relative position
y-axis: ER_local
line 1: correct rollout mean
line 2: incorrect rollout mean
shade: bootstrap confidence interval
```

要观察：

```text
正确样本是否先高后低
错误样本是否后期保持高 ER 或出现反弹
错误样本是否整体波动更大
```

#### 图 A2：按 response relative position 对齐

由于不同 response 长度不同，可以把 chunk 位置归一化：

```text
relative position = chunk_start / response_length
```

分桶：

```text
0%-10%, 10%-20%, ..., 90%-100%
```

这可以减少长短回答带来的位置偏差。

#### 图 A3：ER_local 的 binned correctness curve

对每个 chunk 位置 `k`，把 rollout 按 ER_local 分 bin：

```text
low ER -> high ER
```

观察每个 bin 的 empirical correctness：

```text
x-axis: ER bin
y-axis: correctness rate
```

重点不是要求单调，而是看是否存在 position-dependent pattern。

#### 图 A4：Within-question AUROC by chunk

对每个 chunk 位置 `k`，在同一道题内部比较正确/错误 rollout：

```text
AUROC_k = P[s(q, y^+) > s(q, y^-) | same q, chunk k]
```

这里 `s` 可以是：

```text
ER_local
-ER_local
calibrated ER_local
```

画：

```text
x-axis: chunk index / relative position
y-axis: within-question AUROC
```

这个图回答：

```text
哪个生成阶段的 local ER 最有区分度？
```

### 2.5 校准方式

不能直接把不同 chunk 位置的 raw ER 混在一起比较。推荐做 position-conditioned calibration：

```text
zER(q, i, k) = (ER(q, i, k) - mean_k) / std_k
```

其中 `mean_k` 和 `std_k` 在验证集上按 chunk 位置统计。

也可以按 relative position 做 calibration：

```text
zER(q, i, b) = (ER(q, i, b) - mean_b) / std_b
```

这样 `zER` 的含义变成：

```text
当前 rollout 在这个推理阶段的语义展开程度，是否高于同阶段平均水平？
```

### 2.6 与 `u_k` 的关系

不要直接设：

```text
u_k = ER_k
```

更合理的是把 ER 作为 `u_k` 的一个特征：

```text
u_k = g(zER_k, Delta zER_k, volatility_k, curvature_k, position_k)
```

最初可以用简单模型：

```text
logistic regression
small MLP
isotonic calibration
```

监督目标：

```text
final correctness
```

或者更强的目标：

```text
V_k = P(final answer correct | prefix up to chunk k)
u_k = 1 - V_k
```

### 2.7 可能结论

理想 positive finding：

```text
Correct rollouts show early semantic expansion followed by mid/late stabilization.
Incorrect rollouts show persistent high ER, late ER rebound, or unstable ER dynamics.
Position-calibrated local ER provides within-question discriminative signal.
```

negative finding 也有价值：

```text
Raw local ER alone cannot distinguish correctness.
ER needs to be combined with volatility, curvature, token entropy, or probe-based correctness score.
```

---

## 3. 实验 B：Cross-rollout ER

### 3.1 目标

实验 B 的目标是验证：

```text
同一道题的多条 rollout 在同一生成阶段的 hidden-state 表征是否逐渐收敛？
这种跨 rollout 的语义分歧是否和题目难度、正确率、majority vote、ST-BoN selection 有关？
```

这类指标不是单条 rollout 的质量分数，而是 group-level uncertainty / consensus signal。

### 3.2 计算对象

对每道题 `q`、每条 rollout `i`、每个 chunk `k`，先把该 chunk 压成一个向量：

```text
g(q, i, k) = mean_t h_{q,i,k,t}
```

或者使用：

```text
last token of chunk
chunk mean hidden
attention-weighted chunk mean
```

**当前执行版本批注（2026-06-05）**：

第一版先使用已经生成的 `chunklayer/h_gpu{0..7}.npz`，其中：

```text
h_chunk: (n_chunks, 37, 4096)
```

这里的 `h_chunk[c, l, :]` 表示第 `c` 个 response chunk 内所有 token 在第 `l` 层 hidden state 的平均值。因此 Cross-rollout ER 的第一版堆叠方式为：

```text
G_mean(q, k, l) =
[
  h_mean(q, rollout_1, chunk_k, layer_l);
  h_mean(q, rollout_2, chunk_k, layer_l);
  ...
  h_mean(q, rollout_N, chunk_k, layer_l)
] in R^{N x d}
```

优先层：

```text
layer 36：last layer chunk-mean hidden
layer 24：middle-late chunk-mean hidden
```

也就是说，当前版本首先做的是：

```text
同题、同 chunk 位置、同一层，把 8 条 rollout 的 chunk-mean hidden vector 堆成 N x 4096 矩阵，再计算 ER_group。
```

第二版计划做 `last-token` 版本：

```text
G_last(q, k, l) =
[
  h_last(q, rollout_1, chunk_k, layer_l);
  h_last(q, rollout_2, chunk_k, layer_l);
  ...
  h_last(q, rollout_N, chunk_k, layer_l)
] in R^{N x d}
```

其中 `h_last` 是该 chunk 最后一个 response token 在第 `l` 层的 hidden state。这个版本**需要重新 forward**，因为当前 `.npz` 只保存了 chunk mean hidden，不能从均值还原 chunk 内最后一个 token 的 hidden state。等重新 forward 时，应在提取 `h_chunk` 的同时额外保存：

```text
h_last: (n_chunks, 37, 4096)
```

后续实验 B 应同时报告：

```text
ER_group_mean(q,k,l)  # 基于 chunk mean hidden
ER_group_last(q,k,l)  # 基于 chunk last-token hidden，待 token-forward 后补做
```

固定题目 `q` 和 chunk 位置 `k`，把同题所有 rollout 的 chunk 表征堆成矩阵：

```text
G(q, k) = [g(q, 1, k);
           g(q, 2, k);
           ...
           g(q, N, k)] in R^{N x d}
```

然后计算：

```text
ER_group(q, k) = ER(G(q, k))
```

直觉：

```text
ER_group 高：同题不同 rollout 在该阶段进入了多个语义方向，分歧大
ER_group 低：同题不同 rollout 在该阶段聚集到相近语义区域，共识强
```

### 3.3 这和 ST-BoN 的关系

ST-BoN 做的是：

```text
同一道题
同一个 prefix 长度 c
比较不同 rollout 的 CoE scalar
选最靠近群体中心的 rollout
```

Cross-rollout ER 不直接选 winner，而是先问：

```text
这个题在第 k 阶段的 rollout 群体是否有明显共识？
```

如果 `ER_group(q,k)` 很高，说明候选 trajectories 很分散。这时 ST-BoN 的 centrality 可能不稳定，因为没有明显中心。

如果 `ER_group(q,k)` 下降，说明多个 rollout 逐渐收敛到相似语义区域。这时 majority vote 或 centrality selection 可能更可靠。

### 3.4 数据需求

Cross-rollout ER 要求每道题有多条 rollout：

```text
每题 rollout 数 N >= 4
更推荐 N = 8 / 16 / 32 / 64
```

对每个题目和 chunk 位置，需要至少有足够 rollout 到达该位置。对于短 response，后期 chunk 可能样本不足，需要记录有效样本数：

```text
valid_rollouts(q,k)
```

建议只在：

```text
valid_rollouts(q,k) >= 4 或 >= 8
```

时计算 `ER_group`。

### 3.5 主要图表

#### 图 B1：高正确率题 vs 低正确率题的 ER_group 曲线

先按题目最终正确率分组：

```text
easy/high-accuracy questions
medium questions
hard/low-accuracy questions
```

画：

```text
x-axis: chunk index / relative position
y-axis: ER_group(q,k)
line: question group mean
```

要观察：

```text
高正确率题是否更早出现 ER_group 下降
低正确率题是否一直保持高 ER_group
```

#### 图 B2：ER_group 与 answer diversity 的关系

对每个题目 `q`，计算 answer diversity：

```text
final_answer_entropy(q)
unique_answer_count(q)
majority_vote_confidence(q)
```

看：

```text
ER_group(q,k) 是否和 final answer entropy 正相关
ER_group(q,k) 是否和 majority vote confidence 负相关
```

这可以证明 hidden-state semantic dispersion 和 text-level answer diversity 是否一致。

#### 图 B3：ER_group 与 ST-BoN 成功率

对每道题和每个 chunk 位置，记录：

```text
ER_group(q,k)
ST-BoN winner correctness at k
majority vote correctness
oracle any-correct
```

分析：

```text
ER_group 低时，ST-BoN 是否更容易选对？
ER_group 高时，ST-BoN 是否退化到 random？
```

这能解释：

```text
ST-BoN 失败是不是因为同题 rollout hidden representations 没有形成可靠中心？
```

#### 图 B4：Cross-rollout ER 的阶段转折点

对每道题寻找：

```text
k_converge = first k where ER_group(q,k) falls below threshold
```

或者：

```text
Delta ER_group(q,k) = ER_group(q,k) - ER_group(q,k-1)
```

看：

```text
k_converge 是否预测题目准确率？
早收敛是否对应更高 majority vote accuracy？
持续不收敛是否对应 all-wrong / hard question？
```

### 3.6 指标和统计检验

推荐报告：

```text
Spearman correlation:
ER_group(q,k) vs question accuracy
ER_group(q,k) vs answer entropy
ER_group(q,k) vs majority confidence

AUROC:
ER_group(q,k) 预测 high-accuracy vs low-accuracy question

Regression with controls:
question accuracy ~ ER_group + response length + answer entropy
```

如果样本按题组织，建议用：

```text
bootstrap by question
clustered confidence interval
```

### 3.7 可能结论

理想 positive finding：

```text
Questions with successful reasoning show decreasing cross-rollout ER over generation progress,
suggesting semantic consensus formation across independent rollouts.
Hard or all-wrong questions maintain high cross-rollout ER, indicating persistent semantic disagreement.
```

对 ST-BoN 的解释：

```text
ST-BoN requires a meaningful group center.
When ER_group is high, rollout representations are multi-modal and centrality-based selection becomes unreliable.
When ER_group drops, centrality or majority-based selection becomes more meaningful.
```

negative finding：

```text
Cross-rollout ER does not correlate with correctness but correlates with answer diversity or response length.
Then it is better viewed as question-level uncertainty rather than a correctness signal.
```

---

## 4. 实验 C：Prefix Angular Deviation

### 4.1 目标

实验 C 的目标是把 VERL 中的 prefix-level ER 替换成 prefix-level angular metric，验证：

```text
累计到当前 prefix 为止，hidden-state 语义轨迹的平均转向程度，
是否能刻画 reasoning 过程中的稳定性、发散或收敛？
```

这一路线主要用于和 VERL 对齐：

```text
VERL:
prefix hidden states -> ER(prefix) -> ERV / ERA

Angular version:
prefix hidden states -> AngleMean(prefix) -> Angular Velocity / Angular Acceleration
```

注意：这个指标不应被解释成 semantic aggregation。它更适合解释为：

```text
semantic trajectory directional stability / semantic turning
```

也就是语义轨迹的方向稳定性和转向程度。

### 4.2 计算对象

固定某一层 `l`：

```text
last layer
middle-late layer, e.g. layer 29
```

对一条 rollout 的 response hidden states：

```text
h_1^l, h_2^l, ..., h_T^l
```

为了降低 token-level 噪声，建议先做 micro-window pooling：

```text
g_j^l = mean hidden of tokens [j*w, ..., (j+1)*w-1]
```

其中：

```text
w = 4 / 8 / 16 tokens
```

然后定义 semantic displacement：

```text
v_j^l = g_j^l - g_{j-1}^l
```

定义相邻 displacement 的偏转角：

```text
theta_j^l = arccos( <v_j^l, v_{j-1}^l> / (||v_j^l|| ||v_{j-1}^l||) )
```

对 prefix checkpoint `c`，定义 prefix 平均偏转角：

```text
PAD(c) = mean_{j <= c} theta_j
```

其中 PAD = Prefix Angular Deviation。

直觉：

```text
PAD 低：到当前 prefix 为止，语义轨迹整体较平滑
PAD 高：到当前 prefix 为止，语义轨迹频繁转向
```

### 4.3 类比 VERL 的速度和加速度

为避免直接做 raw adjacent delta，可以借鉴 VERL 的 historical-baseline 形式。

给定 prefix checkpoints：

```text
c_1, c_2, ..., c_K
```

先计算：

```text
m_k = PAD(c_k)
```

定义 angular velocity：

```text
delta_k = m_k - mean(m_1, ..., m_{k-1})
AV = mean_k delta_k
```

定义 angular acceleration：

```text
AA = mean_k (delta_k - delta_{k-1})
```

直觉：

```text
AV 高：当前阶段的平均语义转向高于历史水平
AA 高：语义转向程度正在加速上升
AA 低或负：语义转向趋于稳定或下降
```

如果用于 chunk-level 曲线，也可以保留每个 checkpoint 的：

```text
delta_k
delta_k - delta_{k-1}
```

而不是只保存全局 `AV / AA`。

### 4.4 数据设置

当前设定为 MathVerse，单条 rollout 长度在 1536 tokens 内。建议：

```text
max response length <= 1536
micro-window size w = 8
prefix checkpoints = [128, 256, 384, 512, 768, 1024, 1280, 1536]
layers = last layer + layer 29
```

如果 response 不足某个 checkpoint，则跳过该 checkpoint，并记录：

```text
valid_token_count
valid_micro_window_count
```

### 4.5 主要图表

#### 图 C1：Correct vs Incorrect 的 PAD 曲线

```text
x-axis: prefix length / relative position
y-axis: PAD(c)
line 1: correct rollout mean
line 2: incorrect rollout mean
```

要观察：

```text
错误 rollout 是否在中后期 PAD 更高
正确 rollout 是否在后期 PAD 下降或趋稳
```

#### 图 C2：Angular Velocity / Acceleration 曲线

```text
x-axis: prefix checkpoint
y-axis: delta_k 或 delta_k - delta_{k-1}
```

要观察：

```text
错误 rollout 是否出现后期 angular acceleration spike
正确 rollout 是否逐渐稳定
```

#### 图 C3：Within-question AUROC by prefix

在同一道题内部比较正确/错误 rollout：

```text
score = PAD(c)
score = -PAD(c)
score = calibrated PAD(c)
```

画：

```text
x-axis: prefix checkpoint
y-axis: within-question AUROC
```

这回答：

```text
哪个 prefix 阶段的语义转向指标最能区分同题好坏 rollout？
```

### 4.6 校准方式

PAD 也有明显 position effect。推荐：

```text
zPAD(c) = (PAD(c) - mean_c) / std_c
```

其中 `mean_c / std_c` 在验证集上按 prefix checkpoint 统计。

这样 `zPAD(c)` 的含义是：

```text
当前 rollout 在这个 prefix 阶段的语义转向程度是否高于同阶段平均水平。
```

### 4.7 可能结论

理想 positive finding：

```text
Incorrect rollouts show higher late-prefix angular deviation and stronger angular acceleration,
suggesting unstable semantic turning before wrong final answers.
```

negative finding：

```text
Prefix-level PAD is too smoothed by historical averaging.
Local chunk angular metrics are needed to capture short-range reasoning instability.
```

---

## 5. 实验 D：Local Chunk Angular Dynamics

### 5.1 目标

实验 D 的目标是补足实验 C 的 prefix 稀释问题，直接观察每个 chunk 内的局部语义偏转：

```text
每个 reasoning phase 内，hidden-state 轨迹是否平滑、震荡、突然转向或持续发散？
```

相比实验 C：

```text
Prefix Angular Deviation：稳定，对齐 VERL，但容易被历史平均稀释
Local Chunk Angular Dynamics：局部敏感，更适合 process diagnosis 和 u_k
```

### 5.2 计算对象

仍然固定某一层 `l`，并先做 micro-window pooling：

```text
g_j^l = mean hidden of a micro-window
v_j^l = g_j^l - g_{j-1}^l
theta_j^l = angle(v_j^l, v_{j-1}^l)
```

把 response 分成大 chunk：

```text
chunk size = 128 / 256 / 512 tokens
```

对每个 chunk `k` 内的角度集合：

```text
Theta_k = {theta_j | j in chunk k}
```

计算局部角动力学特征：

```text
LAD_mean(k) = mean(Theta_k)
LAD_std(k) = std(Theta_k)
LAD_p90(k) = p90(Theta_k)
LAD_max(k) = max(Theta_k)
spike_rate(k) = P(theta_j > tau)
```

其中 LAD = Local Angular Deviation。

**实验 D 指标预注册批注（2026-06-05）**：

为了避免一次性报告过多 angular 指标导致多重比较问题，实验 D 第一版将指标分成 **primary** 和 **secondary** 两类。

Primary 指标用于正式 AUROC / within-question discriminativeness 报告：

```text
cos_mean          # 相邻 displacement 的平均方向一致性，baseline 约为 0
cos_p10           # 较低分位的方向一致性，捕捉极端转向事件
spike_rate_90     # P(theta > pi/2)，超过正交的局部转向比例
AV_cos_mean       # cos_mean 的 chunk-to-chunk velocity
```

Primary 指标选择理由：

```text
cos_mean 是最干净的方向稳定性指标；
cos_p10 捕捉极端不稳定事件；
spike_rate_90 是离散化的 extreme-turn 指标；
AV_cos_mean 刻画局部角动力学随生成阶段的变化。
```

Secondary 指标用于趋势图、robustness check 和 exploratory analysis，不作为第一版正式 claim 的主依据：

```text
LAD_mean
LAD_std
LAD_p90
LAD_max
cos_std
cos_min
spike_rate_105
AE_norm_B6
AA_cos_mean
AV_LAD_p90
AA_LAD_p90
```

这样如果 primary 指标已有稳定信号，secondary 可作为补充解释；如果 primary 均无信号，则不应依赖 secondary 中偶然出现的单个高 AUROC 来做强 claim。

### 5.3 Angle Entropy 作为补充

如果需要刻画 chunk 内“转向模式是否集中”，可以加入 angle entropy。

将 `[0, pi]` 分成 `B` 个 bin：

```text
p_b = count(theta in bin b) / total_count
```

计算：

```text
AE(k) = - sum_b p_b log p_b
AE_norm(k) = AE(k) / log B
```

直觉：

```text
AE 低：chunk 内偏转角集中，转向模式较单一
AE 高：chunk 内偏转角分布分散，转向模式更复杂或混乱
```

但 AE 会丢掉顺序信息，因此不应作为唯一核心指标。它应和：

```text
LAD_mean
LAD_std
spike_rate
angular velocity
angular acceleration
```

一起使用。

### 5.4 Chunk-level 速度和加速度

对每个 chunk 得到一个 angular state：

```text
m_k = LAD_mean(k)
```

然后计算局部变化：

```text
AV_local(k) = m_k - m_{k-1}
AA_local(k) = AV_local(k) - AV_local(k-1)
```

也可以使用 historical baseline：

```text
delta_k = m_k - mean(m_1, ..., m_{k-1})
AA_hist(k) = delta_k - delta_{k-1}
```

相比 raw adjacent delta，historical baseline 更接近 VERL 的 ERV / ERA 设计，抗噪性更好。

### 5.5 主要图表

#### 图 D1：Correct vs Incorrect 的 LAD 曲线

```text
x-axis: chunk index / relative position
y-axis: LAD_mean / LAD_p90 / spike_rate
```

观察：

```text
错误 rollout 是否后期 LAD 更高
正确 rollout 是否中后期更稳定
```

#### 图 D2：Angle Entropy 曲线

```text
x-axis: chunk index
y-axis: AE_norm
```

观察：

```text
错误 rollout 是否在某些 chunk 出现更高角熵
正确 rollout 是否角度分布更集中
```

#### 图 D3：Angular spike heatmap

以 rollout 为行、chunk 为列：

```text
cell value = LAD_p90 或 spike_rate
```

按 correctness 排序，观察：

```text
错误 rollout 是否有更明显的 late-stage angular spike pattern
```

#### 图 D4：与 ER 的二维关系

把 local ER 和 angular metrics 组合：

```text
x-axis: ER_local(k)
y-axis: LAD_mean(k) 或 AE_norm(k)
color: correctness
```

解释四种状态：

```text
ER 高 + LAD 低：探索较广，但转向有结构
ER 高 + LAD 高：多方向且频繁转向，可能发散
ER 低 + LAD 低：稳定收敛或过早坍缩
ER 低 + LAD 高：狭窄空间内震荡，可能纠结/重复
```

### 5.6 与 `u_k` 的关系

实验 D 比实验 C 更适合构造 process-level `u_k`。

候选形式：

```text
u_k = g(zER_k, zLAD_k, zAE_k, AV_local(k), AA_local(k), token_entropy_k, position_k)
```

其中：

```text
zLAD_k = position-calibrated LAD
zAE_k = position-calibrated AE
```

如果只想先做无训练版本，可以定义：

```text
instability_k = zLAD_mean(k) + zAE(k) + zSpikeRate(k)
```

但论文主方法最好不要停留在手写加权和，而应进一步验证：

```text
这些 angular dynamics 是否提供 ER / logprob / entropy 之外的增量信息。
```

### 5.7 可能结论

理想 positive finding：

```text
Wrong reasoning trajectories exhibit late-stage angular instability:
higher local angular deviation, higher angle entropy, and more angular acceleration spikes.
```

negative finding：

```text
Token-level angular metrics are too lexical-sensitive.
Micro-window pooling or chunk-level aggregation is necessary.
```

---

## 6. 各类实验的区别

下表对比 A-E 与 G 这几个 single-rollout 轨迹指标实验（F 是训练动力学元实验，不在此轴上）：

| 维度 | A: Per-rollout local ER | B: Cross-rollout ER | C: Prefix Angular Deviation | D: Local Chunk Angular Dynamics | E: Path Velocity & Acceleration | G: Cross-Chunk Angular Dynamics |
|---|---|---|---|---|---|---|
| 基本对象 | 单条 rollout 的一个 chunk | 同题多条 rollout 在同一 chunk 位置 | 单条 rollout 的累计 prefix | 单条 rollout 的局部 chunk | 单条 rollout 的 cross-chunk 位移序列 | 单条 rollout 的中等粒度点位移序列 |
| 核心矩阵/序列 | chunk 内 token hidden states | 不同 rollout 的 chunk representation | prefix 内 angle 序列 | chunk 内 angle 序列 | chunk 间 L2 距离序列 d_k | 中等粒度点间位移的 angle 序列 |
| 衡量内容 | 局部语义展开宽度 | 同题多 rollout 语义分歧/共识 | 累计语义轨迹转向程度 | 局部语义转向、震荡、角加速度 | 位移量的时间模式：加减速、后期稳定性 | 宏观语义轨迹的方向稳定性/转向 |
| 物理含义 | semantic breadth | semantic consensus/diversity | semantic directional stability | semantic angular instability | semantic displacement dynamics | macro semantic directional stability |
| 适合用途 | 构造 `u_k` 的 ER 特征 | 理解题目难度、BoN、consensus | 对齐 VERL 的 prefix dynamics | process diagnosis / `u_k` 候选 | 拆解 path_length，寻找时间模式信号 | 补 E 的方向分量；norm-invariant 信号 |
| 是否适合 RL reward | 较适合 | 不直接适合，需要多 rollout | 可作为 trajectory-level shaping | 可作为 step/chunk-level shaping | 最适合作为 step/chunk-level shaping | 较适合，且跨 checkpoint 鲁棒 |
| 是否需要同题其他 rollout | 不需要 | 需要 | 不需要 | 不需要 | 不需要 | 不需要 |
| 是否需要 token-level hidden states | 是 | 否（chunk representation） | 是 | 是 | 否（只需 chunk-mean） | 是（中等粒度池化，需重新 forward） |
| 数据来源 | 需要 forward | chunk-mean 即可 | 需要 forward | 需要 forward | 现有 chunklayer npz | 需要 forward（sweep 多粒度） |

---

## 7. 推荐执行顺序

### Step 1：先做最小 angular smoke test

原因：

```text
MathVerse rollout length <= 1536，适合先验证 angular metrics 是否数值稳定。
如果逐 token angle 太噪，尽早用 micro-window pooling 修正。
```

最小实验：

```text
50-100 题
每题 4-8 rollouts
layer = last layer
micro-window size = 8
chunk size = 256
metrics = LAD_mean, LAD_p90, spike_rate, AE_norm
evaluation = correct/incorrect 曲线 + NaN/分布检查
```

### Step 2：做实验 D：Local Chunk Angular Dynamics

原因：

```text
它最接近 process-level u_k。
如果 local angular instability 有 within-question discriminativeness，就很值得继续。
```

最小实验：

```text
200-500 题
每题 4-8 rollouts
layers = last layer + middle-late layer
metrics = LAD_mean, LAD_std, LAD_p90, spike_rate, AE_norm, AV_local, AA_local
evaluation = within-question AUROC + curve visualization
```

### Step 3：做实验 A：Per-rollout local ER

原因：

```text
ER 衡量 semantic breadth，angular metrics 衡量 semantic turning。
两者互补，组合起来比单独一个 scalar 更有解释力。
```

最小实验：

```text
同一批数据
同样 chunk 切分
metrics = ER_local, zER_local, Delta zER_local
evaluation = 与 angular metrics 做互补性分析
```

### Step 4：做实验 C：Prefix Angular Deviation

原因：

```text
它对齐 VERL 的 prefix-level ER/ERV/ERA 结构。
可以作为论文中和 VERL 最直接的 conceptual bridge。
```

最小实验：

```text
prefix checkpoints = [128, 256, 384, 512, 768, 1024, 1280, 1536]
metrics = PAD, zPAD, AV, AA
evaluation = prefix AUROC + 正误曲线
```

### Step 5：做实验 B：Cross-rollout ER

原因：

```text
它解释同题多 rollout 的语义分歧，帮助理解 ST-BoN / majority vote / question difficulty。
```

最小实验：

```text
每题 N >= 8 rollouts
chunk representation = chunk mean hidden
metrics = ER_group(q,k)
evaluation = correlation with question accuracy / answer entropy / majority vote confidence
```

### Step 6：把 ER + Angular features 结合成 semantic risk

候选特征：

```text
zER_k
Delta zER_k
zLAD_k
zAE_k
AV_local(k)
AA_local(k)
token entropy mean
token logprob mean
position index
response length so far
```

训练一个轻量风险函数：

```text
u_k = g(features_k)
```

目标：

```text
final correctness
或 prefix promise V_k = P(final correct | prefix up to k)
```

### Step 7：如果有信号，再考虑 RL

不要直接把 raw ER / raw angle 当 reward。建议：

```text
u_k = calibrated semantic risk
Delta u_k = u_{k-1} - u_k
```

作为 bounded shaping：

```text
A'_t = A_t + lambda * clip(Delta u_k)
```

或者：

```text
r_k = lambda * clip(u_{k-1} - u_k)
```

保留 final correctness reward，hidden-state signal 只做辅助。

---

## 8. 风险和注意事项

### 8.1 ER 不等于 correctness

ER 的语义是：

```text
effective semantic dimensionality
```

不是：

```text
probability of correctness
```

因此不能预设：

```text
ER 越高越好
ER 越低越好
```

必须按位置和任务进行校准。

### 8.2 Angular metrics 不等于 semantic aggregation

偏转角指标衡量的是：

```text
semantic trajectory turning / directional instability
```

不是：

```text
semantic clustering / aggregation
```

如果要讨论聚集或分歧，应使用：

```text
ER
cross-rollout ER
cluster compactness
```

### 8.3 Token-level angle 可能过噪

逐 token 偏转角会受以下因素影响：

```text
词汇选择
标点
公式符号
tokenization artifacts
```

因此推荐先做：

```text
micro-window pooling
chunk-level aggregation
position-conditioned calibration
```

### 8.4 长度效应

chunk size、response length、valid token count 都会影响 ER 和 angular metrics。需要：

```text
固定 chunk size
记录有效 token 数
按 chunk position calibration
控制 response length
```

### 8.5 层选择

VERL 使用 last layer，但 correctness probe 可能在中后层更强。因此至少比较：

```text
last layer
middle-late layer
```

如果资源允许，再做 layer sweep。

### 8.6 不保存完整 hidden states

尤其在 long-CoT setting 下，不要保存全层全 token hidden。建议：

```text
在线计算 ER 和 angular metrics
只保存 chunk-level metrics
只保留少量层
```

### 8.7 Cross-rollout ER 不适合直接当单条 reward

因为它依赖同题其他 rollout：

```text
ER_group(q,k) = f({y_i}_{i=1}^N)
```

它更适合解释 group uncertainty / BoN，而不是直接给单条 RL trajectory 打分。

---

## 9. 实验 E：Path Velocity & Acceleration (Cross-Chunk Displacement Dynamics)

### 9.1 目标

实验 E 的目标是把 Method 3 的 path_length（全局标量 sum(d_k)）拆解为逐 chunk 时序，验证：

```text
逐 chunk 位移 d_k 的时间模式（速度、加速度、后期稳定性）
是否比 path_length 本身提供更强的 correctness 区分信号？
```

动机来自实验 A 的 ER → ERV 提升：

```text
raw ER:          AUROC ~0.62
ERV late_min:    AUROC ~0.69（提升 +0.07）
path_length:     AUROC ~0.71
path dynamics:   AUROC ~?（预期提升）
```

path_length = sum(d_k) 丢弃了所有时间信息。如果错误 rollout 的 d_k 具有特定的时间模式（后期突增、震荡、不稳定），逐 chunk dynamics 可能捕获 path_length 无法表达的信号。

### 9.2 基本定义

固定层 `l`（l = 36 或 24），对一条 rollout 的 N+1 个 chunk，取第 l 层的 token-mean hidden state：

```text
h_0^(l), h_1^(l), ..., h_N^(l)    ∈ R^d, d=4096
```

逐步位移（"速度"）：

```text
d_k = ||h_{k+1}^(l) - h_k^(l)||    k = 0, 1, ..., N-1
```

逐步加速度：

```text
pv_k = d_k - d_{k-1}    k = 1, ..., N-1
```

二阶加速度：

```text
pa_k = pv_k - pv_{k-1}    k = 2, ..., N-1
```

### 9.3 核心假设

```text
正确 rollout：d_k 后期趋于平稳或递减（语义状态收敛，不再大幅移动）
错误 rollout：d_k 后期突增（跳到新方向）、震荡（犹豫/反复）、或持续高位（发散）
```

### 9.4 主指标

#### 全局统计量（每条 rollout 一个标量）

| 标量名 | 公式 | 直觉 |
|---|---|---|
| `d_mean` | mean(d_k) | 平均步幅（≈ path_length / N） |
| `d_std` | std(d_k) | 步幅波动——越大越不稳定 |
| `d_cv` | std(d_k) / mean(d_k) | 步幅变异系数——归一化波动 |
| `d_max` | max(d_k) | 最大单步跳变 |
| `d_late_mean` | mean(d_k for last 25% chunks) | 后期平均步幅 |
| `d_late_max` | max(d_k for last 25% chunks) | 后期最大步幅 |
| `d_late_std` | std(d_k for last 25% chunks) | 后期步幅波动 |
| `d_ratio_late_early` | d_late_mean / d_early_mean | 后期 vs 前期步幅比——>1 表示加速 |

#### 速度变化统计量

| 标量名 | 公式 | 直觉 |
|---|---|---|
| `pv_late_min` | min(pv_k for last 25%) | 后期最大减速（负值=步幅骤降=收敛） |
| `pv_late_max` | max(pv_k for last 25%) | 后期最大加速（正值=步幅骤增=发散） |
| `pv_late_mean` | mean(pv_k for last 25%) | 后期平均加速方向 |
| `pv_late_std` | std(pv_k for last 25%) | 后期加速波动——越大越不稳 |
| `pv_abs_late_mean` | mean(|pv_k| for last 25%) | 后期绝对加速度均值 |

#### Historical baseline 变体

与 ERV 类比，用 historical mean 做 baseline：

```text
delta_k = d_k - mean(d_0, ..., d_{k-1})
```

| 标量名 | 公式 | 直觉 |
|---|---|---|
| `d_hist_late_min` | min(delta_k for last 25%) | 后期最低偏离——最大收敛信号 |
| `d_hist_late_max` | max(delta_k for last 25%) | 后期最高偏离——最大发散信号 |
| `d_hist_late_mean` | mean(delta_k for last 25%) | 后期平均偏离方向 |

### 9.5 Primary vs Secondary

Pre-commit 以下 4 个为 primary（正式报告 AUROC + CI）：

```text
Primary:
  d_late_mean         → 后期步幅均值（-方向：越小越好）
  d_ratio_late_early  → 后/前步幅比（-方向：越小越好）
  pv_late_max         → 后期最大加速（-方向：越小越好）
  d_hist_late_min     → historical baseline 后期最低偏离（+方向：越大越好=越收敛）
```

其余为 secondary/exploratory。

### 9.6 方向承诺

| 标量 | 预期方向 | 直觉 |
|---|---|---|
| d_late_mean | 小→对 | 正确 rollout 后期位移小，已收敛 |
| d_late_max | 小→对 | 正确 rollout 后期无突变 |
| d_late_std | 小→对 | 正确 rollout 后期步幅稳定 |
| d_ratio_late_early | 小→对 | 正确 rollout 后期不比前期走得更远 |
| d_std / d_cv | 小→对 | 正确 rollout 步幅更均匀 |
| pv_late_max | 小→对 | 正确 rollout 后期无急加速 |
| pv_late_std | 小→对 | 正确 rollout 后期加速稳定 |
| d_hist_late_min | 大→对 | 正确 rollout 后期不会远低于历史均值（或：最"负"的点也不太负） |

### 9.7 数据来源

**不需要新 forward。** 使用现有 Method 3 的 chunklayer 数据：

```text
output_311q/chunklayer/h_gpu{0..7}.npz
  h_chunk: (n_chunks, 37, 4096) fp16
```

取固定层 k 的 per-chunk token-mean hidden state，计算相邻 chunk 间的 L2 距离即可。

### 9.8 与 path_length 和 ERV 的关系

```text
path_length = sum(d_k)         → 丢弃时间信息
d_late_mean = mean(d_k, last)  → 保留位置信息
pv_k = d_k - d_{k-1}          → 保留时间动态

ERV_k = ER_k - ER_{k-1}       → 语义宽度的变化
pv_k = d_k - d_{k-1}          → 位移量的变化
```

ERV 衡量"语义展开/收敛的速度"，pv_k 衡量"空间移动快慢的变化"。两者在高 ER 收敛 + 低位移的 regime 中可能相关，但当模型在低维子空间内大幅移动（ER 低但 d_k 高）时会解耦。

预期：
- 如果 path dynamics 比 path_length 提升明显（类比 ERV vs raw ER），说明时间模式确实有信息
- 如果和 path_length 接近，说明 sum 已经是足够好的 summary statistic

### 9.9 主要图表

#### 图 E1：Correct vs Incorrect 的 d_k 曲线

```text
x-axis: chunk index / relative position
y-axis: d_k (mean across rollouts)
line 1: correct rollout mean
line 2: incorrect rollout mean
shade: 95% CI
```

观察：

```text
正确 rollout 的 d_k 是否后期下降
错误 rollout 的 d_k 是否后期上升或震荡
两者是否在早期相似、后期分叉
```

#### 图 E2：pv_k（加速度）曲线

```text
x-axis: chunk index
y-axis: pv_k = d_k - d_{k-1}
```

观察：

```text
错误 rollout 是否后期出现正加速度 spike
正确 rollout 是否后期 pv_k → 0 或负值
```

#### 图 E3：Primary AUROC by chunk

类似实验 D 的图：用逐 chunk 的 d_k 本身作为 score，计算 within-question AUROC：

```text
x-axis: chunk index
y-axis: AUROC using -d_k as score
```

这回答：哪个生成阶段的位移大小最能区分正误。

#### 图 E4：Path dynamics vs ERV scatter

```text
x-axis: pv_late_max 或 d_hist_late_min
y-axis: erv_adj_late_min
color: correctness
```

观察两者是否提供互补区分力。

### 9.10 与 combo 实验的关系

如果 path dynamics 比 path_length 有提升，再做：

```text
combo_path_dynamics_erv = standardize(d_hist_late_min) + standardize(erv_late_min)
```

验证组合是否超过各自单独。

### 9.11 可能结论

理想 positive finding：

```text
Path velocity/acceleration dynamics provide stronger correctness signal than path_length sum,
confirming that temporal patterns in hidden-state displacement carry information
beyond total displacement magnitude.
Correct rollouts show decelerating displacement (convergence),
while incorrect rollouts show late-stage acceleration spikes or sustained high velocity.
```

如果 path dynamics ≈ path_length：

```text
sum(d_k) is already a sufficient summary of displacement dynamics for correctness prediction.
The temporal pattern of d_k is less informative than its total magnitude.
This suggests that incorrect rollouts simply traverse more total distance
rather than exhibiting a specific temporal profile.
```

---

## 10. 实验 F：Training-Step Hidden Trajectory Dynamics

### 10.1 目标

实验 F 的目标不是再问固定 checkpoint 下哪个指标最好，而是研究训练过程中 hidden-state trajectory 诊断信号是否稳定、增强，或者退化。

更准确的问题定义是：

```text
随着训练 step 推进，hidden-trajectory 指标在同题内区分 correct vs incorrect rollout 的能力是否稳定或增强？
同时，这些指标的 raw trend 是否显示模型轨迹正在向更稳定、更收敛的模式迁移？
```

因此实验 F 必须同时保留两类曲线：

```text
1. Raw trend:
   ER / ERV / path_length / d_late_mean 等指标随训练 step 的均值变化。
   这回答：训练是否改变了 hidden trajectory 的形态？

2. Discriminative trend:
   每个 checkpoint 内的 within-question AUROC(step)。
   这回答：该指标作为正确性诊断 / semantic risk signal 是否稳定有效？
```

不要把这两个问题混在一起。Raw trend 是描述性结果；AUROC trend 才是指标有效性的主要证据。

### 10.2 为什么不能只看 raw trend

跨 checkpoint 的 hidden 几何不能直接视为同一坐标系下的可比物理量。训练会带来：

```text
hidden norm 漂移
表示各向异性变化
response length 变化
num_chunks 分布变化
输出模板化程度变化
overall correct/incorrect 比例变化
```

例如，若训练后 `path_length` 下降，可能表示正确的语义收敛，也可能只是：

```text
response 变短
hidden norm 变小
模型输出更模板化
overall 中 correct rollout 占比上升
```

特别是 overall mean 有机械混合效应：

```text
overall_mean(step)
= acc(step) * mean_correct(step)
+ (1 - acc(step)) * mean_incorrect(step)
```

只要 accuracy 上升，overall 曲线就会自然向 correct 曲线靠近。因此：

```text
overall raw trend 可以画，但不能作为主结论。
真正有解释力的是 correct 子集、incorrect 子集各自怎么变，以及两者的可分性 AUROC 怎么变。
```

### 10.3 数据设置

第一版不建议直接全量跑。建议先做小规模 checkpoint sweep：

```text
eval prompts:
  MathVerse mixed MCQ subset 中固定抽 100-300 题

rollouts:
  4 或 8 rollouts/question

decoding:
  temperature = 0.7
  top_p = 0.95
  max_tokens = 1024
  query field 与当前 MathVerse CoT 实验保持一致

chunk:
  chunk_size = 256
  layers = 24, 36
```

Checkpoint 来源是实验 F 的前置条件。理想输入是同一个训练 run 的中间 checkpoint：

```text
step 0 / 20 / 40 / 60 / 80 / 100 / 120 / 140 / 160
```

其中 step 0 最好是当前使用的 instruct checkpoint 或训练前 checkpoint，这样实验 F 能和现有 MathVerse 结果直接衔接。

如果暂时没有完整训练轨迹，可以退一步做阶段模型先导验证：

```text
base -> instruct -> RL / SFT checkpoint
```

但这种阶段模型实验只能说明不同模型阶段的差异，不能严格说明同一次训练过程的 step dynamics。

### 10.4 主指标

第一版主图只放少量高信号指标，避免多重比较和噪声淹没。

推荐 primary geometry metrics：

```text
M3_L36_path_length       score = -path_length
E_L36_d_late_mean        score = -d_late_mean
A_L36_ERV_adj_late_min   direction 按当前 checkpoint 内验证，同时报告正反两向
```

推荐 behavior anchors：

```text
accuracy
response_length
num_chunks
```

`cos_mean` / angular metrics 第一版作为 exploratory，不进入主 claim。原因是当前固定 checkpoint 上 angular 信号较弱，AUROC 约 0.55-0.58，在 100-300 题的小规模 step sweep 中容易被 bootstrap 噪声淹没。

### 10.5 主评估：within-question AUROC(step)

对每个 checkpoint `s`、每个指标 `m`，在每道 mixed-correctness 题内部计算：

```text
AUROC_s,m(q) = P[ score_s,m(q, y_correct) > score_s,m(q, y_wrong) | same q ]
```

然后对有效题目取平均：

```text
MeanAUROC_s,m = mean_q AUROC_s,m(q)
```

主图：

```text
x-axis: training step
y-axis: mean within-question AUROC
lines:
  -path_length
  -d_late_mean
  ERV feature
shade:
  bootstrap CI by question
```

这个图回答：

```text
在策略训练和 checkpoint 漂移下，hidden trajectory diagnostic 是否仍能稳定区分同题内正确/错误 rollout？
```

理想 positive finding：

```text
accuracy 上升时，-path_length / -d_late_mean / ERV 的 within-question AUROC 保持稳定或上升。
```

这说明这些指标不是只在单个 checkpoint 偶然有效，而是在训练过程中仍然是可用的 semantic risk diagnostic。

### 10.6 辅助评估：raw trend

为了接近训练动力学图，可以画 raw metric mean over step：

```text
x-axis: training step
y-axis: metric mean
lines:
  overall
  correct rollouts
  incorrect rollouts
```

建议画：

```text
ER_local mean
ERV late_min
path_length
d_late_mean
response_length
accuracy / reward / critic score
```

但解释时必须区分：

```text
raw trend:
  说明训练改变了 hidden trajectory 形态。

AUROC trend:
  说明该形态变化是否与 correct vs incorrect 可分性有关。
```

如果 raw mean 下降，但 AUROC 不变或下降，只能说训练改变了几何统计量，不能说它变成了更好的 correctness diagnostic。

### 10.7 跨 checkpoint 校准原则

所有 position-conditioned calibration 必须在每个 checkpoint 内重算：

```text
mean_s,k
std_s,k
zMetric_s,k = (Metric_s,k - mean_s,k) / std_s,k
```

不能把 step 0 的 calibration 套到 step 160，因为训练会改变：

```text
hidden norm
长度分布
chunk position 分布
answer style
```

对 L2 类指标，建议同时报告 normalized variants：

```text
path_length / num_steps
d_late_mean
cosine path distance
```

以减少 response length 和 hidden norm 漂移对 raw L2 的影响。

### 10.8 漂移诊断

每个 checkpoint 都需要额外记录 hidden geometry drift diagnostics：

```text
mean hidden norm
std hidden norm
mean chunk-step norm
mean pairwise chunk cosine distance
global anisotropy / participation ratio
num_chunks distribution
response_length distribution
accuracy
```

这些诊断不是 reward 指标，而是用来判断 raw metric trend 是否可能来自表征尺度漂移或各向异性变化。

如果出现：

```text
path_length raw mean 下降
mean hidden norm 同时明显下降
```

则不能直接解释为语义轨迹收敛。需要依赖 checkpoint 内 AUROC 或 standardized gap 来支持 correctness-related 解释。

### 10.9 主要图表

#### 图 F1：Behavior Anchors

```text
x-axis: training step
y-axis left: accuracy / reward / critic score
y-axis right: response length 或 num_chunks
```

作用：

```text
确认训练确实改变了行为层表现，并记录长度是否变化。
```

#### 图 F2：Raw Hidden-Trajectory Trends

```text
panels:
  ER_local mean
  ERV late_min
  path_length
  d_late_mean

lines:
  overall
  correct
  incorrect
```

作用：

```text
描述训练过程中 hidden geometry 的变化。
```

注意：

```text
overall line 只作描述，不作为主 claim。
```

#### 图 F3：Within-Question AUROC Over Training

```text
panels or lines:
  -path_length
  -d_late_mean
  ERV late_min

x-axis: training step
y-axis: mean per-question AUROC
shade: bootstrap CI
```

这是实验 F 的核心图。

#### 图 F4：Correct-Incorrect Standardized Gap

在每个 checkpoint 内做 position / feature 标准化后，计算：

```text
gap_s,m = mean(score_correct) - mean(score_incorrect)
```

或：

```text
gap_s,m = Cohen's d within checkpoint
```

作用：

```text
作为 AUROC 的补充，显示 correct/incorrect 的相对距离是否随训练扩大。
```

#### 图 F5：Geometry Drift Diagnostics

```text
mean hidden norm
participation ratio / anisotropy
num_chunks
response length
```

作用：

```text
解释 raw metric trend 是否受到 global geometry drift 污染。
```

### 10.10 可能结论

最强 positive finding：

```text
As training improves accuracy, hidden trajectory metrics such as -path_length and -d_late_mean
remain stable or become stronger within-question discriminators of correct vs incorrect rollouts.
This suggests that semantic trajectory stability is not a checkpoint-specific artifact,
but a robust diagnostic under policy drift.
```

如果 raw trend 变化明显但 AUROC 不变：

```text
Training changes the global hidden geometry, but the correctness-discriminative content of the metric
does not improve. Raw ER/path trends should be interpreted as representation drift or behavior shift,
not necessarily as a better semantic risk signal.
```

如果 AUROC 下降：

```text
The metric is checkpoint-sensitive and may not be reliable as a fixed reward/risk signal during RL.
It may require checkpoint-specific calibration or a trained verifier.
```

### 10.11 与 RL roadmap 的关系

实验 F 对后续 RL 最重要的不是证明某个 raw metric 随训练下降，而是证明：

```text
在训练导致策略分布变化后，该指标仍然能在同题内区分正确和错误 rollout。
```

如果 `-path_length`、`-d_late_mean` 或 ERV 在多个 checkpoint 上 AUROC 稳定高于 0.5，并且没有被 hidden norm / length drift 完全解释，那么它们才更适合作为：

```text
semantic risk feature
reward shaping diagnostic
u_k 的候选输入
```

因此实验 F 是从离线诊断走向 RL reward design 的前置稳定性检查。

---

## 11. 实验 G：Cross-Chunk Angular Dynamics（宏观语义转向）

### 11.1 目标

实验 G 把实验 E 的 cross-chunk 位移序列补上"方向"分量。实验 E 只算了相邻位移的大小 `d_k = ||p_{k+1} - p_k||`（path_length / d_late_mean），完全没用方向信息。实验 G 计算相邻 cross-chunk 位移之间的转角，验证：

```text
在 reasoning 的宏观尺度（中等粒度点之间）上，hidden-state 轨迹的方向是否稳定？
正确 rollout 的宏观语义轨迹是否更"直"（朝一致方向推进），
错误 rollout 是否在 reasoning 步骤之间频繁改变方向（折返 / 震荡 / 急转）？
```

它和实验 D 的区别是尺度，不是公式：

```text
实验 D：chunk 内部，micro-window（8 token）尺度的局部方向抖动
实验 G：贯穿全 response，中等粒度（~64 token）点之间的宏观方向转向
```

实验 D 测的是"局部用词级别的抖动"，实验 G 测的是"推理步骤级别的转向"。这填补了 ER.md 第 12 节三维分解中"方向"维度在宏观尺度上的空白：

```text
magnitude（幅度）：实验 E path dynamics（已做）
dimensionality（维度）：实验 A/B ER（已做）
direction（方向）：实验 D 局部（已做，弱）+ 实验 G 宏观（本实验，待做）
```

### 11.2 为什么选中等粒度（核心设计决策）

逐 token 和 chunk-256 两端都不可行，中等粒度（~64 token/点）是唯一同时满足"统计稳"和"语义足"的工作点：

```text
micro-window（w=8，即实验 D 现状）：
  位移向量大量反映词汇 / 标点 / tokenization 抖动，信噪比低
  实验 D 最好的 cos_mean 只有 AUROC 0.58，很可能就吃了这个亏

chunk-256（即实验 E 的点粒度）：
  response <= 1536，平均 num_chunks ~= 4
  N 个点 -> N-1 个位移 -> N-2 个角度
  4 个 chunk 只有 2 个角度，2-3 chunk 的 rollout 只有 0-1 个角度
  per-rollout 角度统计量噪声淹没信号，会死在样本量上

中等粒度（w ~= 64）：
  ~1536 / 64 ~= 24 个点 -> ~22 个角度
  统计上够稳（可算分位数、std）
  64 token ~= 一两句话 / 一个推理小步，位移开始承载"语义动作"而非"用词抖动"
```

<G_PLACEHOLDER>

### 11.3 计算对象

固定某一层 `l`（last layer 36 或 middle-late layer 24）。

第一步，把整条 response 按中等粒度 `w` 池化成一串轨迹点（贯穿全 response，不再以 256-chunk 为单位）：

```text
p_j^l = pool(tokens [j*w, ..., (j+1)*w-1] 的第 l 层 hidden)    j = 0, 1, ..., M-1
```

其中 `pool` 有两种表征（见 11.6 sweep）：

```text
mean：window 内 token hidden 的平均
last：window 内最后一个 token 的 hidden（因果注意力下是该段的状态快照）
```

第二步，cross-point 位移：

```text
D_j^l = p_{j+1}^l - p_j^l        j = 0, ..., M-2
```

第三步，相邻位移的转角（与实验 D 同公式，但作用在宏观点上）：

```text
cos_j = <D_j, D_{j-1}> / (||D_j|| ||D_{j-1}||)
theta_j = arccos(clip(cos_j, -1, 1))        j = 1, ..., M-2
```

直觉：

```text
theta 小 / cos 高：宏观轨迹朝一致方向推进，reasoning 方向稳定
theta 大 / cos 低或负：reasoning 在步骤之间折返、震荡、急转
```

### 11.4 指标定义

#### Rollout-level 全局标量（每条 rollout 一个值）

```text
gcos_mean      = mean(cos_j)              宏观方向一致性（越高越稳）
gcos_std       = std(cos_j)               方向稳定性的波动
gcos_p10       = p10(cos_j)               极端转向事件
gcos_min       = min(cos_j)               最差单步转向
gtheta_mean    = mean(theta_j)            平均转角
gspike_rate_90 = P(theta_j > pi/2)        超过正交的急转弯比例
```

#### Late-phase 标量（对齐实验 D/E 的 late 设计）

```text
gcos_late_mean = mean(cos_j for last 25% points)
gcos_late_min  = min(cos_j for last 25% points)
gtheta_late_mean = mean(theta_j for last 25% points)
```

#### Chunk-to-chunk 速度（angular velocity）

```text
AV_gcos(j) = gcos_running(j) - gcos_running(j-1)
```

其中 `gcos_running` 可用 adjacent 或 historical baseline，与实验 E 的 pv / d_hist 设计一致。

### 11.5 Primary vs Secondary（预注册）

为避免多重比较，pre-commit 以下为 primary（正式报告 within-question AUROC + CI）：

```text
Primary:
  gcos_mean        (+方向：越高越对)   宏观方向一致性，最干净的指标
  gcos_late_mean   (+方向：越高越对)   后期方向稳定性，对齐 late-phase 故事
  gcos_p10         (+方向：越高越对)   极端转向事件
  gtheta_late_mean (-方向：越低越对)   后期转角，gcos_late_mean 的角度对偶
```

其余（`gcos_std / gcos_min / gspike_rate_90 / AV_gcos / historical 变体`）为 secondary / exploratory。

### 11.6 粒度与表征 sweep（本实验的核心变量）

实验 G 第一版就是一个 sweep，目的是定出最优的 (粒度 w, 表征) 组合。一次 forward 同时存所有组合，池化是白送的，forward 是唯一瓶颈。

```text
粒度 sweep：    w in {32, 64, 128}
表征 sweep：    pool in {mean, last}
层 sweep：      l in {24, 36}
```

共 3 x 2 x 2 = 12 种组合。预期：

```text
w=8（实验 D 现状，作为下界对照）：信噪比低
w=32：可能仍偏抖
w=64：预期甜点位
w=128：点数 ~12，角度 ~10，可能偏少但更宏观
last 预期 >= mean：last 是因果状态快照，mean 把段内早期未整合 token 也平均进去
```

报告时为每个 primary 指标画 AUROC vs w 的曲线（mean / last 两条线，layer 24 / 36 分面），选出最优组合后再做正式 claim。

### 11.7 数据来源

**需要单独重新 forward。** 现有 `chunklayer/h_gpu*.npz` 存的是 256-token chunk 的 mean，无法还原 32/64/128 粒度的 mean 或任意 last-token（更细粒度的 mean 不能从更粗的 mean 还原，last 更不能）。

forward 时一次性保存所有 sweep 需要的池化点：

```text
对每条 rollout、每个 sweep 层 l in {24, 36}：
  存 p_mean^(w)  for w in {32, 64, 128}
  存 p_last^(w)  for w in {32, 64, 128}
只存中等粒度池化点（每条 rollout 几十个 d=4096 向量），不存全 token hidden。
```

forward 逻辑可复用实验 D 脚本（`run_local_angular_qwen3vl.py`）的 response 边界定位（`common_prefix_len`）和 hidden 抽取部分，只把池化粒度和聚合方式换成 sweep 版。

### 11.8 与 ER 的结合（ER + angular）

实验 G 的一个主要动机是和 ER 互补。ER 衡量 semantic breadth（维度），angular 衡量 direction（方向），两者正交。

#### 对齐方式

ER_local 是 per-256-chunk 一个值，而 gcos_j 是中等粒度点序列。结合时把两者都聚合到 rollout-level 或对齐到同一 late-phase 窗口，避免索引错位：

```text
rollout-level：
  ER 特征：erv_adj_late_min（实验 A 最强，AUROC 0.69）
  angular 特征：gcos_late_mean（实验 G primary）
```

#### 评估增量价值（而非手调加权和）

不要只画 ER x angular 的二维 scatter 然后调权重。正式评估应回答"在已有 path + ER 基础上，宏观 angular 是否还有独立贡献"：

```text
1. within-question 偏 AUROC：
   控制 path_length / erv 后，gcos_late_mean 是否仍有区分力

2. 逻辑回归独立系数：
   is_correct ~ z_path + z_erv + z_gcos_late_mean
   看 gcos 项的系数是否显著、符号是否符合预注册方向

3. combo AUROC：
   combo = standardize(path) + standardize(erv) + standardize(gcos_late_mean)
   是否超过 path + erv 的 0.7153
```

#### 一个独立价值点：尺度不变性

cosine / 角度自带 norm 归一化，**不受 hidden-norm 漂移影响**。这意味着：

```text
实验 F（训练 step）中，path_length 跨 checkpoint 不可比（norm 漂移），
但 gcos 系指标天然跨 checkpoint 可比。
因此宏观 angular 不只是"再加一点 AUROC"，
它还是训练动力学分析中更鲁棒的方向信号。
```

### 11.9 主要图表

```text
图 G1：Correct vs Incorrect 的 gcos_j 曲线（x: relative position, y: cos, 两条线 + CI）
图 G2：gcos 的 within-question AUROC by point position
图 G3：AUROC vs 粒度 w 的 sweep 曲线（mean / last 两条线，layer 24 / 36 分面）—— 本实验核心图
图 G4：gcos_late_mean x erv_adj_late_min 二维散点（按 correctness 着色），看互补性
图 G5：combo（path + erv + gcos）vs path + erv 的 AUROC 对比
```

### 11.10 可能结论

理想 positive finding：

```text
At a mid-range pooling scale (~64 tokens), incorrect rollouts exhibit lower macro-level
directional consistency: their cross-point semantic trajectory turns more between reasoning
steps, especially in late phase. This macro angular signal is stronger than the micro-window
angular signal of Experiment D, is orthogonal to path length and ER, and adds incremental
within-question discriminative power.
```

negative finding 也有价值：

```text
Macro angular consistency does not exceed the micro-window result, or is fully explained
by path length / ER. Then direction at the macro scale carries little independent signal
beyond magnitude and breadth, and the displacement magnitude (path) already captures
most of the trajectory-stability story.
```

最弱但仍有用的结论（尺度不变性）：

```text
Even if macro angular adds little static AUROC, its norm-invariance makes it a more
robust trajectory signal across training checkpoints than path_length.
```

---

## 12. 预期论文叙事

如果结果支持，可以这样组织：

```text
Existing hidden-state methods often compress a full response into a scalar,
but such global summaries are weak under long-CoT or multi-step reasoning settings.

Inspired by semantic-space dynamics, we study three complementary properties
of hidden-state trajectories:

1. semantic displacement dynamics: how far and how fast the trajectory moves (path velocity/acceleration);
2. semantic breadth dynamics: how the effective dimensionality expands and contracts (ER/ERV/ERA);
3. semantic turning dynamics: whether the trajectory direction is stable or oscillating (angular metrics).

These three views decompose trajectory behavior into magnitude, dimensionality, and direction.
```

进一步可以提出：

```text
u_k should be a calibrated semantic risk function,
not a raw hidden-state geometry score.
```

核心 novelty：

```text
1. 把 hidden-state trajectory 从全局标量拆解为 chunk-level 时序动态。
2. 区分位移量（path dynamics）、语义宽度（ER dynamics）、方向稳定性（angular dynamics）三个维度。
3. 证明时间模式（velocity/acceleration）比静态聚合（sum/mean）提供更强信号。
4. 用 within-question discriminativeness 验证指标真的能区分同题好坏轨迹。
5. 为后续 RL credit assignment 提供 calibrated semantic risk signal。
```
