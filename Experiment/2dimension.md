# 2Dimension：Long-Response 横向轨迹 × 纵向层选择的 Hidden-State Credit 实验

## 0. 研究目标与范围

本文档规划一组只使用 **long response** 的发现性实验，目标是回答：

```text
能否从长推理 rollout 的 hidden-state 轨迹中构造一个局部 credit 候选信号：
横向使用随生成进度发生的表示位移，纵向使用层间表示重组来选择重要层，
并最终服务于 GRPO / RLVR 的 span-level reward shaping？
```

当前阶段只做离线发现与验证，不修改 RL 训练。训练期允许使用 ground-truth 和同题 correct/wrong
rollout 构造 privileged reference，但任何候选分数在给 query rollout 打分时都不能读取 query 自身标签。

所有主实验固定使用第 1 节的 Qwen3-VL-8B-Thinking long-response setting。短 Instruct rollout、短
`h_chunk` 及其派生结果只能作为历史背景，不进入主分析、特征选择或成功判据。

执行顺序固定为：

```text
实验 0：多分辨率 hidden dynamics discovery；token-level 测幅度/熵，span-pooled 测角度/跨 rollout 方向。
实验 1：Long span 两层 pilot；验证 success-trajectory 的角度和幅度加权角度。
实验 2：复用实验 1 的 long hidden 特征及现有 long baseline，定位增量来源，不新增 forward。
实验 3：仅在实验 1+2 通过后做 long span 全层 forward，验证纵向 L2 / spectral-entropy selector。
```

---

## 1. Long-Response Setting 与现有数据

### 1.1 固定 setting

主实验统一采用：

```text
model                 Qwen/Qwen3-VL-8B-Thinking
dataset               MathVerse long-response smoke500 MCQ
rollouts/question     8
generation max_tokens 16384
primary subset        non-truncated + 可定位完整 think closing boundary
analysis segment      response 起点到 </think> 之前的完整 think token 段
answer segment        只用于判定 correctness，不进入 hidden 轨迹
```

Qwen3-VL-Thinking 的 opening `<think>` 可能是隐式的，因此 token 边界按“response 起点到 `</think>`”确定，
不能依赖字符串切分。没有可靠 closing boundary 的 rollout 不进入主分析，单列为 segment sensitivity。

**不设置 `think_length >= 2k` 等硬阈值。** 主分析保留所有符合上述条件的 rollout，并固定按以下长度桶
分层报告和控制：

```text
<2k / 2-4k / 4-8k / 8k+ think tokens
```

这样避免按生成后的长度筛选造成偏差，也避免破坏同题 correct/wrong group。truncated rollout 不混入主分析，
但必须单列报告其长度、正确率与候选指标，量化 truncation selection bias。

### 1.2 H200 上已确认的数据

2026-07-21 已在 H200 上只读确认：

```text
/data2/hjk/projects/AI-HiddenState-ER/data_long/
  rollouts_thinking_smoke500_mt16384_labeled.jsonl

/data2/hjk/projects/AI-HiddenState-ER/long_cot_smoke/smoke500_mt16384/
  LONG_SMOKE_REPORT.md

/data2/hjk/projects/AI-HiddenState-ER/long_trajectory_smoke/smoke500_clean/
  LONG_TRAJECTORY_SMOKE_RESULTS.md
  long_path_*.parquet
  long_macro_angular*.parquet
  long_local_angular*.parquet
```

数据概况：

```text
questions                         500
rollouts                          4000
clean non-truncated rollouts      3858
truncated rollouts                142（3.55%）
clean mixed-correctness questions 206
mean response tokens              4649.9
mean think tokens                 3742.4
p50 / p90 think tokens            2554 / 8949
max think tokens                  16350
```

在 3858 条 clean rollout 中，同题至少 `2 correct + 2 wrong` 的问题约 125 个，至少 `3+3` 的问题约
72 个。正式提取前必须从源 JSONL 重新生成 eligibility audit，以实际完成 think-boundary 过滤后的数量为准。

### 1.3 旧 short `h_chunk` 不进入主实验

旧缓存：

```text
/data2/hjk/projects/AI-HiddenState-ER/chunklayer/h_gpu{0..7}.npz
```

来自 Qwen3-VL-8B-Instruct 短回答。直接读取元数据得到：

```text
rollouts               8936
median response tokens 1026
max response tokens    1031
response >= 1536       0
```

因此它不能支持本项目的 long-response 主张。旧缓存只用于回顾短/长 setting 差异，不用于实验 0 的
多分辨率 discovery、实验 1 的四格方向、实验 2 的增量模型或实验 3 的 selector。

### 1.4 Long 数据目前缺少什么

现有 long parquet 保存了 path、macro angular、local angular 等标量，没有保存 span-mean、span-last 或
任意可重算的 hidden vector，也没有全层 token/span 谱统计。因此：

- long rollout 不需要重新生成；
- 实验 0 需要在有界 discovery 子集上做一次全层 hidden forward，但 token-level 结果在线压缩为标量，
  跨 rollout 的 span vectors 只在单题内临时保留；
- 实验 1 对实验 0 未覆盖的问题只补做 L24/L36 最小 hidden forward，并复用重叠问题的兼容 span 输出；
- 实验 2 复用实验 1 输出和现有 long 标量，不新增 forward；
- 实验 3 只有通过 gate 后才做全层 hidden forward。

---

## 2. 既有 Long-Response 结果给出的约束

以下是设计先验，不把它们误写成普适结论：

```text
[C1] long-CoT 的全局 path 信号明显弱于旧短 CoT；长度与相对进度控制是刚需。
[C2] prompt 加/不加答案选项构造的 correct direction 在 think 段约为随机，不能直接复用。
[C3] long-CoT 的 macro angular 有弱信号，但局部相邻窗口 cosine 基本随机。
[C4] 运动幅度可能只在特定进度或长度桶有效；纯 cosine 可能因小位移而不稳定。
[C5] layer 变化更适合作为横向信号的 selector，而不是单独称为“纵向语义”。
```

同一 long-response 数据上的参照值：

```text
L36 path_length                         best AUROC 0.5820
L24 path_length                         best AUROC 0.5761
L24 w=32 last macro gcos_min            AUROC 0.5650
L36 w=64 last macro gcos_min            AUROC 0.5643
local angular                           约 0.50
long option-logit think_final_margin    AUROC 约 0.789
```

这些 baseline 必须在实验 0/1 的同一 eligible subset 上重新评估后才能比较。option-logit 是外部强参照，
不是 hidden 指标必须超过的停止阈值；新 hidden 指标的目标是提供更局部、可用于 span credit、且在
length/path/logit level 之外仍有增量的信息。

---

## 3. Novelty 边界与研究假设

直接使用 answer likelihood / answer logits 做 token-level credit 已经较拥挤：

- GeneralThinker（arXiv:2605.27934）：使用 ground-truth answer likelihood 派生 token-wise 信号；
- OPPO（arXiv:2605.21851）：使用 oracle-conditioned likelihood ratio 和递归 success value；
- Where Hindsight Credit Can Reside（arXiv:2604.11056）：研究 RLVR 中 hindsight credit 的 signed capacity。

因此本计划不把 answer logit 本身作为方法，只将它用于 baseline 和条件增量控制。

待验证的假设为：

```text
[H1] long-response 横向 hidden 位移包含 rollout 是否沿某条成功轨迹推进的信号。
[H2] 纯角度可能很弱，但“位移幅度 × 成功轨迹选择性”可能有增量。
[H3] 成功轨迹不是唯一均值方向；相对 correct/wrong reference manifold 的选择性更稳妥。
[H4] span 内 centered spectral entropy 的层间变化可以选择应信任的层。
```

这里的“横向”应严格称为 token-progress dynamics。它可能包含语义、token identity、上下文积累和
hidden norm 变化，不能在实验前直接等同于语义。

---

## 4. 共用符号与防泄漏协议

### 4.1 Span 表示、横向位移与相对进度

对问题 `q` 的 rollout `i`、span `k`、layer `l`，主切分固定为：

```text
window = 128 think tokens
stride = 64 think tokens
direction primary        span-mean hidden
direction control        span endpoint last-token hidden
```

主分析只保留完整的 128-token span；末尾不足 128 token 的 partial span 不做 padding，也不进入主分数，
但必须报告每条 rollout 被覆盖的 think-token 比例。这样保证不同 span 的 SVD rank 和 mean 方差可比。

记 span 表示为：

\[
g_{i,k,l}\in\mathbb R^d,
\qquad
d_{i,k,l}=g_{i,k,l}-g_{i,k-1,l}.
\]

span-mean 先对局部窗口去噪，再计算位移和角度；last-token 实际对应每 64 token 采样一次 endpoint，
用于检查结果是否依赖窗口平均。角度流的主/对照地位必须在实验 0 discovery 结束后、任何 locked evaluation
之前冻结，不能在 confirmatory 结果出来后交换。

相对进度定义为：

\[
\rho_{i,k}
=
\frac{\operatorname{span\_center}_{i,k}}{\operatorname{think\_length}_i}.
\]

主分析使用 10 个等宽 relative-progress bins；5-bin 版本作为敏感性分析，无 progress matching 版本只作
消融。先在 `rollout × progress-bin` 内聚合，再让 progress bin 等权，避免长回答因 span 更多而自动
获得更大权重。

### 4.2 Discovery/confirmatory 问题划分

这里的 `cohort` 指按同一套 eligibility 条件筛出、承担同一种分析用途的一组题。实验 1 的
**primary cohort（主分析题集）**要求满足主 segment 条件且至少有 `3 correct + 3 wrong`。这样任意一条
rollout 作为 query 被剔除后，两侧仍至少各有 2 条 reference，能够保留“同题存在多条正确路径”这一结构。

现有 clean 数据中约 72 题满足 `3+3`；正式数字以 complete-think eligibility audit 为准。至少 `2+2`
但不满足 `3+3` 的问题只进入 **secondary coverage cohort（补充覆盖题集）**，因为它们在
leave-one-out 后每侧可能只剩一条 reference，不能可靠代表多路径集合。

在查看新方向指标之前，按主分析题集的 `question_id` 固定划分：

```text
discovery pool      70%，供实验 0/1/2 pilot、扩展、选层和特征消融
confirmatory pool   30%，在实验 3 规格冻结前不计算新 2Dimension 指标
```

划分按 correct/wrong 数量和 think-length profile 分层，并固定随机种子。既有 long path/logit 报告已经看过
全量数据，因此 held-out 的含义是“未用于新 2Dimension 指标的选择”，不是声称这些问题从未分析过。

若过滤后 eligible questions 太少，导致 30% confirmatory pool 无法给出有效 CI，则改用全量 nested
grouped CV；此时实验 3 必须称为第二阶段 discovery，不能称为 confirmatory。

### 4.3 Balanced leave-one-rollout-out reference

每题只有 8 条 rollout，若使用 2-fold，会把本来就很少的正确路径再砍掉一半。主分析因此使用
**balanced leave-one-rollout-out（balanced LOO）**，把每条 rollout 依次作为 query，并从其余 rollout
构造 observed success/failure reference sets。

对问题 `q`，设最终正确/错误 rollout 数分别为 `n_q+`、`n_q-`，固定每侧 reference 数量：

\[
m_q=\min(n_q^+-1,n_q^--1).
\]

对 query rollout `i` 和标签 `b`：

\[
R_{q,i}^{b}
=
\{j:j\ne i,\ y_j=b\}.
\]

从 `R+`、`R-` 各选择恰好 `m_q` 条 reference，并穷举所有 balanced subset pairs：

\[
\mathcal B_{q,i}
=
\binom{R_{q,i}^{+}}{m_q}
\times
\binom{R_{q,i}^{-}}{m_q}.
\]

每题总共只有 8 条 rollout，在 `3+3`、`3+4`、`3+5` 或 `4+4` 的可行构成下，每个 query 最多只有
18 个 balanced subset pairs，因此无需随机抽样。主 query score 对所有 subset pairs 取均值，并保留
subset-level 分布用于稳定性分析。`m_q` 对该题所有 query 固定，不会因 query 属于 correct/wrong 而产生
集合大小差异。对于恰好 `3+3` 的题，`m_q=2`；正确路径较多的题会在全部 subset pairs 中得到覆盖。

每个 query/balanced-subset 中：

- correct/wrong reference 始终各为 `m_q` 条；
- query 按 rollout id 从两侧 reference 中统一剔除，`++` 和 `--` 均无 self-match；
- 每条 reference rollout 在同 progress bin 中最多贡献一个与 query `rho` 最近的 span displacement；
- query correctness 只用于事后分组评估，不能参与 query score 计算；
- layer、符号、超参数和模型组合只能在 discovery training questions 中选择。

不同 query 会共享部分 reference，因此所有 CI 和 permutation 都以 question 为独立单位，不能把 query/span
当独立样本。严格 stratified 2-fold 作为 sensitivity analysis；naive in-sample 只用于量化 self-match 泄漏。

### 4.4 四格角度与对称交互

令 `a` 表示 query 标签，`b` 表示 reference 标签，`+` 为 correct，`-` 为 wrong。对 query 位移，
从同题、同 progress bin、标签为 `b` 的等量 reference 候选中取 top-K cosine 的均值：

\[
S_i^b
=
\operatorname{meanTopK}_{j\in R_b}
\cos(d_i,d_j).
\]

主分析固定 `K=1`，即 nearest observed path；`K=2` 和 `K=all` 作为 top-2 / mean-reference 消融。定义：

这里的 `R+` 是**多条 observed success trajectories 的集合**，不是一条唯一正确轨迹。`K=1` 的含义是：
只要 query 当前移动接近任意一条已观察到的成功路径，就认为它仍有 success-path support。由于每题只有
8 条 rollout，这个集合不能覆盖所有真实正确解法，因此全文只能称 observed success reference set，不能
称完整的 correct manifold。

另外，final-answer correct 只定义 success-conditioned rollout，不保证其中每个 span 都逻辑正确。正确
rollout 可能先走错再修正、绕路或猜对；因此实验 1 检验的是“当前移动是否得到已观察成功路径的支持”，
不能把低 `S+` 的位置直接称为逻辑错误点。

\[
A_{\rm angle}^{ab}
=
\mathbb E[S_i^b\mid y_i=a].
\]

必须同时报告：

\[
A^{++},\quad A^{+-},\quad A^{-+},\quad A^{--}.
\]

query 对 observed correct/wrong reference sets 的选择性分数为：

\[
D_{i,\rm angle}=S_i^+-S_i^-.
\]

核心对称交互为：

\[
I_{\rm angle}
=(A^{++}-A^{+-})-(A^{-+}-A^{--}).
\]

只检验 `A++ > A-+` 会产生“正确组本来就更紧”的自证风险；四格交互要求 correct query 相对偏向
observed correct reference set 的程度，显著强于 wrong query 的这种偏向。

### 4.5 幅度加权版

纯 cosine 在 `||d_i||` 很小时不稳定，因此必须同时计算 query-amplitude 加权版：

\[
P_i^b=\|d_i\|_2S_i^b,
\qquad
D_{i,\rm amp}=P_i^+-P_i^-
=\|d_i\|_2D_{i,\rm angle}.
\]

对应交互为：

\[
I_{\rm amp}
=(A_{\rm amp}^{++}-A_{\rm amp}^{+-})
-(A_{\rm amp}^{-+}-A_{\rm amp}^{--}).
\]

第一版不乘 reference 的 `||d_j||`，避免高幅度 reference 主导 nearest-neighbor。每层单独报告 raw
幅度版；跨层组合时再用 discovery training data 中同 layer、同 progress bin 的 median/IQR 校准。

---

## 5. 实验 0：多分辨率横纵 Hidden Dynamics Discovery

### 5.1 目的与边界

实验 0 不先构造一个统一分数，而是分别回答三个更基础的问题：

```text
E0-A  单条 rollout 内，横向 movement length、length change 是否在 correct/wrong 间存在差异？
E0-B  progress-matched 的跨 rollout 长度支持和方向支持，是否更接近正确而非错误轨迹？
E0-C  同一 token 的纵向 layer update norm、centered entropy 与层间转向是否包含正确性结构？
```

这里采用按统计量选择分辨率的 Hybrid C：

```text
token-amplitude stream   横向/纵向 L2、长度变化、centered entropy；在线计算标量后聚合到 bin
span-direction stream    横向转折角、纵向层更新角、跨 rollout 方向；span pooling 后再计算
common analysis unit     question × rollout × relative-progress-bin × layer
```

理由是高维空间中的逐 token 位移方向容易集中在近似正交背景附近；角度类量先做 span pooling 可以降低
token identity 与局部噪声。L2 norm 和 entropy 也受 layer scale 与高维集中影响，但不需要保存方向向量，
可以在线计算后用同 layer、同 progress bin 的 robust normalization 控制。

实验 0 只做 discovery atlas、correct/wrong 差异和 SNR 诊断，不做 RL，不把多个量相乘，也不以 standalone
AUROC 最大的组合直接进入实验 1。最终答案标签只用于分组评估和构造 leave-one-out reference；query 自身标签
不能进入其候选分数。

### 5.2 数据、cohort 与 forward

复用第 1.1 节的固定 long rollout，不重新生成：

```text
smoke                         8 个 primary discovery 问题
formal discovery             16-24 个 primary discovery 问题
eligibility                   clean complete-think + 每题至少 3 correct / 3 wrong
layers                        embedding output + 全部 transformer hidden-state indexes
analysis segment              完整 think token 段
new generation                no
```

题目按 correct/wrong 数量和 `<2k / 2-4k / 4-8k / 8k+` 长度桶分层抽取。实验 0 可以与实验 1 共用 discovery
问题，因为二者都属于规格发现阶段，但 confirmatory pool 在所有分辨率、layer、符号和统计口径冻结前保持不可见。

2026-07-22 已完成的实验 1 原始运行把 `span-last` 声明为 primary、`span-mean` 声明为 sensitivity。实验 0 是
在查看该结果后加入的，因此不能追溯性地把那次运行改称“预注册的 span-mean 主结果”。实验 0 可以为后续 locked
evaluation 冻结新的 span-mean angle 规格，但论文和报告必须保留这一时间顺序。

### 5.3 Token-Amplitude Stream：横向/纵向长度与 Centered Entropy

对 rollout `i`、生成 token `t`、layer `l`，横向逐 token 更新为：

\[
\delta^{\rm time}_{i,t,l}=h^l_{i,t}-h^l_{i,t-1},
\qquad
r^{\rm time}_{i,t,l}=\|\delta^{\rm time}_{i,t,l}\|_2.
\]

“横向长度差”单独定义为相邻 movement length 的一阶差分，而不是把 `r` 本身也称为长度差：

\[
\Delta r^{\rm time}_{i,t,l}
=r^{\rm time}_{i,t,l}-r^{\rm time}_{i,t-1,l}.
\]

固定 token、沿 layer 的纵向更新为：

\[
\delta^{\rm layer}_{i,t,l}=h^l_{i,t}-h^{l-1}_{i,t},
\qquad
r^{\rm layer}_{i,t,l}=\|\delta^{\rm layer}_{i,t,l}\|_2,
\]

\[
\Delta r^{\rm layer}_{i,t,l}
=r^{\rm layer}_{i,t,l}-r^{\rm layer}_{i,t,l-1}.
\]

实验 0 的 entropy 作用于纵向更新向量，而不是原始 residual hidden，也不是实验 3 的 span spectral entropy。
先沿 hidden coordinate 中心化：

\[
z_{i,t,l,j}
=\delta^{\rm layer}_{i,t,l,j}
-\frac{1}{d}\sum_{m=1}^{d}\delta^{\rm layer}_{i,t,l,m}.
\]

再把中心化激活变成非负能量分布：

\[
p_{i,t,l,j}
=\frac{z_{i,t,l,j}^2}{\sum_m z_{i,t,l,m}^2+\epsilon},
\]

\[
H^{\rm coord}_{i,t,l}
=-\frac{\sum_jp_{i,t,l,j}\log(p_{i,t,l,j}+\epsilon)}{\log d},
\qquad
N^{\rm eff}_{i,t,l}
=\exp\left(-\sum_jp_{i,t,l,j}\log(p_{i,t,l,j}+\epsilon)\right).
\]

`Hcoord` 高表示 layer update 能量分布在较多 coordinates，低表示集中在较少 coordinates。不能使用
`softmax(h-mean(h))` 声称完成了中心化熵，因为 softmax 对统一平移不变，减均值不会改变结果。
`Hcoord` 依赖模型学到的 hidden-coordinate basis，因此只作为同一模型内的 discovery 指标；不能把它
解释成旋转不变的表示复杂度，也不能直接与实验 3 的 span spectral entropy 混称为同一种 entropy。

每个 `rollout × progress-bin × layer` 至少保存：

```text
mean / median / P90 horizontal_norm
mean / median horizontal_norm_delta
mean / median / P90 vertical_norm
mean / median vertical_norm_delta
mean centered_coordinate_entropy / effective_dimensions
token_count 与有效差分数量
```

P90 用于保留短暂 activity burst；主比较同时报告 mean 与 median，避免单个极端 token 主导。

### 5.4 Span-Direction Stream：Pooling 后的横向与纵向角度

角度主表示固定为先做 span-mean，再做差分：

\[
g^l_{i,k}
=\frac{1}{|S_{i,k}|}\sum_{t\in S_{i,k}}h^l_{i,t},
\qquad
D^{\rm time}_{i,k,l}=g^l_{i,k}-g^l_{i,k-1}.
\]

主规格为 `window/stride=128/64`；`64/32` 与 `256/128` 只用于实验 0 的 SNR sensitivity。
`span-last` 是 endpoint control。不能先把每个 token displacement 归一化再平均，因为这会让微小噪声位移
和大幅度位移获得相同权重。

单 rollout 横向转折以 cosine 为主统计量：

\[
c^{\rm time-turn}_{i,k,l}
=\frac{(D^{\rm time}_{i,k,l})^\top D^{\rm time}_{i,k-1,l}}
{\|D^{\rm time}_{i,k,l}\|_2\|D^{\rm time}_{i,k-1,l}\|_2+\epsilon}.
\]

角度制只用于解释和绘图：

\[
\theta^{\rm time-turn}_{i,k,l}
=\arccos\left(\operatorname{clip}(c^{\rm time-turn}_{i,k,l},-1,1)\right).
\]

纵向角度同样在 span-pooled 表示上计算：

\[
D^{\rm layer}_{i,k,l}=g^l_{i,k}-g^{l-1}_{i,k},
\]

\[
c^{\rm layer-turn}_{i,k,l}
=\frac{(D^{\rm layer}_{i,k,l})^\top D^{\rm layer}_{i,k,l-1}}
{\|D^{\rm layer}_{i,k,l}\|_2\|D^{\rm layer}_{i,k,l-1}\|_2+\epsilon}.
\]

该量表示连续两层是否沿近似一致的方向更新当前局部语义窗口。它不同于 `||h_l-h_{l-1}||`，也不同于
某层 update 与 residual state 本身的夹角；后者可以作为预声明 control，但不能事后替换主纵向角度。

### 5.5 跨 Rollout：正确/错误长度支持与方向支持

跨 rollout 比较只在同题、同 layer、同 relative-progress bin 中进行。query rollout 按 id 从所有 reference
中剔除，并对 correct/wrong reference 数量做等量 balanced LOO。

对每个 query span，每条 reference rollout 在该 bin 内最多贡献一个与 query 的 `rho` 最近的 span；若该
rollout 在该 bin 中没有可用 span，则不进入该 query 的 reference set。下式中的集合运算是简记：实际计算时，
先在每个 correct/wrong 等量子集内分别得到 score，再对预声明的 balanced subsets 取平均，避免较长 rollout
或 span 更密的 rollout 获得更大权重。

对 span displacement 单位方向：

\[
U_{i,k,l}=\frac{D^{\rm time}_{i,k,l}}{\|D^{\rm time}_{i,k,l}\|_2+\epsilon}.
\]

主方向支持保留多路径 nearest-reference set：

\[
S^+_{i,k,l}=\max_{j\in G_q^+\setminus i}\cos(U_{i,k,l},U_{j,b,l}),
\qquad
S^-_{i,k,l}=\max_{j\in G_q^-\setminus i}\cos(U_{i,k,l},U_{j,b,l}),
\]

\[
D^{\rm set-dir}_{i,k,l}=S^+_{i,k,l}-S^-_{i,k,l}.
\]

正确/错误单中心只作可解释性对照：

\[
\mu^+_{q,b,l}=\operatorname{normalize}\left(\sum_{j\in G_q^+\setminus i}U_{j,b,l}\right),
\qquad
\mu^-_{q,b,l}=\operatorname{normalize}\left(\sum_{j\in G_q^-\setminus i}U_{j,b,l}\right),
\]

\[
D^{\rm proto-dir}_{i,k,l}
=\cos(U_{i,k,l},\mu^+_{q,b,l})-\cos(U_{i,k,l},\mu^-_{q,b,l}).
\]

若正确路径方向多模态，单中心可能发生向量抵消，因此不能把 prototype 优于 set 或 set 优于 prototype
解释为普遍规律；两者回答的是“公共方向”与“任一可行成功路径”两个不同问题。

跨 rollout 长度使用正标量的几何均值。令 `R=||Dtime||`：

\[
m^+_{q,b,l}
=\exp\left(\frac{1}{|G_q^+\setminus i|}\sum_{j\in G_q^+\setminus i}\log(R_{j,b,l}+\epsilon)\right),
\]

错误组同理得到 `m-`，长度支持差为：

\[
D^{\rm length}_{i,k,l}
=|\log R_{i,k,l}-\log m^-_{q,b,l}|
-|\log R_{i,k,l}-\log m^+_{q,b,l}|.
\]

正值表示 query 的 movement length 更接近正确 rollout 的典型幅度。实验 0 只比较这些 query score 在
correct/wrong rollout 上的分布；实验 1 才使用完整四格交互排除“某一组本来就更紧”的自证风险。

### 5.6 Relative-Progress 对齐、在线计算与永久存储

token 与 span 都映射到共同相对进度：

\[
\rho_{i,t}=\frac{t}{T_i},
\qquad
b_{i,t}=\min(\lfloor B\rho_{i,t}\rfloor,B-1).
\]

主分析使用 `B=10`；`B=20` 只用于观察局部结构是否被 10-bin 过度平滑。不同分辨率只共享 bin key，
不把 token 指标与 span 指标假装成逐点一一对应。

按 question 顺序处理：

```text
单条 rollout forward
→ 在线计算 token-level horizontal/vertical norm 与 coordinate entropy
→ 聚合到 progress bins
→ 临时保留该 rollout 的全层 span-mean / span-last vectors
→ 立即释放 token hidden

该题 8 条 rollout 完成
→ balanced LOO 跨 rollout 长度/方向评分
→ 保存标量表与少量审计样本
→ 删除该题临时 span vectors
```

永久主表的粒度固定为：

```text
question_id / rollout_id / correctness / progress_bin / layer
horizontal_norm / horizontal_norm_delta
vertical_norm / vertical_norm_delta / coordinate_entropy / effective_dimensions
span_turn_cos / span_layer_turn_cos
cross_set_direction / cross_prototype_direction / cross_length_support
think_length / token_count / span_count / segment status
```

不永久保存完整 `[T,L,d]` token hidden。为了复核数值，仅允许固定少量 audit questions 保存 float16 span
vectors；audit 题必须在运行前确定，不能根据效果大小挑选。

### 5.7 可视化、SNR 检验与统计口径

实验 0 的主图不是单一 AUC 排名，而是 correct/wrong dynamics atlas：

```text
E0-F1  horizontal_norm × span_turn_cos scatter / hexbin，按 layer 与 progress 分面
E0-F2  cross_length_support × cross_set_direction scatter，叠加 prototype control
E0-F3  coordinate_entropy × vertical_norm / span_layer_turn_cos scatter
E0-F4  layer × progress 的 correct-minus-wrong Hedges' g heatmap
E0-F5  64/32、128/64、256/128 的 span-mean、token-angle 与 span-last 的 angle SNR / reliability 对照
E0-F6  四个 think-length bins 的 effect curve 与样本覆盖
```

散点的统计单位是 `rollout × progress-bin × layer`，但 CI、置换和模型比较的独立单位始终是 question。
必须同时画 raw 与 within-question centered 版本，防止题目难度和回答长度制造表面分离。

角度 pooling 的收益不能只靠“高维噪声”解释，必须实测：

1. 在固定 audit subset 上把 token-level turning cosine 作为负面对照；
2. 比较各 span 规格的 Hedges' `g`、question-bootstrap CI 与题间符号一致率；
3. 用 matched random-pair cosine 得到同 layer/progress 的正交背景均值和方差；
4. 报告 span angle 的 split-half/reference-bootstrap reliability；
5. 检查收益是否只来自更少样本、更长有效 lag 或单一异常问题。

所有 feature 至少控制 `think_length`、有效 token/span 数、mean hidden/update norm、relative progress 和 layer
scale。题内 label permutation 必须重新计算 correct/wrong reference geometry，不能只在最终表上交换标签。

### 5.8 实验 0 的决策规则与实验 1 接口

实验 0 不以某个散点“看起来分开”作为通过。进入实验 1 前需要：

1. `span-mean` 角度相对 token-angle/span-last 至少在 effect stability 或 reliability 上有一致优势；
2. 横向长度、角度或二者联合结构不只由单一 progress bin、layer、长度桶或问题驱动；
3. 跨 rollout `D_set-dir` 在 leave-one-out、balanced references 和 label permutation 下方向合理；
4. coordinate entropy / vertical norm 若有信号，控制 layer scale 与 horizontal norm 后仍保留；
5. 所有进入实验 1 的 representation、window/stride、layer 候选和 score 符号在 confirmatory 前冻结。

若 token-level 长度/纵向 entropy 有差异，但跨 rollout direction 完全为 null，则保留实验 0 的描述性结果，
不进入 success-trajectory 四格主线。若只有 span-pooled direction 稳定，则实验 1 聚焦 outcome-conditioned
direction；纵向 activity 留在实验 3 selector 路线，不提前与横向分数相乘。

---

## 6. 实验 1：Long Success-Trajectory 四格交互

### 6.1 目的

判断 long-response correct/wrong rollout 的 span-level 横向位移是否具有 outcome-specific 方向结构，
并区分该结构来自纯角度还是来自有实际幅度的运动。

### 6.2 Pilot 数据与最小 forward

从 primary discovery pool 固定选取 32-40 个 `3+3` 问题，按 correct/wrong 数量和四个 think-length bins
分层。复用已有 long rollout 文本，不重新生成，只对完整 think token 段做 hidden forward：

**硬约束：实验 1 禁止重新调用 vLLM 采样。** rollout 文本、rollout id、最终答案、correctness、
finish reason 和截断状态全部以现有 labeled JSONL 为准；模型仅以 `eval + inference_mode` 重放固定序列并
提取 hidden states，因此实验成本是 representation extraction，不是 generation。

```text
model            Qwen/Qwen3-VL-8B-Thinking
layers           24, 36
window / stride  128 / 64
saved vectors    span-last + span-mean，float16
saved metadata   question/rollout/label/span bounds/rho/think length/segment status
```

选择 L24/L36 是因为现有 long path 和 macro-angular 已在这两层形成可直接比较的 baseline。实验 1 不抽
37 层，也不永久保存每 token hidden。forward 中临时 token hidden 在完成 span pooling 后立即释放。

若 pilot 通过第 6.5 节 gate，再以完全冻结的提取和评分方式扩展到 primary discovery pool 其余问题；
随后将冻结分数应用到 `2+2` 补充覆盖题集，单独报告 reference 稀疏时是否仍保持方向。pilot
问题保留在 discovery，不能进入 confirmatory pool。

### 6.3 主分析

对 `layer 24/36 × last/mean`：

1. 构造 span displacement `d` 和 10-bin relative progress；
2. balanced LOO + 穷举等量 reference subset pairs，计算并平均 `S+`、`S-`；
3. 输出 angle/amplitude 的四格表；
4. 计算 `I_angle`、`I_amp` 及 question-bootstrap 95% CI；
5. 计算 `D_angle`、`D_amp` 的 within-question corr 和 pairwise AUC；
6. 记录每个 query/span 实际选中的 nearest reference rollout id；
7. 绘制 layer、progress、length-bin 曲线和 correct/wrong 分布。

后续主规格为实验 0 冻结的 `span-mean + K=1 + 10 progress bins`。`span-last`、`K=2/all`、5 bins 均为
预先声明的敏感性分析，不能事后取最大值作为主结果。2026-07-22 已完成运行仍按其原始
`span-last primary / span-mean sensitivity` 口径审计，不能追溯性改名。

### 6.4 必做控制

```text
C1  对称 reference：完整报告 A++ / A+- / A-+ / A-- 和 interaction。
C2  幅度：angle 与 query-amplitude 两版同时报告，不能只保留较好版本。
C3  进度：10-bin relative-progress matched 为主，5-bin 和无匹配为消融。
C4  数量：每题所有 query 固定相同 m_q；穷举 correct/wrong 等量 subset pairs。
C5  多路径：balanced LOO 为主；K=2/all、严格 2-fold 和 2+2 补充题集为 sensitivity。
C6  泄漏：query 必须按 id 剔除；naive in-sample 只用于量化 self-match 泄漏。
C7  置换：题内打乱标签，重新构造 reference sets 并重跑整套 LOO，得到 interaction null。
C8  Coverage：报告 nearest-reference 使用频率；移除最常被选中的 reference 后重算稳健性。
C9  长度：报告四个 think-length bins，并控制 think_length、n_spans、mean ||d||。
C10 截断/分段：主分析只用 clean complete-think；其他状态只作 sensitivity。
```

### 6.5 Pilot gate

pilot 不以单次 `p<0.05` 作为唯一标准。进入 discovery 扩展需要：

1. 对全部 balanced subsets 平均后的 `I_amp` 为正，并在多数 question-bootstrap resamples 中保持正号；
2. L24/L36 至少一层稳定，但效应不只来自单一 progress bin；
3. 题内 label permutation 后 interaction 回到 null；
4. 控制 think length、n_spans 和 mean displacement 后效应不归零；
5. amplitude 版至少比纯 angle 更稳定，或明确出现可解释的互补模式；
6. 效应不由单条正确 reference 独占，移除最常选 reference 后符号不翻转。

结果解释：

```text
I_angle > 0，I_amp > 0      方向选择性和有效运动量都成立。
I_angle ≈ 0，I_amp > 0      只有大幅度运动中的方向有意义；纯角度失败不否定 H2。
I_angle > 0，I_amp ≈ 0      存在方向结构，但尚未对应可用推进量。
二者均约为 0                当前 long span 表示下没有 success-trajectory 信号，停止主线。
```

---

## 7. 实验 2：Long Baseline 增量来源（不新增 forward）

### 7.1 目的与数据

实验 1 即使得到正 interaction，也可能只是重新表达 long path length、macro angular 或长度。实验 2
只回答“新方向选择性增加了什么”，不新增 hidden forward。

复用：

- 实验 1 冻结后的 `D_angle` / `D_amp`；
- 同题同 rollout 的现有 long path、macro/local angular parquet；
- 同一 long 数据上的 option-logit trajectory；
- 源 JSONL 中 think length、answer、truncation 和 correctness。

所有 baseline 必须限制在实验 1 的同一 eligible questions/rollouts 上重新评估。特征标准化、缺失值、
layer 选择和模型系数均只在 discovery training questions 中确定，外层按 `question_id` 分组交叉验证。

### 7.2 分层模型

```text
M0  think_length + n_spans + progress coverage                    nuisance baseline
M1  M0 + long d_mean + long path_length + macro-angular           现有 long hidden baseline
M2  M0 + D_angle                                                  纯方向选择性
M3  M0 + D_amp                                                    幅度门控方向
M4  M1 + D_angle                                                  方向是否超出现有 long geometry
M5  M1 + D_amp                                                    主增量检验
M6  M1 + long option-logit level                                  logit-controlled baseline
M7  M6 + D_amp                                                    与强 logit verifier 的条件增量
```

`M1` 使用预先固定、正则化的 baseline bundle；同时单独报告以 `d_mean`、`path_length`、macro-angular
为唯一 baseline 的版本，不能只挑让新特征增量最大的对照进入主表。

额外来源消融：

- `S+` 单边 correct-reference vs `D=S+-S-` 对称分数；
- global progress vs relative-progress matching；
- nearest-manifold `K=1` vs mean-direction `K=all`；
- span-last vs span-mean；
- `D_amp` vs 单独 `||d||` 与 `D_angle`，确认收益确实来自交互。

主要报告 outer-fold within-question pairwise AUC、within-question corr、`ΔAUC(M5-M1)`、
`ΔAUC(M7-M6)` 及 question-bootstrap 95% CI。long option-logit 约 0.789 只作参照，不能作为硬上界。

### 7.3 进入实验 3 的门槛

只有同时满足以下条件才做全层 forward：

1. discovery 扩展后的 `I_amp` 稳定为正；
2. 题内标签置换不能复现 interaction；
3. `D_amp` 相对 long path/macro baseline 有稳定正增量，而不只是 standalone AUC；
4. 控制 think length、n_spans 和 relative progress 后效应仍存在；
5. 信号不依赖某一个 balanced-reference subset、单一 layer 或单一 progress/length bin。

不要求 hidden 指标超过 option-logit。建议把稳定 `ΔAUC >= 0.01` 作为值得继续的实用阈值，但统计判断
仍以 bootstrap CI、跨 split 稳定性和局部 progress 结构为主。

---

## 8. 实验 3：Long Span 全层 Selector（需要新 forward）

### 8.1 Discovery calibration 与 confirmatory 条件

实验 3 分两段：

```text
3A  从 discovery pool 另取 16-24 题做全层 selector calibration；只调尺度、tau 和存储流程。
3B  冻结全部规格后，在 30% confirmatory pool 做一次 locked evaluation。
```

只有在 3B 前冻结以下内容，结果才能称为 confirmatory：

```text
候选 score、符号、layer set / 聚合、span size、stride、K、progress bins、
selector 标准化、tau、balanced LOO/subsampling、primary endpoint 和成功阈值。
```

如果 confirmatory pool 不足，或查看其新指标后继续调参，实验 3 必须降级为第二阶段 discovery。

### 8.2 Forward、span 与存储

继续使用第 1.1 节的同一 Thinking long-response setting和 clean complete-think 主子集，不重新生成
rollout。主设定：

```text
window / stride     128 / 64 think tokens
horizontal primary span-mean hidden
horizontal control span-last hidden
layers              全部 hidden-state indexes；相邻 block selector 使用 l=1..36
storage             保存 span 级压缩结果，不永久保存每 token hidden
```

`64/32` 和 `256/128` 只在 3A 做敏感性分析，3B 只运行冻结规格。semantic-step 边界作为 secondary，
不作为主切分，因为过短 step 的 SVD 不稳定且不同 rollout 样本量不可比。

为控制 long-response 存储，推荐按 question 处理 8 条 rollout：临时保留该题 span vectors，完成 balanced
LOO reference scoring 和 selector 统计后立即释放。永久文件至少保存：

```text
question/rollout/correctness/span bounds/rho/think length/segment status
冻结 layer set 的 span-last / span-mean（如需复核，float16）
每层 centered spectral entropy E、H、ER
相邻层 ΔE、|ΔE|、Δlayer_L2
balanced-LOO D_angle / D_amp、reference subset/id 与 selector-weighted scores
```

### 8.3 Centered spectral entropy

对 span `k`、layer `l`，把 span 内 token hidden 堆成：

\[
X_{i,k,l}
=
\begin{bmatrix}
h_{t_1,l}\\
\vdots\\
h_{t_n,l}
\end{bmatrix}
\in\mathbb R^{n\times d}.
\]

沿 token 维减均值：

\[
X^c_{i,k,l}=X_{i,k,l}-\mathbf 1\mu_{i,k,l}^{\top}.
\]

对 `Xc` 做 SVD，得到奇异值 `sigma_r`：

\[
p_r=\frac{\sigma_r}{\sum_j\sigma_j},
\qquad
H_{i,k,l}=-\sum_r p_r\log(p_r+\epsilon),
\]

\[
E_{i,k,l}=\frac{H_{i,k,l}}{\log R},
\qquad
ER_{i,k,l}=\exp(H_{i,k,l}).
\]

主规格固定 `R=min(n-1,d)`（token-centering 后的最大非零 rank）、`epsilon=1e-12`；数值上为零的
奇异值不贡献熵。3A 必须先用仓库既有 centered ER 函数对相同矩阵做一致性测试，再运行 3B。

`E` 是归一化谱熵，`ER` 用于与仓库既有 centered ER 实现核对。它们与“把单个 hidden vector 的
各坐标平方归一化后算熵”不同；后者依赖 hidden-coordinate basis，因此只在实验 0 作为同一模型内的
discovery diagnostic，不作为实验 3 的主 selector、跨模型复杂度或 rotation-invariant claim。

纵向 entropy selector 使用：

\[
\Delta E_{i,k,l}=E_{i,k,l}-E_{i,k,l-1},
\qquad
s^E_{i,k,l}=|\Delta E_{i,k,l}|.
\]

保留 signed `ΔE` 做机制解释，使用 `|ΔE|` 表示该层发生表示重组的强度。

### 8.4 相邻层 L2 selector

令 `g` 为同一 span 的 span-last 或 span-mean hidden：

\[
s^{L2}_{i,k,l}
=
\|g_{i,k,l}-g_{i,k,l-1}\|_2,
\qquad l=1,\ldots,36.
\]

不能用 `||h_L36-h_L24||` 冒充相邻层 selector。index 0 是 embedding output，不称为 transformer layer 0。

### 8.5 横纵组合与主对照

主分析沿用实验 1 冻结后的 `D_amp`，`D_angle` 作为对照。先用 3A training data 的
`layer × progress-bin` median/IQR 分别校准 `s_E` 和 `s_L2`，再只沿 layer 归一化。

定义：

\[
U_{i,k}=\frac{1}{|\mathcal L|}\sum_{l\in\mathcal L}D_{i,k,l,\rm amp},
\]

\[
W^E_{i,k}
=
\sum_{l\in\mathcal L}
\operatorname{softmax}_l(\widetilde s^E_{i,k,l}/\tau_E)
D_{i,k,l,\rm amp},
\]

\[
W^{L2}_{i,k}
=
\sum_{l\in\mathcal L}
\operatorname{softmax}_l(\widetilde s^{L2}_{i,k,l}/\tau_{L2})
D_{i,k,l,\rm amp}.
\]

纵向 selector 的核心检验不是它自身预测正确性的 AUC，而是：

\[
\Delta AUC_E=AUC(W_E)-AUC(U),
\qquad
\Delta AUC_{L2}=AUC(W_{L2})-AUC(U).
\]

同时报告 `WE` 与 `WL2` 的 layer-rank overlap。entropy 自身 AUC 很低但 `WE > U`，仍支持它是 selector；
若 `WE ≈ WL2 ≈ U`，则纵向选择没有增量。

### 8.6 后续局部有效性验证

只有 3B 主检验为正后，才对高/低 `D_amp` 或 selector span 做 continuation sampling，检验后续成功率。
span-Wasserstein、angular OT 和 activation patching 属于后续机制实验，不进入第一版主分析。

---

## 9. 统计口径、产出与停止规则

### 9.1 统一统计口径

- 所有主结果只使用第 1.1 节 long-response setting；
- 主单位是 question，不把 span 当独立样本计算 CI；
- 实验 0 主指标是 `layer × progress` 的 correct/wrong Hedges' `g`、题间符号一致率和 question-level bootstrap CI；
- 实验 1/2 主指标是 balanced-LOO within-question pairwise AUC 和 question-level bootstrap CI；
- 实验 0 同时报 scatter/hexbin、per-progress、per-layer 与 per-length effect；实验 1/2 另报 within-question
  Spearman 和预声明的四格 interaction；
- 模型比较使用完全相同的 eligible subset 和 outer folds；
- 全层探索只发生在实验 0 discovery 与 3A，3B 不选层、不调参；
- 同时报 effect size、CI、有效问题数、rollout/segment 排除流图，不只报 p-value。

### 9.2 每步产出

```text
LONG_EXPERIMENT_0_RESULTS.md              多分辨率 dynamics atlas、SNR、progress/layer/length 图
long_experiment_0_bin_features.parquet    question × rollout × bin × layer 标量
long_experiment_0_question_effects.csv    Hedges' g、bootstrap、permutation 与题间一致性
long_experiment_0_audit_spans*.npz        运行前固定少量 audit 题的 float16 span vectors

LONG_EXPERIMENT_1_RESULTS.md              四格表、interaction、progress/length 曲线
long_experiment_1_span_vectors*.npz       L24/L36 span-last/mean 压缩向量
long_experiment_1_features.parquet        balanced-LOO query 分数与 selected reference id
long_experiment_1_interactions.csv        question × layer × progress interaction

LONG_EXPERIMENT_2_RESULTS.md              long baseline nested model 与增量 AUC
long_experiment_2_incremental_auc.csv     outer-fold / bootstrap 结果

LONG_EXPERIMENT_3_RESULTS.md              U / WL2 / WE locked evaluation
long_experiment_3_span_features.parquet   全层 span 级压缩统计
```

### 9.3 停止规则

```text
实验 0 span-pooled angle 无 SNR/reliability 增益      不进入四格方向主线，保留 amplitude/entropy 负或描述性结果。
实验 0 仅 activity 有差异、跨 rollout direction 为 null 不进入实验 1；纵向 activity 仅保留为 selector 候选。
实验 1 pilot 无稳定 I_angle/I_amp              不扩展，记录 long-response 负结果。
实验 1 扩展有 interaction、实验 2 无条件增量    不做全层；说明它只是 long path/length 的重表达。
实验 2 有稳定增量                              冻结设计，进入实验 3A/3B。
实验 3B WE/WL2 均不优于 U                      不把纵向 selector 引入 RL。
实验 3B 仅 WL2 有效                            保留 StALT 式 L2，放弃 entropy novelty claim。
实验 3B WE 稳定优于 U 与 WL2                   才进入 long-response span-level RL shaping。
```

---

## 10. 本阶段允许与不允许的结论

实验 0 最多支持：

```text
在固定 long-response setting 中，正确/错误 rollout 的 token-amplitude 与 span-direction dynamics
是否在特定 layer、relative progress 和 length strata 上存在可重复差异，以及 span pooling 是否提高
角度类量的稳定性；它不证明这些 activity 或方向差异已经是局部因果 credit。
```

实验 1/2 最多支持：

```text
在 Qwen3-VL-8B-Thinking 的 MathVerse long-response setting 中，
progress-matched span displacement 相对 success/failure reference manifold 存在选择性，
且该选择性是否在 long path、length 和 option-logit 之外提供增量。
```

实验 3 最多进一步支持：

```text
long-response span 内谱结构的层间变化可以选择更有用的横向 hidden credit 信号。
```

在 continuation 或干预实验之前，不能声称指标已经定位了因果 credit；在真正接入并改善 long-response
RL 训练之前，也不能声称它已经解决了 credit assignment。当前目标是得到一个定义清楚、无明显泄漏、
值得进入 RL 阶段验证的 candidate signal。
