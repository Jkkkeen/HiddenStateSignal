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

因此它不能支持本项目的 long-response 主张。旧缓存只用于回顾短/长 setting 差异，不用于实验 1 的
四格方向、实验 2 的增量模型或实验 3 的 selector。

### 1.4 Long 数据目前缺少什么

现有 long parquet 保存了 path、macro angular、local angular 等标量，没有保存 span-mean、span-last 或
任意可重算的 hidden vector，也没有全层 token/span 谱统计。因此：

- long rollout 不需要重新生成；
- 实验 1 必须做一次最小 hidden forward；
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

这些 baseline 必须在实验 1 的同一 eligible subset 上重新评估后才能比较。option-logit 是外部强参照，
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
primary representation   span endpoint last-token hidden
secondary representation span-mean hidden
```

主分析只保留完整的 128-token span；末尾不足 128 token 的 partial span 不做 padding，也不进入主分数，
但必须报告每条 rollout 被覆盖的 think-token 比例。这样保证不同 span 的 SVD rank 和 mean 方差可比。

记 span 表示为：

\[
g_{i,k,l}\in\mathbb R^d,
\qquad
d_{i,k,l}=g_{i,k,l}-g_{i,k-1,l}.
\]

last-token 表示读完整个 span 后的 prefix state；span-mean 用于检验结果是否依赖单点状态。主实验不能
在看到结果后交换二者的 primary/secondary 地位。

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
discovery pool      70%，供实验 1/2 pilot、扩展、选层和特征消融
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

## 5. 实验 1：Long Success-Trajectory 四格交互

### 5.1 目的

判断 long-response correct/wrong rollout 的 span-level 横向位移是否具有 outcome-specific 方向结构，
并区分该结构来自纯角度还是来自有实际幅度的运动。

### 5.2 Pilot 数据与最小 forward

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

若 pilot 通过第 5.5 节 gate，再以完全冻结的提取和评分方式扩展到 primary discovery pool 其余问题；
随后将冻结分数应用到 `2+2` 补充覆盖题集，单独报告 reference 稀疏时是否仍保持方向。pilot
问题保留在 discovery，不能进入 confirmatory pool。

### 5.3 主分析

对 `layer 24/36 × last/mean`：

1. 构造 span displacement `d` 和 10-bin relative progress；
2. balanced LOO + 穷举等量 reference subset pairs，计算并平均 `S+`、`S-`；
3. 输出 angle/amplitude 的四格表；
4. 计算 `I_angle`、`I_amp` 及 question-bootstrap 95% CI；
5. 计算 `D_angle`、`D_amp` 的 within-question corr 和 pairwise AUC；
6. 记录每个 query/span 实际选中的 nearest reference rollout id；
7. 绘制 layer、progress、length-bin 曲线和 correct/wrong 分布。

主规格为 `span-last + K=1 + 10 progress bins`。`span-mean`、`K=2/all`、5 bins 均为预先声明的敏感性
分析，不能事后取最大值作为主结果。

### 5.4 必做控制

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

### 5.5 Pilot gate

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

## 6. 实验 2：Long Baseline 增量来源（不新增 forward）

### 6.1 目的与数据

实验 1 即使得到正 interaction，也可能只是重新表达 long path length、macro angular 或长度。实验 2
只回答“新方向选择性增加了什么”，不新增 hidden forward。

复用：

- 实验 1 冻结后的 `D_angle` / `D_amp`；
- 同题同 rollout 的现有 long path、macro/local angular parquet；
- 同一 long 数据上的 option-logit trajectory；
- 源 JSONL 中 think length、answer、truncation 和 correctness。

所有 baseline 必须限制在实验 1 的同一 eligible questions/rollouts 上重新评估。特征标准化、缺失值、
layer 选择和模型系数均只在 discovery training questions 中确定，外层按 `question_id` 分组交叉验证。

### 6.2 分层模型

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

### 6.3 进入实验 3 的门槛

只有同时满足以下条件才做全层 forward：

1. discovery 扩展后的 `I_amp` 稳定为正；
2. 题内标签置换不能复现 interaction；
3. `D_amp` 相对 long path/macro baseline 有稳定正增量，而不只是 standalone AUC；
4. 控制 think length、n_spans 和 relative progress 后效应仍存在；
5. 信号不依赖某一个 balanced-reference subset、单一 layer 或单一 progress/length bin。

不要求 hidden 指标超过 option-logit。建议把稳定 `ΔAUC >= 0.01` 作为值得继续的实用阈值，但统计判断
仍以 bootstrap CI、跨 split 稳定性和局部 progress 结构为主。

---

## 7. 实验 3：Long Span 全层 Selector（需要新 forward）

### 7.1 Discovery calibration 与 confirmatory 条件

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

### 7.2 Forward、span 与存储

继续使用第 1.1 节的同一 Thinking long-response setting和 clean complete-think 主子集，不重新生成
rollout。主设定：

```text
window / stride     128 / 64 think tokens
horizontal primary span-last hidden
horizontal control span-mean hidden
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

### 7.3 Centered spectral entropy

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
各坐标平方归一化后算熵”不同；后者依赖任意 hidden coordinate basis，不作为本计划指标。

纵向 entropy selector 使用：

\[
\Delta E_{i,k,l}=E_{i,k,l}-E_{i,k,l-1},
\qquad
s^E_{i,k,l}=|\Delta E_{i,k,l}|.
\]

保留 signed `ΔE` 做机制解释，使用 `|ΔE|` 表示该层发生表示重组的强度。

### 7.4 相邻层 L2 selector

令 `g` 为同一 span 的 span-last 或 span-mean hidden：

\[
s^{L2}_{i,k,l}
=
\|g_{i,k,l}-g_{i,k,l-1}\|_2,
\qquad l=1,\ldots,36.
\]

不能用 `||h_L36-h_L24||` 冒充相邻层 selector。index 0 是 embedding output，不称为 transformer layer 0。

### 7.5 横纵组合与主对照

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

### 7.6 后续局部有效性验证

只有 3B 主检验为正后，才对高/低 `D_amp` 或 selector span 做 continuation sampling，检验后续成功率。
span-Wasserstein、angular OT 和 activation patching 属于后续机制实验，不进入第一版主分析。

---

## 8. 统计口径、产出与停止规则

### 8.1 统一统计口径

- 所有主结果只使用第 1.1 节 long-response setting；
- 主单位是 question，不把 span 当独立样本计算 CI；
- 主指标是 balanced-LOO within-question pairwise AUC 和 question-level bootstrap CI；
- 同时报 within-question Spearman、四格 interaction、per-progress 与 per-length effect；
- 模型比较使用完全相同的 eligible subset 和 outer folds；
- 全层探索只发生在 3A，3B 不选层、不调参；
- 同时报 effect size、CI、有效问题数、rollout/segment 排除流图，不只报 p-value。

### 8.2 每步产出

```text
LONG_EXPERIMENT_1_RESULTS.md              四格表、interaction、progress/length 曲线
long_experiment_1_span_vectors*.npz       L24/L36 span-last/mean 压缩向量
long_experiment_1_features.parquet        balanced-LOO query 分数与 selected reference id
long_experiment_1_interactions.csv        question × layer × progress interaction

LONG_EXPERIMENT_2_RESULTS.md              long baseline nested model 与增量 AUC
long_experiment_2_incremental_auc.csv     outer-fold / bootstrap 结果

LONG_EXPERIMENT_3_RESULTS.md              U / WL2 / WE locked evaluation
long_experiment_3_span_features.parquet   全层 span 级压缩统计
```

### 8.3 停止规则

```text
实验 1 pilot 无稳定 I_angle/I_amp              不扩展，记录 long-response 负结果。
实验 1 扩展有 interaction、实验 2 无条件增量    不做全层；说明它只是 long path/length 的重表达。
实验 2 有稳定增量                              冻结设计，进入实验 3A/3B。
实验 3B WE/WL2 均不优于 U                      不把纵向 selector 引入 RL。
实验 3B 仅 WL2 有效                            保留 StALT 式 L2，放弃 entropy novelty claim。
实验 3B WE 稳定优于 U 与 WL2                   才进入 long-response span-level RL shaping。
```

---

## 9. 本阶段允许与不允许的结论

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
