# 2Dimension：横向轨迹 × 纵向层选择的 hidden-state credit 信号实验

## 0. 研究目标与范围

本文档规划一组由便宜到昂贵的发现性实验，目标是回答：

```text
能否从 rollout 的 hidden-state 轨迹中构造一个局部 credit 候选信号：
横向使用随生成进度发生的表示位移，纵向使用层间表示重组来选择重要层，
并最终服务于 GRPO / RLVR 的 span-level reward shaping？
```

当前阶段只做离线发现与验证，不修改 RL 训练。训练期允许使用 ground-truth 和同题 correct/wrong
rollout 构造 privileged reference，但任何候选分数在给 query rollout 打分时都不能读取 query 自身标签。

执行顺序固定为：

```text
实验 1：复用现有 h_chunk，验证 success-trajectory 的角度与幅度加权角度。
实验 2：复用现有结果，判断新方向信号相对 path / ERV 到底增加了什么。
实验 3：仅在 1+2 通过后重新 forward，验证 token/span 级纵向 selector。
```

---

## 1. 已确认的数据条件

### 1.1 H200 缓存仍然存在

2026-07-21 已在 H200 上只读确认：

```text
/data2/hjk/projects/AI-HiddenState-ER/chunklayer/h_gpu{0..7}.npz
```

8 个文件总计约 9.0 GB，均包含 `h_chunk.npy`。抽查 `h_gpu0.npz`：

```text
h_chunk.shape = (4318, 37, 4096)
h_chunk.dtype = float16
```

每个 shard 还保存 `rollout_idx`、`chunk_start`、`chunk_end`、`meta_is_correct`、
`meta_response_len` 等索引和标签。因此实验 1、实验 2 不需要重新 forward。

### 1.2 缓存能做什么、不能做什么

当前 `h_chunk[i,k,l]` 是第 `i` 条 rollout、第 `k` 个 256-token chunk、模型第 `l` 层的
token-mean hidden state。

`h_chunk` 的 37 个 index 对应 embedding output 加 36 个 transformer block output；涉及相邻 block
的纵向分析使用 `l=1..36` 与 `l-1` 配对，避免把 index 0 误称为 transformer layer 0。

它可以计算：

- 37 层上的 chunk-to-chunk 横向位移；
- 相邻层 chunk-mean 的纵向 L2 变化；
- correct/wrong rollout 的 progress-matched 方向关系；
- 与现有 path length、late displacement、ERV/ERA 的增量关系。

它不能恢复：

- chunk 内任意单个 token hidden；
- semantic-step last-token hidden；
- span 内 token × hidden 分布；
- centered spectral entropy 或 span-Wasserstein。

因此实验 3 重新 forward 的原因是需要 token/span 内部结构，不是缺少多层 hidden。

---

## 2. 既有结果给出的约束

以下是设计先验，不把它们误写成普适结论：

```text
[C1] 现有 long-CoT 横向几何指标大多只有 AUROC 0.50-0.58，长度与进度混杂明显。
[C2] prompt 加/不加答案选项构造的方向在 think 段约为 0.52-0.53，不能直接复用。
[C3] 既有结果中运动幅度通常强于纯 cosine；方向可能只在足够大的位移上有意义。
[C4] layer 变化更适合作为横向信号的 selector，而不是单独作为“纵向语义”。
```

现有强 hidden baseline：

```text
L36 d_late_mean                AUROC 0.7211
L36 path_length                AUROC 0.7109
L36 ERV                        AUROC 约 0.69
path + ERV                     AUROC 0.7153（只比 path 小幅提高）
```

现有 option-logit `trimmed_margin_mean` 的 within-question corr 约 0.54、pairwise AUC 约 0.80。
它是外部强 baseline，不是 hidden 指标必须超过的停止阈值。新 hidden 指标的目标是提供更早、更局部、
且在 path/ERV/logit level 之外仍有增量的信息。

---

## 3. Novelty 边界与研究假设

直接使用 answer likelihood / answer logits 做 token-level credit 已经较拥挤：

- GeneralThinker（arXiv:2605.27934）：使用 ground-truth answer likelihood 派生 token-wise 信号；
- OPPO（arXiv:2605.21851）：使用 oracle-conditioned likelihood ratio 和递归 success value；
- Where Hindsight Credit Can Reside（arXiv:2604.11056）：研究 RLVR 中 hindsight credit 的 signed capacity。

因此本计划不把 answer logit 本身作为方法，只将它用于 baseline 和增量控制。

待验证的假设为：

```text
[H1] 横向 hidden 位移包含 rollout 是否沿某条成功轨迹推进的信号。
[H2] 纯角度可能很弱，但“位移幅度 × 成功轨迹选择性”可能有增量。
[H3] 成功轨迹不是唯一均值方向；相对 correct/wrong reference manifold 的选择性更稳妥。
[H4] token/span 内 centered spectral entropy 的层间变化可以选择应信任的层。
```

这里的“横向”应严格称为 token-progress dynamics。它可能包含语义、token identity、上下文积累和
hidden norm 变化，不能在实验前直接等同于语义。

---

## 4. 共用符号与防泄漏协议

### 4.1 横向位移与相对进度

对问题 `q` 的 rollout `i`、chunk `k`、layer `l`：

\[
h_{i,k,l}\in\mathbb R^d,
\qquad
d_{i,k,l}=h_{i,k,l}-h_{i,k-1,l}.
\]

用目标 chunk 的中心定义相对进度：

\[
\rho_{i,k}
=
\frac{\operatorname{start}_{i,k}+\operatorname{end}_{i,k}}
{2L_i},
\]

其中 `L_i` 是该 rollout 的 response token 数。主分析把 `rho` 分成 5 个等宽区间；无进度匹配版本
仅作为消融。先在 `rollout × progress-bin` 内聚合，再让 progress bin 等权，避免长回答因 chunk 更多
而自动获得更大权重。

### 4.2 Discovery/confirmatory 问题划分

在查看新方向指标之前，按 `question_id` 固定划分：

```text
discovery pool      70% eligible questions，只供实验 1/2 选层、选特征和调参
confirmatory pool   30% eligible questions，在实验 3 规格冻结前不计算新指标
```

划分按每题 correct/wrong rollout 数量分层，并固定随机种子。既有 path/ERV/logit 报告已经看过全量数据，
因此这里的 held-out 含义是“未用于新 2Dimension 指标的选择”，不是声称这些题从未被任何旧实验分析。

若 eligible questions 太少，导致 30% confirmatory pool 无法给出有效 CI，则改用全量 nested grouped CV，
但实验 3 必须称为第二阶段 discovery，不能称为 confirmatory。

### 4.3 Reference/query cross-fitting

只保留同题至少有 2 条 correct 和 2 条 wrong rollout 的问题。每题进行 repeated stratified 2-fold
cross-fitting：一半 rollout 构造 reference，另一半只作为 query，然后交换；主结果跨 20 个随机划分取均值。

每次划分中：

- correct/wrong reference 数量取两组的最小值，随机等量抽样；
- query 不进入任何 reference 集，`++` 和 `--` 也不存在 self-match；
- reference 只能与 query 的同一 relative-progress bin 匹配；
- query 的正确性标签只用于事后分组评估，不能参与 query score 的计算；
- layer、符号方向、超参数和特征组合必须只在训练问题中选择。

这比单纯 leave-one-out 更严格。leave-one-out 只作为泄漏幅度消融，不作为正式主结果。

### 4.4 四格角度与对称交互

令 `a` 表示 query 标签，`b` 表示 reference 标签，`+` 为 correct，`-` 为 wrong。对每个 query
位移，先从每条 reference rollout 的同 progress bin 中选与 query `rho` 最近的一条位移，保证每条
reference rollout 最多贡献一个候选；再从标签为 `b` 的等量 reference 候选中取 top-K cosine 的均值：

\[
S_i^b
=
\operatorname{meanTopK}_{j\in R_b}
\cos(d_i,d_j).
\]

主分析固定 `K=1`，即 nearest-manifold；`K=all` 作为均值方向消融。由此定义：

\[
A_{\rm angle}^{ab}
=
\mathbb E[S_i^b\mid y_i=a].
\]

必须同时报告四格：

\[
A^{++},\quad A^{+-},\quad A^{-+},\quad A^{--}.
\]

query 对 correct/wrong manifold 的选择性分数为：

\[
D_{i,\rm angle}=S_i^+-S_i^-.
\]

核心对称交互为：

\[
I_{\rm angle}
=(A^{++}-A^{+-})-(A^{-+}-A^{--}).
\]

只检验 `A^{++} > A^{-+}` 会产生“正确组本来就更紧”的自证风险；四格交互要求 correct query
相对偏向 correct manifold 的程度，显著强于 wrong query 的这种偏向。

### 4.5 幅度加权版

纯 cosine 在 `||d_i||` 很小时不稳定，因此实验 1 必须同时计算 query-amplitude 加权版：

\[
P_i^b=\|d_i\|_2 S_i^b,
\qquad
D_{i,\rm amp}=P_i^+-P_i^-
=\|d_i\|_2D_{i,\rm angle}.
\]

对应四格与交互：

\[
A_{\rm amp}^{ab}
=
\mathbb E[P_i^b\mid y_i=a],
\]

\[
I_{\rm amp}
=(A_{\rm amp}^{++}-A_{\rm amp}^{+-})
-(A_{\rm amp}^{-+}-A_{\rm amp}^{--}).
\]

第一版不乘 reference 的 `||d_j||`，避免高幅度 reference 主导 nearest-neighbor。每层单独报告 raw
幅度版；跨层组合时再用同题、同 progress bin 的 median norm 做尺度归一化。

---

## 5. 实验 1：success-trajectory 四格交互（零 forward）

### 5.1 目的

判断 correct/wrong rollout 的横向位移是否具有 outcome-specific 方向结构，并区分该结构来自纯角度
还是来自有实际幅度的运动。

### 5.2 数据与主分析

直接读取 H200 的 8 个 `h_chunk` shard，先输出每题 correct/wrong 数量和可进入 cross-fitting 的问题数。
只有一个 chunk 的 rollout 没有横向位移，不做零值填充；必须报告其数量、正确率和被排除前后的长度分布。

对每个 hidden-state index `0..36`：

1. 构造相邻 chunk 位移 `d` 和 relative-progress bin；
2. repeated cross-fitting 计算 `S_i+`、`S_i-`；
3. 输出 angle 与 amplitude 的四格表；
4. 计算 `I_angle`、`I_amp` 及 question-bootstrap 95% CI；
5. 计算 query-level `D_angle`、`D_amp` 的 within-question corr 和 pairwise AUC；
6. 绘制 layer curve、progress curve，以及 correct/wrong 的 `D` 分布。

预注册主层为 24、36；37 层全扫描用于 discovery。若要从全层中挑选最佳层用于实验 2，必须在
training questions 中选择，再到 held-out questions 评估，不能在同一批问题上选层并报告性能。

### 5.3 必做控制

```text
C1  对称 reference：完整报告 A++ / A+- / A-+ / A-- 和 interaction。
C2  幅度：angle 与 query-amplitude 两版同时报告，不能只保留表现较好的一版。
C3  进度：relative-progress matched 为主，无匹配版本为消融。
C4  数量：correct/wrong reference 严格等量；固定 K，避免 max 随集合大小机械增大。
C5  泄漏：cross-fit 为主，naive in-sample 与 leave-one-out 只用于量化泄漏。
C6  置换：题内打乱 correct/wrong 标签，整套 cross-fit 重跑，构造 interaction null。
C7  混杂：报告 response length、n_chunks、||d|| 与 D_angle/D_amp 的题内相关。
```

### 5.4 结果解释

```text
I_angle > 0，I_amp > 0      方向选择性和有效运动量都成立。
I_angle ≈ 0，I_amp > 0      只有大幅度运动中的方向有意义；纯角度失败不否定 H2。
I_angle > 0，I_amp ≈ 0      有方向结构，但尚未对应可用的推进量。
二者均约为 0                当前 256-token chunk 粒度下没有 success-trajectory 信号。
```

不能把 `A++` 单独较高解释为成功方向，也不能把全局 pooled AUROC 当主结果。

---

## 6. 实验 2：增量来源与可替代性（零 forward）

### 6.1 目的

实验 1 即使得到正 interaction，也可能只是重新表达已有的 path length、late displacement 或 ERV；
纵向层加权也可能没有任何作用。实验 2 同时回答“方向选择性增加了什么”和“chunk-mean 纵向 L2
能否选择更有用的层”，不引入新 hidden forward。

### 6.2 公平比较集合

所有 baseline 与新指标必须限制在实验 1 的同一组 eligible questions / rollouts 上重新评估，不能把
不同样本上的 AUROC 直接相减。特征标准化、缺失值处理、层选择和模型系数均只在训练问题中完成，
外层按 `question_id` 分组交叉验证。

### 6.3 横向增量的分层消融

```text
M0  length + n_chunks + progress coverage                         nuisance baseline
M1  M0 + d_late_mean                                              主幅度 baseline
M2  M0 + D_angle                                                 纯方向选择性
M3  M0 + D_amp                                                   幅度门控方向
M4  M1 + D_angle                                                 方向是否超出已有幅度
M5  M1 + D_amp                                                   主增量检验
M6  M1 + ERV + D_amp                                             是否超出 path + ERV
M7  M6 + final option-logit level                                与强 logit verifier 的条件增量，仅作参照
```

`path_length` 替换 `d_late_mean`、以及包含两者的正则化 path bundle 作为 secondary analysis；不能只挑
其中增量最大的 baseline 进入主表。

额外做三个来源消融：

- `S+` 单边 correct-reference 分数 vs `D=S+-S-` 对称分数；
- global progress matching vs relative-progress matching；
- nearest-manifold `K=1` vs mean-direction `K=all`。

主要报告：outer-fold within-question pairwise AUC、within-question corr、`ΔAUC(M5-M1)`、
`ΔAUC(M6-(M1+ERV))`，以及 question-bootstrap 95% CI。option-logit 0.80 只作参照，不能作为硬上界。

### 6.4 便宜的纵向 L2 selector 检验

现有 `h_chunk` 已覆盖 37 个 hidden-state index，因此可以先在 chunk-mean 粒度计算：

\[
s^{L2}_{i,k,l}=\|h_{i,k,l}-h_{i,k,l-1}\|_2,
\qquad l=1,\ldots,36.
\]

对训练问题中每个 `layer × progress-bin` 的 `s_L2` 做 median/IQR 标准化，再仅沿 layer 维归一化成
权重。用实验 1 的 `D_amp` 构造：

\[
U_{i,k}=\frac{1}{|\mathcal L|}\sum_{l\in\mathcal L}D_{i,k,l,\rm amp},
\]

\[
W^{L2}_{i,k}
=
\sum_{l\in\mathcal L}
\operatorname{softmax}_l(\widetilde s^{L2}_{i,k,l}/\tau)
D_{i,k,l,\rm amp}.
\]

比较 `W_L2` 与 `U` 的 held-out `ΔAUC`。这是 chunk-mean 层选择的可行性检查，不冒充 StALT 的
token-level 结果。`tau` 只能在 discovery training folds 中选择。

结果用于定位来源：

```text
D_amp 无增量                         横向主线停止。
D_amp 有增量，W_L2 ≈ U               增益来自横向方向；纵向 L2 暂无帮助。
D_amp 有增量，W_L2 > U               横向与纵向选择均有贡献，最支持进入实验 3。
W_L2 > U，但 D_amp standalone 很弱    检查加权是否只重现 layer norm / path 幅度。
```

### 6.5 进入实验 3 的门槛

只有同时满足以下条件才重新 forward：

1. `I_amp` 的方向在预注册层或训练选层后的 held-out questions 上稳定为正；
2. 题内标签置换不能复现该 interaction；
3. 加入 `D_amp` 后相对 path 或 path+ERV 有稳定正增量，而不只是 standalone AUC；
4. 控制 response length、n_chunks 和 relative progress 后效应仍存在；
5. 信号不依赖某一个随机 reference split 或单个 progress bin。

不要求 hidden 指标超过 option-logit pairwise AUC 0.80。建议把稳定 `ΔAUC >= 0.01` 作为值得继续的
实用阈值，但统计判断仍以 bootstrap CI、跨 split 稳定性和局部 progress 结构为主。

---

## 7. 实验 3：token/span 多层 selector（需要新 forward）

### 7.1 Confirmatory 条件

实验 3 只有在开始前冻结以下内容，并使用未参与实验 1/2 选层、选特征的 held-out questions，才能称为
confirmatory：

```text
候选 score、符号方向、主 layer / layer 聚合方式、span size、stride、K、progress bins、
cross-fit 方案、primary endpoint 和成功阈值。
```

如果继续使用同一批问题调整这些选择，应明确称为第二阶段 discovery。

### 7.2 Span 与存储

主设定：

```text
window = 128 response tokens
stride = 64 tokens
representation = span-mean hidden
layers = 全层在线统计；永久保存 span 级压缩结果，不保存每个 token hidden
```

`window/stride = 64/32` 和 `256/128` 只作敏感性分析。semantic-step 边界作为 secondary analysis，
不作为主切分，因为过短 step 的 SVD 不稳定且不同 rollout 的样本量不可比。

每个 span 最少保存：

```text
question_id / rollout_id / correctness / span_start / span_end / relative_progress
span_mean 或 span_last（fp16，供横向方向计算）
每层 centered spectral entropy E
相邻层 ΔE、|ΔE| 和 chunk/span-mean Δlayer_L2
```

### 7.3 Centered spectral entropy

对 span `k`、layer `l`，把该 span 的 token hidden 堆成：

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

先沿 token 维减均值：

\[
X^c_{i,k,l}=X_{i,k,l}-\mathbf 1\mu_{i,k,l}^{\top}.
\]

对 `Xc` 做 SVD，得到奇异值 `sigma_r`：

\[
p_r=\frac{\sigma_r}{\sum_j\sigma_j},
\qquad
E_{i,k,l}
=
-\frac{\sum_r p_r\log(p_r+\epsilon)}{\log R}.
\]

这里的 `E` 是归一化谱熵。还应保存未归一化熵 `H=-sum(p log p)` 和有效秩 `ER=exp(H)`，以便与仓库
已有 centered ER 实现核对。`E` 衡量 span 内 token 表示占据多少个有效变化方向。它与“把单个 hidden vector 的各坐标平方
归一化后算熵”不同；后者依赖任意 hidden coordinate basis，不作为本计划指标。

纵向 selector 使用：

\[
\Delta E_{i,k,l}=E_{i,k,l}-E_{i,k,l-1},
\qquad
s^E_{i,k,l}=|\Delta E_{i,k,l}|.
\]

保留 signed `ΔE` 做机制解释，使用 `|ΔE|` 表示该层发生表示重组的强度。相邻层 L2 selector 为：

\[
s^{L2}_{i,k,l}
=
\|g_{i,k,l}-g_{i,k,l-1}\|_2,
\]

其中 `g` 是 span-mean hidden。不能用 `||h_L36-h_L24||` 冒充相邻层 selector。

### 7.4 横纵组合与主对照

沿用实验 1 冻结后的 `D_angle` / `D_amp`。主分析使用 `D_amp`，`D_angle` 作为预先保留的对照，分别构造：

```text
U   不加纵向 selector 的横向分数
WL2 用相邻层 L2 对各层横向分数加权
WE  用 |ΔE| 对各层横向分数加权
```

先使用 discovery/training split 的 `layer × progress-bin` median/IQR 对 `s_E` 和 `s_L2` 分别校准，
再只在 layer 维归一化。例如：

\[
w_{i,k,l}^{E}
=
\operatorname{softmax}_l(s^E_{i,k,l}/\tau).
\]

`tau` 必须在 discovery/training split 中确定并冻结。纵向 selector 的核心检验不是它自身能否预测正确性，
而是：

\[
\Delta AUC_E=AUC(W_E)-AUC(U),
\qquad
\Delta AUC_{L2}=AUC(W_{L2})-AUC(U).
\]

同时报告 `WE` 与 `WL2` 选中层的 rank overlap。若 entropy 自身 AUC 很低但 `WE > U`，仍然支持它是
selector；若 `WE ≈ WL2 ≈ U`，则纵向选择没有增量。

### 7.5 可选的局部有效性验证

只有主检验为正后，才对高/低 `D_amp` 或高/低 selector span 做 continuation sampling，检验后续成功率。
这比仅用最终 correctness 做相关性更接近局部 credit assignment。span-Wasserstein、angular OT 或
activation patching 均属于后续机制实验，不进入第一版主分析。

---

## 8. 统计口径、产出与停止规则

### 8.1 统一统计口径

- 主单位是 question，不把 chunk/span 当独立样本计算置信区间；
- 主指标是 cross-fitted within-question pairwise AUC 和 question-level bootstrap CI；
- 同时报 within-question Spearman、四格 interaction、per-progress effect；
- 全层探索必须做 held-out 选层或多重比较校正；
- 所有模型比较使用完全相同的 eligible subset 和 outer folds；
- 同时报 effect size、CI 和有效问题数，不只报 p-value。

### 8.2 每步产出

```text
EXPERIMENT_1_RESULTS.md              四格表、interaction、layer/progress 曲线
experiment_1_features.parquet       cross-fitted query 分数
experiment_1_interactions.csv       question × layer × progress 的 interaction

EXPERIMENT_2_RESULTS.md              nested model 与增量 AUC
experiment_2_incremental_auc.csv    outer-fold / bootstrap 结果

EXPERIMENT_3_RESULTS.md              U / WL2 / WE 与 confirmatory endpoint
experiment_3_span_features.parquet  span 级压缩统计
```

### 8.3 停止规则

```text
实验 1 无稳定 I_angle/I_amp               停止 success-direction 主线，记录负结果。
实验 1 有 interaction、实验 2 无条件增量   不做实验 3；说明它只是 path/ERV 的重表达。
实验 2 有稳定增量                          冻结设计与 held-out set，进入实验 3。
实验 3 WE/WL2 均不优于 U                   不把纵向 selector 引入 RL。
实验 3 仅 WL2 有效                         保留 StALT 式 L2，放弃 entropy novelty claim。
实验 3 WE 稳定优于 U 与 WL2                才进入 span-level RL shaping 设计。
```

---

## 9. 本阶段允许与不允许的结论

实验 1/2 最多支持：

```text
progress-matched hidden displacement 相对 success/failure reference manifold 存在选择性，
且该选择性是否在 path/ERV 之外提供增量。
```

实验 3 最多进一步支持：

```text
span 内谱结构的层间变化可以选择更有用的横向 hidden credit 信号。
```

在 continuation 或干预实验之前，不能声称指标已经定位了因果 credit；在真正接入并改善 RL 训练之前，
也不能声称它已经解决了 credit assignment。当前目标是得到一个定义清楚、无明显泄漏、值得进入 RL
阶段验证的 candidate signal。
