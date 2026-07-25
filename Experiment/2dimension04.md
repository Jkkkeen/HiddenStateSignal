# 2Dimension-04: Fixed-256 Chunk 横向/纵向 Hidden Dynamics

## 0. 背景与目的

前序实验给出三个直接动机：

- `raw activation entropy` 的 `L14-L19 / bin6` 候选在 120 题独立验证中未复现：within-Q AUC `0.5212`，95% CI 覆盖 `0.5`，加入长度后的 AUC 增量为 `-0.0047`。发现集效应大幅缩水，与 winner's curse 一致，因此旧 entropy band 不进入 RL。
- 旧 entropy 候选先逐 token 算熵，再把一个相对 progress bin 内约 1000-1700 个 token 的熵取均值。若信号只在短暂 reasoning burst 中出现，这种聚合可能造成稀释。
- 早期 short-CoT Instruct 实验使用 256-token chunk-mean hidden，L36 的 `d_late_mean` AUROC 约 `0.72`，path length AUROC 约 `0.71`；但没有严格 length gate，也不能直接外推到 long-CoT Thinking。

本实验在 long-response Qwen3-VL-8B-Thinking rollout 上重新使用固定 256-token chunk，同时扫描：

1. 横向：同一层沿 chunk 发展的轨迹；
2. 纵向：同一 chunk 沿 layer 发展的轨迹；
3. 三种 chunk hidden 表示；
4. 几何量、raw entropy 与 difference entropy。

本轮只做 discovery + held-out confirmation，不做 RL 训练。纵向层数固定，因此不受“随 chunk 数机械累加”的 count-scaling 影响，但其数值仍可能关联 response length、难度和相对进程，必须通过与横向相同的 length gate。

## 1. 数据与切分

```text
model                 Qwen/Qwen3-VL-8B-Thinking
dataset               MathVerse testmini, answer in A/B/C/D
rollouts/question     8
temperature           0.7
top_p                 0.95
max_tokens            16384
analysis segment      response start 至完整 </think> 前
chunk size            256 response tokens
hidden-state index    0..36（0=embedding 输出，1..36=Transformer block 输出）
question eligibility  >=2 correct 且 >=2 wrong
split unit            question
split seed            20260726
relative-progress bins 20 (secondary visualization only)
```

- 复用已经生成的 long-response rollout；生成过数据不等于看过本实验的新指标。
- 在计算、查看任何 2Dimension-04 hidden 指标前，对全部合格题按固定 seed 排序并冻结 discovery/confirm 切分。前一半进入 discovery，后一半进入 confirm，question 完全不重叠。
- confirm 的含义是：这些题在冻结前未用于查看或选择任何 2Dimension-04 chunk 指标。它们可以曾用于旧 entropy 等不同 feature family，但重叠情况必须写入 manifest，不得用旧结果调整 04 的 confirm 规格。
- 切分文件、问题顺序、输入 rollout 文件和配置均保存 SHA256。若生成池补题，补题顺序必须在计算 04 指标前冻结。
- discovery/confirm 的实际问题数、rollout 数、2+2 与严格 3+3 数量在运行前写入 manifest。严格 3+3 作为敏感性 cohort。
- 在 hidden forward 前只根据题数与每题 correct/wrong 数做功效表，目标是主 confirm cohort 对 within-Q AUC `0.55` 至少有 `0.80` 功效。若现有生成池达不到该目标，必须先按冻结 seed 顺序扩充生成池，再重新冻结最终 question split；不能在查看 04 hidden 指标后扩池。

### 1.1 Full 与 Partial Chunk

- `full chunk`：恰好包含 256 个 thinking tokens，进入所有主分析。
- `terminal partial chunk`：最后不足 256 tokens 的尾段，标记 `is_partial=true` 并单独保存。
- partial chunk 可以用于单点 raw entropy 和纵向 layer profile 的敏感性图。
- partial chunk 不进入横向 movement、turning、path 和 straightness 主分析，因为最后一步的 token 间隔不足 256。
- partial movement 只作敏感性分析，必须报告实际 `token_gap`；不得与 full-chunk movement 混合求均值。
- 若 partial chunk 少于 25 tokens，`R_last25` 使用该 partial 的全部 tokens，同时记录 `tail_token_count`。

## 2. 三种 Chunk 表示

对 rollout `i`、chunk `c`、layer `l`，hidden state 记为 `h_{i,t,l}`，构造：

\[
R^{last}_{i,c,l}=h_{i,e_c,l}
\]

\[
R^{mean}_{i,c,l}=\frac{1}{|C_{i,c}|}\sum_{t\in C_{i,c}}h_{i,t,l}
\]

\[
R^{last25}_{i,c,l}=\frac{1}{|T^{25}_{i,c}|}
\sum_{t\in T^{25}_{i,c}}h_{i,t,l}
\]

其中 `e_c` 是 chunk 最后一个 token，`T25` 是 chunk 的最后 25 个 tokens。

- `R_last`：prefix endpoint snapshot，但可能受最后一个 token 的词义和句法影响。
- `R_mean`：最平滑，复用早期 chunk-mean 语义，但可能稀释短暂变化。
- `R_last25`：endpoint 与局部去噪之间的折中。

三种表示都是 discovery 轴。discovery 结束后，基于机制解释、连续区域稳定性、边界可靠性和 length increment 冻结一个 confirm 主表示；另外两种只作为冻结指标的 robustness checks。不得在 confirm 结果出来后更换主表示。

## 3. 共用几何定义

给定有序点列：

\[
P=[p_0,p_1,\ldots,p_{K-1}],\qquad s_k=p_{k+1}-p_k
\]

### 3.1 Step Magnitude

\[
m_k=\|s_k\|_2
\]

保存局部序列，并计算 `mean / median / std / max / late_mean / late_max / ratio_late_early`。横向累计 path length 作为预先声明的长度敏感阴性对照。

对任意局部标量序列，`early`/`late` 固定为最前/最后 25% 的有效值；窗口大小为 `max(4, ceil(0.25*n_valid))`。若整条路径少于 8 个有效值，则所有 early/late 汇总记为缺失，不进入候选网格。

### 3.2 Absolute State Angle

绝对 hidden state 与原点连线的夹角：

\[
a^{state}_k=
\arccos\frac{p_k^\top p_{k+1}}
{\|p_k\|_2\|p_{k+1}\|_2+\epsilon}
\]

它回答“相邻绝对表示相对原点旋转了多少”，与 trajectory turning 不同。该指标作为 discovery 项单独命名为 `state_angle_raw`；由于 residual stream 可能存在强公共方向，不作为预设主假设。

并行计算去公共分量版本。令 \(\mathcal D\) 为 discovery rollout 集，只在该集合中按 representation 与 layer 估计无标签、rollout 等权的公共向量：

\[
\mu^{disc}_{r,l}
=
\frac{1}{|\mathcal D|}\sum_{i\in\mathcal D}
\left(
\operatorname{mean}_{c\in full\ chunks(i)}R^r_{i,c,l}
\right)
\]

令：

\[
\widetilde R^r_{i,c,l}=R^r_{i,c,l}-\mu^{disc}_{r,l}
\]

再使用 \(\widetilde R\) 代替原公式中的 \(p\)，分别计算横向与纵向的 `state_angle_demean`。它回答“去掉该层跨样本共享的残差流基线后，相邻状态方向改变了多少”。confirm 阶段必须使用 discovery 已冻结的 \(\mu^{disc}_{r,l}\)，不得用 confirm 数据重新估计公共向量。`state_angle_raw` 与 `state_angle_demean` 并列保存。

### 3.3 Trajectory Turning

真正的轨迹转折建立在相邻步向量上：

\[
c^{turn}_k=
\frac{s_{k-1}^\top s_k}
{\|s_{k-1}\|_2\|s_k\|_2+\epsilon}
\]

\[
\theta^{turn}_k=\arccos(\operatorname{clip}(c^{turn}_k,-1,1))
\]

同时保存 `turn_cos` 和 `turn_angle`。为避免重复进入候选网格：

- `direction_consistency = mean(turn_cos)` 作为平均方向一致性指标；
- `turn_angle` 只报告 `std(turn_angle)` 与 `late_std(turn_angle)`，解释为曲率/转向不稳定性；
- 不把 `mean(turn_angle)` 作为独立候选。它与 `direction_consistency` 并非严格相等，但信息高度冗余。

零位移或非有限方向必须排除并报告排除率。

### 3.4 Path、Net 与 Straightness

对一个至少含两个有效 displacement 的 block `B`：

\[
L_B=\sum_{k\in B}\|s_k\|_2,
\qquad
N_B=\left\|\sum_{k\in B}s_k\right\|_2
\]

\[
S_B=\frac{N_B}{L_B+\epsilon},
\qquad
D_B=\log(L_B+\epsilon)-\log(N_B+\epsilon)
\]

同时计算 whole-sequence 与 rolling-4-displacement 版本。rolling 版本用于折线图：横向约覆盖 1024 tokens，纵向约覆盖 4 次 layer update。单步不计算 straightness，因为单步 straightness 恒为 1。

## 4. Entropy：两种运算顺序

对任意向量 `x in R^D`，先做 per-vector coordinate centering：

\[
\widetilde x_j=x_j-\frac1D\sum_mx_m
\]

\[
p_j=\frac{\widetilde x_j^2}{\sum_m\widetilde x_m^2+\epsilon},
\qquad
H(x)=-\frac{\sum_jp_j\log(p_j+\epsilon)}{\log D}
\]

### 4.1 Pooled-State Raw Entropy

对三种 chunk 表示分别计算：

\[
H(R^{last}_{i,c,l}),\qquad
H(R^{mean}_{i,c,l}),\qquad
H(R^{last25}_{i,c,l})
\]

其中 `H(R_mean)` 的含义是：先把 chunk 内所有 token hidden vectors 平均成一个向量 `R_mean`，再对这个平均向量计算 coordinate-energy entropy。它不等于 chunk 内 token entropy 的平均。

### 4.2 Token-First Raw Entropy Control

先对每个 token 算 `H(h_{i,t,l})`，再在固定 256-token chunk 内保存：

\[
\operatorname{mean}_{t\in C}H(h_t),\quad
\operatorname{median}_{t\in C}H(h_t),\quad
P90_{t\in C}H(h_t),\quad
\operatorname{top25mean}_{t\in C}H(h_t)
\]

这组对照用于区分：新信号来自更小的 chunk、来自 burst-sensitive 聚合，还是来自先 pooling hidden 再算 entropy 的运算顺序变化。

### 4.3 Difference Entropy

横向与纵向的每个步向量均计算：

\[
H^{diff}_k=H(s_k)
\]

横向 `s_k` 是 chunk-to-chunk update；纵向 `s_l` 是 layer-to-layer update。raw entropy 是单点量，difference entropy 才是方向相关的更新量。

### 4.4 Token-First Difference-Entropy Control

为了与旧实验的计算顺序直接对照，额外在线计算 token-level update entropy：

\[
H^{time-token}_{t,l}=H(h_{t,l}-h_{t-1,l})
\]

\[
H^{layer-token}_{t,l}=H(h_{t,l}-h_{t,l-1})
\]

随后在每个固定 256-token chunk 内分别保存 `mean / median / P90 / top25mean`。这与 `H(R_c-R_{c-1})`、`H(R_l-R_{l-1})` 不同：前者逐 token 先算差的熵，后者先把 chunk 表示压成一个向量，再计算 chunk/layer 级差向量的熵。

## 5. 横向与纵向

### 5.1 横向 Temporal Trajectory

固定 layer `l`：

\[
P^{time}_{i,l}=
[R_{i,0,l},R_{i,1,l},\ldots]
\]

discovery 对 hidden-state index `0..36` 的全部 37 层、三种 representation 计算局部 magnitude、raw/demean state angle、turning、rolling straightness、raw entropy 和 difference entropy。旧实验常看的 L24/L36 只作为历史参考，不限制本轮全层扫描。whole path length/straightness 同时保存，但累计量必须通过 length gate。confirm 只计算 discovery 冻结的 layer 或连续 layer band。

### 5.2 纵向 Depth Trajectory

固定 chunk `c`：

\[
P^{layer}_{i,c}=
[R_{i,c,0},R_{i,c,1},\ldots,R_{i,c,36}]
\]

对所有 full chunk、三种 representation 计算相邻层 magnitude、raw/demean state angle、turning、rolling-layer straightness、raw entropy 和 difference entropy。纵向包含 `0->1, 1->2, ..., 35->36` 共 36 个 layer update，update 指标按终点层 `1..36` 编号；需要两个相邻 update 的 turning 指标按终点层 `2..36` 编号。纵向不受 chunk-count 累加，但仍须控制 think length、相对进程和 partial/coverage 选择偏差。

### 5.3 纵向层间尺度诊断

纵向更新：

\[
\Delta^v_{i,c,l}=R_{i,c,l+1}-R_{i,c,l}
\]

表示每一层新写入 residual stream 的量。embedding 到早期层与中后层的更新尺度可能系统性不同，因此 raw vertical movement 可能被少数层的基线尺度主导。discovery 中除 raw 值外，按 representation 与 update layer 估计无标签稳健尺度：

\[
q^{disc}_{r,l}
=
\operatorname{median}_{i\in\mathcal D}
\left[
\operatorname{median}_{c\in full\ chunks(i)}
\|\Delta^{v,r}_{i,c,l}\|_2
\right]
\]

纵向方向量的近零阈值固定为：

\[
\tau^{v}_{r,l}
=
\max\left(10^{-8},10^{-4}q^{disc}_{r,l}\right)
\]

任一参与 cosine/angle 的 update norm 不高于对应 \(\tau^{v}_{r,l}\) 时，该方向量记为无效。

并行报告：

\[
m^{v,scaled}_{i,c,l}
=
\frac{\|\Delta^{v}_{i,c,l}\|_2}
{q^{disc}_{r,l}+\epsilon}
\]

冻结纵向候选前必须同时检查 raw 与 scale-normalized correct/wrong layer profile。若效应只出现在 `0->1` 等天然大尺度边界、在尺度归一化后消失，或主要由接近零更新导致不稳定角度，则判为尺度伪影，不得冻结。cosine 本身虽对整体缩放不敏感，但近零 update 的角度不稳定，仍须按上述 \(\tau^v_{r,l}\) 排除并报告各层排除率。confirm 使用冻结的 \(q^{disc}_{r,l}\) 与 \(\tau^v_{r,l}\)，不重新估计。

本轮暂不计算跨 rollout correct/wrong set-direction 或 prototype direction；单 rollout 内的 direction consistency 由 `turn_cos` 表示。跨 rollout 方向性留作独立后续实验，避免本轮网格继续膨胀。

## 6. 折线图与二维 Atlas

所有 correct/wrong 曲线使用 question-equal aggregation：先在每题内分别平均 correct 和 wrong rollout，再跨 question 平均；95% CI 以 question bootstrap 计算。不得把所有 rollout 直接池化。每张随位置变化的图必须同时报告有效 question 数。

### 6.1 横向折线

- absolute chunk index：x 为 `0,1,2,...`，使用真实 full chunks，不插值；同时画 coverage。
- end alignment：最后一个 full chunk 为 `0`，向前为 `-1,-2,...`；partial terminal 不作为 movement endpoint。
- relative progress：只把已经算好的 chunk scalar 按 chunk center/endpoint 的相对位置分入固定 20 个等宽桶；先在 rollout×bin 内平均，再做 question-equal aggregation。不得对 hidden vectors 或角度曲线插值。
- 对每个 representation、metric 和 layer 输出 correct/wrong 趋势；报告可以提供 layer selector，静态论文图使用冻结 layer/band。

### 6.2 纵向折线

- x 为 layer `0..36`；difference/turning 指标的 x 表示对应 layer update 的终点层。
- 每个可用 full chunk 均输出一幅 correct/wrong layer profile，而不只画“末段”和“全 chunk 平均”。
- vertical movement 同时绘制 raw 与 discovery-scale-normalized profile，并标出各 update layer 的有效方向数/排除率。
- 输出 multi-page PDF 或 HTML selector，以 representation、metric、chunk 切换；同时输出 early/middle/late/terminal-fixed 的静态汇总图。
- terminal partial 的纵向 profile 单独成图，不与最后一个 full chunk 混合。

### 6.3 二维 Atlas

- 横向指标：`layer × chunk` 的 correct-minus-wrong effect、within-Q AUC 和有效问题数热图。
- 纵向指标：`layer/update × chunk` 的对应热图。
- 三种 representation 分开，不在同一 cell 内平均。
- primary-support 阈值预先固定为 `N_min=max(20, ceil(0.5*n_cohort_questions))`；低于阈值的 cell 只显示 coverage，不进入候选区域或 confirm 冻结。

## 7. Discovery、连续区域与冻结

### 7.1 Discovery

discovery 半允许查看全部 representation × direction × metric × layer × chunk atlas 和折线图，但不得把最高单格直接作为候选。

对每个 `representation × metric × direction` family 使用 cluster-based question permutation：

1. 在 cell 层面得到每道题的 correct-minus-wrong 差，再计算跨题 one-sample `t` statistic；
2. cell-forming threshold 固定为 `|t| >= 2.0`，连接相邻 layer/chunk 的同方向 cell；
3. cluster mass 定义为 cluster 内 `|t|` 之和；
4. 在每道题内重排 rollout correct/wrong 标签，保持该题正确/错误数量不变；
5. 每次置换保存最大 cluster mass，4000 次形成 family-wise 零分布；
6. 报告 cluster-level permutation p-value、layer/chunk 范围、representation 一致性和 coverage。

### 7.2 冻结不超过 3 个主假设

每个冻结候选必须明确：

```text
representation
horizontal / vertical
metric 与精确公式
raw/demean state-angle 版本，或 raw/scale-normalized vertical 版本
layer 或连续 layer band
chunk index / relative-progress / end-aligned window
rolling window 或 late window
聚合算子
higher-is-correct / lower-is-correct
完整/partial chunk 规则
```

候选优先级：

1. 连续 layer×chunk 区域、question sign 稳定且至少两种 representation 同方向；
2. 纵向或横向的非累计、可定位指标；
3. 横向末端收敛指标；
4. path length 等累计量仅作为预声明阴性对照。

discovery 后冻结一个主 representation；另外两种 representation 只对同一冻结指标做 robustness，不形成新的 confirm 主检验。最多三个候选必须预先排成 `primary -> secondary-1 -> secondary-2` 的固定层级顺序。

## 8. Confirm 与两级判据

### 8.1 冻结与盲法

confirm hidden 指标在 discovery 候选、主 representation、公共向量、纵向尺度与 exclusion threshold 全部写入冻结 manifest 后才可查看。confirm 只计算和报告冻结候选，不重新扫描 layer、chunk、representation、符号或聚合方式。

### 8.2 Gate 1：信号复现

逐题 correct-vs-wrong pairwise AUC，question 等权；4000 次 question bootstrap 的 95% CI 下界必须 `>0.5`。

### 8.3 Gate 2：RL 候选 Length Gate

在 5-fold question-grouped OOF 中，比较 `feature + think_length` 与 `think_length-only` 的逐题 AUC；配对 question bootstrap 的 AUC 增量 95% CI 下界必须 `>0`。

横向和纵向候选均必须通过两个 gate。两级都通过才进入后续 RL credit-assignment 设计；只通过 Gate 1 视为内部相关信号；Gate 1 失败视为未复现。

### 8.4 层级检验与敏感性分析

若冻结多于一个候选，采用固定的 `primary -> secondary-1 -> secondary-2` 层级检验：只有前一个候选的 Gate 1 通过，下一个候选的 Gate 1 才是正式检验；否则后续候选只报告描述性 CI。每个候选只有 Gate 1 正式通过后才检验自己的 Gate 2。

敏感性分析可加入 `n_full_chunks / is_partial / relative_progress`，但不得替代预注册的 think-length gate，也不得改变冻结的候选顺序。

## 9. 计算与存储

### 9.1 执行方式

- H200 单卡、batch size 1、tmux、question-level 断点续跑；新建隔离 run directory，不覆盖已有结果。
- forward 中在线构造三种 chunk representations 与 token-first entropy scalar；主指标逐 rollout 计算并释放，不同时保留同题 8 条 rollout 的向量。
- hidden index `0..36` 全扫。三种 representation 相比单表示会使 chunk-vector 缓存约增至 3 倍，但不应把完整 token hidden 永久写盘。
- 永久只保存 scalar parquet、metadata、coverage、排除率和固定少量预先选定 audit questions 的 fp16 vectors。

### 9.2 资源量级与 Demean 处理

按 hidden size 4096、64 个 full chunks、37 个 hidden-state index、fp16 估算，单条最长 rollout 的三表示 chunk-vector cache 为：

\[
64\times3\times37\times4096\times2\ \mathrm{bytes}
\approx56\ \mathrm{MiB}
\]

若直接保留 16384 token 的 37 层 fp16 hidden，仅 activation tensor 就约 `4.63 GiB/rollout`，还不含模型权重、视觉 token、attention workspace 和临时张量。因此实现应在 inference mode 下按层/rollout在线聚合，不长期持有或写盘完整 token hidden。

`state_angle_demean` 需要先得到 discovery 公共向量。实现前在以下两条路线中按 smoke 结果冻结一种：

1. discovery 临时保存三表示 chunk vectors，计算 \(\mu^{disc}\) 与 demean scalar 后立即删除；所需临时盘约为 `56 MiB × discovery rollout 数`，另留至少 20% 余量；
2. 若临时盘不足，则先在线累计 \(\mu^{disc}\)，再只对 discovery 做第二次 forward 计算 demean scalar；该路线节省磁盘但增加约一次 discovery forward 成本。

confirm 若冻结了 demean 指标，直接使用 discovery 公共向量，一次 forward 即可；若未冻结，则 confirm 不计算该探索项。

### 9.3 上机前 Smoke Gate

正式任务前先跑 `1 道题 × 8 rollouts`，其中必须包含一条接近最大长度的 rollout，并记录：

- 峰值 GPU 显存、CPU RAM、临时磁盘与单 rollout 耗时；
- 三表示×37 层输出 shape、NaN/Inf、零位移排除率与断点恢复；
- scalar 与少量直接计算 audit 的数值一致性；
- 预估正式运行时长和 demean 路线。

只有在无 OOM、GPU 至少保留约 15% 显存余量、临时盘满足所选路线且 smoke 输出审计通过后，才在 tmux 启动正式 discovery。资源表与路线选择写入 run manifest。

## 10. 输出

- discovery/confirm manifest、SHA256、eligibility 与 overlap audit；
- chunk-level raw scalar、horizontal metrics、vertical metrics parquet；
- horizontal absolute/end-aligned/relative-progress correct-wrong curves；
- 每个 full chunk 的 vertical layer profile、multi-page PDF 或 HTML selector；
- layer×chunk effect/AUC/coverage atlas；
- cluster-permutation summary；
- frozen hypothesis manifest；
- confirm gate、OOF、bootstrap 和 Markdown 报告。

## 11. 本轮明确不做

- 不训练 RL；
- 不把 discovery 最高单格直接当结论；
- 不在 confirm 结果后更换 representation、layer、chunk 或指标方向；
- 不重新检验旧的 `L14-L19 / bin6` entropy band；新的 fixed-chunk raw/diff entropy 是独立 exploratory family；
- 暂不加入跨 rollout set-direction、prototype direction 或正确轨迹几何；
- 不覆盖或修改 H200 上已有实验目录。
