# Qwen2.5-7B GRPO 双轴 Hidden-State 动力学发现实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在纯文本、短 response 的标准 GRPO 训练中，系统测量 hidden state 沿 response 时间轴的横向轨迹动力学和沿 Transformer 深度轴的纵向表示变换，定位这些量在回答不同阶段随训练发生的变化，并筛选可进入多 seed 验证和 process-reward 实验的内部信号。

**Architecture:** 使用 `Qwen2.5-7B-Instruct` 在 MATH 上进行单 seed 标准 GRPO discovery run。固定一组 held-out 题，在 base 与五个训练 checkpoint 上用相同 decoding 生成多条 rollout，再离线 forward 一次性取得所有层 hidden states；以 128-token window、32-token stride 构造 `mean` 与 `last` 两种表示，在线归约为标量，不永久保存完整 token hidden states。横向几何在四个冻结层锚点 `L3/L12/L24/Lfinal` 分别计算，纵向使用完整深度；token-level H8 以最终层为主并保留逐层探索 profile。`stride=32` 已由正式训练前的 length-only smoke 按第 3.1 节冻结。

**Tech Stack:** Python、PyTorch、Transformers、标准 GRPO/veRL 训练栈、vLLM rollout、NumPy/SciPy、pandas/Parquet、scikit-learn/statsmodels、Matplotlib/Seaborn。

## Global Constraints

- 本轮是 **single-model / single-method / single-seed discovery**，不修改 reward，不把任何 hidden 指标加入 GRPO。
- 模型固定为纯文本 `Qwen2.5-7B-Instruct`；不使用 VL 模型，不使用 Thinking 模型。
- 数据固定为 MATH：train split 的 Level 3-5 用于训练，test split 中固定抽取 256 题作为 hidden-state evaluation cohort；train/test 题目不得重叠。
- GRPO group size 固定为 8；训练期间的 reward 只使用数学答案正确性与规定的格式检查。
- response 上限固定为 `max_new_tokens=1536`；主分析排除 truncated rollout，并单独报告截断率。
- 训练 checkpoint 固定为 `base / 20% / 40% / 60% / 80% / final` 六个位置。
- 每个 checkpoint 对相同的 256 道 held-out 题各生成 8 条评估 rollout，总评估规模为 `6 x 256 x 8 = 12288` 条。
- 评估题、prompt template、temperature、top-p、最大长度和每条 rollout 的随机种子表在第一次评估前冻结，所有 checkpoint 完全复用。
- hidden 指标只计算生成 response，不包含 system prompt、user prompt 或 padding。
- 轨迹主规格已冻结为 `window=128, stride=32`；`mean_w128_s32` 为主表示，`last_s32` 为预注册敏感性表示，不允许事后选择效果更好的表示作为主结果。
- 横向 H1-H7 的深度锚点固定为 `L3 / L12 / L24 / Lfinal`；四层均报告，不得在结果后按效果选择单层。`Lfinal` 从 `model.config.num_hidden_layers` 读取。
- H8 的最终层为单一 primary view；H8 锚点层与全层 profile 只作探索性定位，不增加主假设。
- 所有四个 response 阶段都报告；`75%-100%` 作为 terminal-control，process-reward 候选优先来自 `0%-75%`。
- response length、policy/token entropy、mean log probability、hidden norm、correctness 和 truncation 是强制控制变量。
- 不因 discovery 结果修改 checkpoint、stage、window、stride、aggregation、指标符号或统计检验；任何新增分析只能标记为 post-hoc exploratory。

---

## 1. 研究问题与论文叙事

### 1.1 核心问题

传统 RLVR 通常在 token action space 中讨论 exploration/exploitation，并用词表熵、log probability 或 response length 作为代理。ER/VERL 将整条 hidden-state response trajectory 压缩为 ER、ERV、ERA，但最后仍形成一个 trajectory-level auxiliary signal，没有显式定位内部变化发生在回答哪个阶段，也没有区分 response 时间轴与网络深度轴。

本实验研究一个二维过程：

\[
\text{training checkpoint}
\times
\text{response progress}
\times
\{\text{horizontal},\text{vertical}\}.
\]

- **Horizontal dynamics：** response 沿 token/chunk 时间推进时，hidden state 在表示空间中移动多远、覆盖多少方向、是否转向、绕路或收敛；这些轨迹分别在冻结的浅层、中层、深层和最终层测量，以观察语义几何在网络深度上的形成过程。
- **Vertical dynamics：** 同一个 chunk 的表示沿 Transformer layers 被如何重写，变换是集中在少数层还是分散在多层，层间路径是否连贯。

“横向更接近语义探索、纵向更接近逻辑深入”只作为待检验机制假设，不在发现实验之前当作事实。

### 1.2 可证伪假设

- **H-A：Stage redistribution。** GRPO 不只是改变整条回答的全局标量，而是重新分配不同 response 阶段的 hidden-state dynamics。
- **H-B：Axis complementarity。** 横向和纵向量在控制 response length 与词表 entropy 后仍具有不完全重合的训练趋势。
- **H-C：Outcome relevance。** 候选指标在同一 checkpoint 内对 correct/wrong rollout 的差异方向，与其随训练和 reward 改善的变化方向一致。
- **H-D：Non-terminal availability。** 至少一个候选信号在 `0%-25%`、`25%-50%` 或 `50%-75%` 已出现，而不是只在答案即将结束时出现。

### 1.3 本轮不做的事情

- 不训练 hidden reward model。
- 不将 H1/H2 或本轮新指标加入 GRPO advantage。
- 不声称指标具有 causal effect。
- 不以单 seed 趋势支持跨模型、跨算法或普遍机制结论。
- 不把正确/错误 AUROC 当作唯一发现标准。

---

## 2. 数据、模型与 GRPO

### 2.1 训练数据

- 使用 MATH train split 中难度 Level 3-5 的题目。
- 以标准化后的 `problem` 作为 prompt，以数据集给出的最终数学答案作为 verifiable target。
- 去除空题、无法解析 gold answer、与 held-out evaluation cohort 重复的题目。
- 保存训练 manifest：`question_id / source_split / subject / level / gold_answer / prompt_hash`。

Level 3-5 用来减少 group 内全对或全错。GRPO 若一个 group 的 8 条 rollout reward 全同，则 group-relative advantage 为零；正式训练前必须通过 smoke 测量 mixed-reward group 比例。

### 2.2 Evaluation cohort

- 从 MATH test split 固定抽取 256 题。
- 按 `level x subject` 分层抽样；同一题在六个 checkpoint 中重复使用。
- 每个 checkpoint 每题生成 8 条 rollout；`rollout_slot` 固定为 `0..7`，但不同 checkpoint 不共享生成随机数状态；保存预先生成的 `question_id x rollout_slot x checkpoint` seed 表。
- evaluation cohort 不参与 GRPO 参数更新，也不用于 checkpoint selection。

### 2.3 模型与训练

- Base model：`Qwen2.5-7B-Instruct`。
- 方法：标准 GRPO，不加入 ER/ERV/ERA、H1/H2 或本计划的任何 auxiliary signal。
- Group size：8。
- Reward：数学答案等价性 + 必需格式检查；分别保存 `answer_reward` 与 `format_reward`，主 correctness 使用答案等价性。
- 单 seed discovery；训练 seed、rollout seed、data-order seed 分开记录。
- 保存 base、20%、40%、60%、80%、final 六个 checkpoint，训练总 step 在启动前由实际训练集大小和 batch 配置确定，并写入 frozen run manifest。

### 2.4 Short-response 与截断规则

- `max_new_tokens=1536`。
- response tokenization 使用模型自己的 tokenizer；所有进度、window 和长度均以 token ID 计数，不使用字符数。
- `finish_reason=length` 或未正常生成最终答案的 rollout 标记为 truncated。
- 主分析仅使用 non-truncated rollout。
- 所有报告同时给出：总截断率、按 checkpoint 截断率、按 correctness 截断率、按 response-length 桶截断率。

---

## 3. Preflight 与冻结顺序

### 3.1 Length-only smoke

在 base model 上先对 evaluation cohort 的前 100 题各生成 4 条 rollout，只读取 token length、finish reason、reward 和 group mixed 状态，不读取任何 hidden 指标。

冻结规则：

1. 若 non-truncated rollout 的中位数在 `stride=64` 下能形成至少 12 个轨迹点，则保留 `window=128, stride=64`。
2. 若中位轨迹点少于 12，则只把 stride 改为 32，window 仍保持 128；该修改发生在读取 hidden 结果前。
3. 若截断率超过 5%，保持 `max_new_tokens=1536` 不变，但将截断作为独立状态，不通过提高上限事后改变 short-response setting。
4. 若 GRPO smoke 中 mixed-reward group 少于 20%，训练题池限制在更高难度/更低 base pass-rate 的预定义 hard subset；不得根据 hidden 指标选题。

### 3.2 GRPO smoke

- 运行最小训练步数，验证 reward、group advantage、KL、checkpoint 保存和恢复。
- 验证训练过程不会输出 NaN/Inf，且至少存在非零 advantage group。
- 验证 base 与 smoke checkpoint 的固定 evaluation prompt 能够用同一模板生成。
- smoke checkpoint 永久排除出正式六 checkpoint 分析。

### 3.3 Hidden extraction smoke

- 随机选择 2 题 x 2 rollout x 2 checkpoint。
- 验证 prompt/response token 边界、所有层 hidden shape、mean/last 轨迹点数和 final endpoint。
- 验证同一次 forward 同时产出两种 representation 和全部标量，无需重复 forward。
- 验证完整 hidden tensors 在标量归约后立即释放，不写入磁盘。

---

## 4. 符号、轨迹点与两种表示

对 checkpoint \(s\)、题目 \(q\)、rollout \(i\)，生成 response token 数为 \(T_{sqi}\)。模型共有 \(L+1\) 组 hidden states，hidden dimension 为 \(D\)；运行时从模型 config 读取 \(L,D\)，不在代码中硬编码。

轨迹 endpoint：

\[
e_k=128+(k-1)\cdot stride,
\]

保留所有 \(e_k\le T\)，并额外加入最终 endpoint \(e_K=T\)；若最终 endpoint 与上一点重复则去重。最终 mean point 使用 response 最后最多 128 个 token，不使用 padding。

对每个 endpoint、每层 \(l\) 构造：

\[
x^{mean}_{k,l}
=
\frac{1}{|W_k|}
\sum_{t\in W_k}h_{t,l},
\qquad
W_k=[\max(1,e_k-127),e_k],
\]

\[
x^{last}_{k,l}=h_{e_k,l}.
\]

- `hidden_states[0]` 是 embedding 输出；Transformer block 输出按其 block 编号索引。横向锚点固定为：

  \[
  \mathcal L_h=\{3,12,24,L\},
  \qquad L=\texttt{model.config.num\_hidden\_layers}.
  \]

  下文 `L3/L12/L24/Lfinal` 分别指这四个 block 输出。对每个 \(l\in\mathcal L_h\)，H1-H7 独立使用轨迹 \(x_{1,l},\ldots,x_{K,l}\) 计算；`anchor_layer` 是预先冻结的分析因子，不得事后选层。
- H8 以 \(l=L\) 为 primary；四个锚点及 \(l=0,\ldots,L\) 的全层 profile 仅用于探索性定位。
- Vertical 使用所有 \(l=0,\ldots,L\)。
- `mean_w128_s32` 是 primary representation。
- `last_s32` 是 sensitivity representation。
- 同一 rollout 的两种 representation 必须来自同一次 hidden forward。

---

## 5. Response progress：四阶段与两种聚合

每个 endpoint 的相对进度：

\[
p_k=e_k/T.
\]

四阶段固定为：

\[
B_1=[0,0.25),\quad
B_2=[0.25,0.50),\quad
B_3=[0.50,0.75),\quad
B_4=[0.75,1.00].
\]

### 5.1 Local-stage 指标

只使用当前阶段内的点或位移：

\[
M^{local}_{sqi,b}
=
\operatorname{Agg}_{k:p_k\in B_b}m_{sqi,k}.
\]

适用于 movement、turning、angular velocity、length-weighted turning、layer-update norm、vertical angle、layer-update turning 与 entropy。

### 5.2 Cumulative-prefix 指标

在 25%、50%、75%、100% 边界，使用不超过边界的全部轨迹点：

\[
M^{cum}_{sqi,b}=M(x_{1:k_b}),
\qquad
k_b=\max\{k:p_k\le b/4\}.
\]

定义起点 \(M^{cum}_{sqi,0}=0\)；若第一个有效 endpoint 已经晚于某个早期边界，则该边界仍按 coverage 记缺失，不把第一个 endpoint 伪装成零进度状态。

若某边界前没有足够点，记为缺失，不向前插值。适用于 path、ER/ERV/ERA、directional ER 和 vertical ER。

阶段增量：

\[
\Delta M_{sqi,b}
=
M^{cum}_{sqi,b}-M^{cum}_{sqi,b-1}.
\]

### 5.3 Coverage

每条 rollout、每个 representation、每个 stage、每个指标单独记录有效性：

- movement：至少 1 个有效位移；
- turning angle：至少 2 个相邻有效位移，即至少 3 个轨迹点；
- angular velocity：至少 3 个相邻有效位移，即至少 4 个轨迹点；
- ER：至少 2 个非重复状态点；
- ERV：至少 2 个 prefix metric 点；
- ERA：至少 3 个 prefix metric 点；
- H1-H7：上述横向 coverage 对四个 `anchor_layer` 分别判断，不因某一 anchor 缺失而删除其他 anchor；
- H8 token state entropy：window 内至少 1 个有效 token scalar；time-diff entropy：window 内至少 1 个有效相邻 token difference，stage 内至少 1 个有效 window；
- layer-update norm / layer-difference entropy：至少 1 个有效 layer update；
- raw activation / centered state / robust-z entropy：至少 1 个有效 layer state；
- raw state angle / demean state angle：至少 2 个有效相邻 layer states；
- layer-update turning angle：至少 2 个连续、norm 超过阈值的 layer updates，即至少 3 个有效 layer states；
- vertical path / vertical ER：至少 3 个有效 layer states，且层间位移 norm 超过数值阈值。

不能用一个全局 complete-case gate 丢掉所有存在局部缺失的 rollout。每张图和每个 cell 都必须标注有效 rollout 数与有效 question 数。

---

## 6. 横向 8 个指标家族

对 H1-H7，以下定义在每个冻结锚点 \(l\in\mathcal L_h\) 独立执行；为简洁起见省略 layer 下标，写作 \(x_k=x_{k,l}\)。每个输出都记录 `anchor_layer`。H8 是 token-level 家族，最终层为 primary，锚点/全层结果只作探索性 profile。定义相邻位移、长度和单位方向：

\[
d_k^h=x_k-x_{k-1},
\qquad
r_k^h=\|d_k^h\|_2,
\qquad
u_k^h=\frac{d_k^h}{r_k^h+\epsilon}.
\]

norm 低于冻结阈值的位移不参与角度计算，并报告排除率；不能把零位移角度设为 0。

### H1. Movement magnitude

\[
r_k^h=\|d_k^h\|_2,
\]

为消除 hidden dimension \(D\) 对绝对欧氏距离的机械影响，同时保存每坐标 RMS 位移：

\[
r_{k,rms}^h
=
\frac{\|d_k^h\|_2}{\sqrt D}.
\]

\[
r_{k,rel}^h
=
\frac{\|d_k^h\|_2}
{\|x_{k-1}\|_2+\epsilon}.
\]

每 stage 对 raw movement、RMS movement 与 relative movement 分别保存 mean、median、P90；primary representative 仍冻结为 `median_relative_movement`。`r_{k,rms}^h` 是跨 hidden dimension 可比的绝对 movement 诊断量，不替代 relative primary。

### H2. Path geometry

对任一局部阶段或累计 prefix 的位移集合：

\[
L^h=\sum_k\|d_k^h\|_2,
\qquad
N^h=\left\|\sum_kd_k^h\right\|_2,
\]

\[
S^h=\frac{N^h}{L^h+\epsilon},
\qquad
D^h=\log(L^h+\epsilon)-\log(N^h+\epsilon).
\]

- \(L^h\)：path length；
- \(N^h\)：net displacement；
- \(S^h\)：straightness；
- \(D^h\)：log-detour。

Primary representative 为 `straightness`；\(L,N,D\) 作为解释和诊断必须同时报告。

### H3. Turning angle

\[
\theta_k^h
=
\arccos\left(
\operatorname{clip}(u_k^{h\top}u_{k-1}^h,-1,1)
\right).
\]

每 stage 保存 median、mean、standard deviation、P90；primary representative 为 `median_turn_angle`。

### H4. Angular velocity

\[
\omega_k^h=\theta_k^h-\theta_{k-1}^h.
\]

同时保存 signed \(\omega_k\) 与 \(|\omega_k|\)。Primary representative 为 `P90(abs_angular_velocity)`，用于捕捉突然转向，而不是把正负变化相互抵消。

### H5. State-trajectory ER / ERV / ERA

累计到第 \(k\) 个 endpoint 的状态矩阵：

\[
Z^h_{1:k}=
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_k
\end{bmatrix}.
\]

对每个 prefix 先在轨迹点维度中心化：

\[
\bar x_k=\frac1k\sum_{a=1}^k x_a,
\qquad
Z_{1:k}^{h,c}=Z_{1:k}^{h}-\mathbf 1\bar x_k^\top.
\]

设 \(Z_{1:k}^{h,c}\) 的非零奇异值为 \(\sigma_j\)：

\[
p_j=\frac{\sigma_j}{\sum_r\sigma_r},
\qquad
ER_k^{h,c}
=
\exp\left(-\sum_jp_j\log(p_j+\epsilon)\right).
\]

Primary ER 使用 \(ER_k^{h,c}\)；因此 centered prefix ER 序列进入 ERV/ERA。该口径与 VERL 附录的 mean-centered Gram 实现一致。未中心化 ER/ERV/ERA 作为 sensitivity 完整保存，但不进入 16 个代表量。

令 \(m_j=ER_j^{h,c}\)，历史偏差为：

\[
\delta_j
=
m_j-\frac{1}{j-1}\sum_{a<j}m_a.
\]

\[
ERV
=
\frac{1}{K-1}\sum_{j=2}^{K}\delta_j,
\qquad
ERA
=
\frac{1}{K-2}\sum_{j=3}^{K}(\delta_j-\delta_{j-1}).
\]

Primary representative 为 cumulative `centered_ERV`；`centered_ER_final` 与 `centered_ERA` 必须同时画出，防止把 ERV 单独解释成 exploration 或 exploitation。对应的 uncentered 三量仅作 sensitivity。

### H6. Length-weighted directional ER

\[
G^h
=
\frac{1}{\sum_kr_k^h}
\sum_kr_k^hu_k^hu_k^{h\top}.
\]

设 \(G^h\) 的非零特征值为 \(\lambda_j\)：

\[
q_j=\frac{\lambda_j}{\sum_a\lambda_a},
\qquad
ER_{dir}^h
=
\exp\left(-\sum_jq_j\log(q_j+\epsilon)\right).
\]

实现时不构造 \(D\times D\) 矩阵；对行向量 \(\sqrt{r_k/\sum r}\,u_k\) 组成的小矩阵做 SVD，其平方奇异值即 \(G^h\) 的非零谱。

该量把 movement length 作为方向权重，但归一化后不保留绝对总路径长度，因此必须与 H2 的 \(L^h\) 联合解释。Primary representative 为 cumulative `directional_ER`。

### H7. Length-weighted turning

\[
\bar r_k=\frac{r_k^h+r_{k-1}^h}{2},
\]

\[
T_{weighted}^h
=
\frac{\sum_k\bar r_k\theta_k^h}
{\sum_k\bar r_k+\epsilon}.
\]

它回答“大幅移动的位置是否也发生大转向”，与“全局用了多少个方向”的 directional ER 不同。Primary representative 为 stage-local `weighted_turning`。

### H8. Token-level entropy dynamics

对任意 token hidden state \(h_{t,l}\in\mathbb R^D\)，定义坐标中心化能量熵：

\[
\tilde h_{t,l,j}=h_{t,l,j}-\frac1D\sum_{m=1}^{D}h_{t,l,m},
\qquad
p_{t,l,j}=\frac{\tilde h_{t,l,j}^2}{\sum_m\tilde h_{t,l,m}^2+\epsilon},
\]

\[
H_{state}^{coord}(t,l)
=
-\frac{\sum_j p_{t,l,j}\log(p_{t,l,j}+\epsilon)}{\log D}.
\]

相邻 token 的时间差分与其坐标中心化熵为：

\[
\Delta h_{t,l}=h_{t,l}-h_{t-1,l},
\qquad
H_{time\text{-}diff}^{coord}(t,l)=H_{state}^{coord}(\Delta h_{t,l}).
\]

另存真正未中心化的 state raw-energy entropy：

\[
H_{state}^{raw}(t,l)
=
-\frac{\sum_j q_{t,l,j}\log(q_{t,l,j}+\epsilon)}{\log D},
\qquad
q_{t,l,j}=\frac{h_{t,l,j}^2}{\sum_mh_{t,l,m}^2+\epsilon}.
\]

对每个 128-token window（stride 32）内的 token scalar 取 median，得到局部 chunk 值；再按第 5.1 节在 stage 内聚合。H8 primary 是最终层 \(l=L\) 的 `median_token_time_diff_coordinate_entropy`。`token_state_coordinate_entropy` 是与旧 Experiment04 `token_raw_entropy` 同算子但更清晰命名的对照；`token_state_raw_energy_entropy` 保留持续高激活坐标的影响。二者均完整保存，但不增加 primary family。H8 token scalar 只计算一次，`representation=token`，不得随 `mean/last` chunk representation 重复。

所有层与四个锚点的 H8 profile 输出 `layer x stage x checkpoint` 曲线/热图，仅用于探索性深度定位；不得据此事后替换最终层 primary。

---

## 7. 纵向 8 个指标家族

对每个 chunk endpoint 和 representation，记所有层 pooled state 为：

\[
y_{k,l}=x_{k,l},
\qquad l=0,\ldots,L.
\]

相邻层写入：

\[
d_{k,l}^v=y_{k,l}-y_{k,l-1},
\qquad l=1,\ldots,L.
\]

### V1. Layer-update norm

\[
a_{k,l}=\|d_{k,l}^v\|_2,
\qquad
a_{k,l}^{rel}
=
\frac{\|d_{k,l}^v\|_2}
{\|y_{k,l-1}\|_2+\epsilon}.
\]

每个 chunk 保存 across-layer mean、median、P90、maximum layer，以及 concentration：

\[
C_k^v
=
\frac{\max_la_{k,l}}
{\sum_la_{k,l}+\epsilon}.
\]

Primary representative 为 stage 内 `P90(relative_layer_update_norm)`；raw norm、across-layer mean/median/P90、maximum layer 与 concentration 同时保存。

### V2. Raw state angle

Raw angle：

\[
\phi_{k,l}^{raw}
=
\arccos\left(
\cos(y_{k,l},y_{k,l-1})
\right).
\]

### V3. State angle demean

在 base checkpoint 的完整 evaluation cohort 上，对每个 representation 和 layer 计算 label-blind、rollout-equal common：

\[
\mu_{r,l}^{base}
=
\frac{1}{R}
\sum_{i=1}^{R}
\left(
\frac{1}{K_i}\sum_{k=1}^{K_i}y_{i,k,r,l}
\right).
\]

该 common 不区分 correct/wrong，不按题目构造正确方向，并冻结后复用于所有训练 checkpoint。

Demean angle：

\[
\phi_{k,l}^{demean}
=
\arccos\left(
\cos(y_{k,l}-\mu_{r,l}^{base},
y_{k,l-1}-\mu_{r,l-1}^{base})
\right).
\]

Raw angle 与 demean angle 均作为独立主指标：其 primary representative 分别为 stage 内、across-layer 的 `median_raw_state_angle` 与 `median_state_angle_demean`。

### V4. Layer-update turning angle

对连续两次 layer update 的单位方向：

\[
v_{k,l}=
\frac{d_{k,l}^v}{\|d_{k,l}^v\|_2+\epsilon},
\qquad l=1,\ldots,L,
\]

定义：

\[
\theta_{k,l}^{v,update}
=
\arccos\left(
\operatorname{clip}(v_{k,l}^{\top}v_{k,l-1},-1,1)
\right),
\qquad l=2,\ldots,L.
\]

norm 低于冻结阈值的 update 不进入角度计算，也不把零 update 的角度设为 0；必须报告排除率。该量回答“第 \(l\) 层的写入方向是否延续第 \(l-1\) 层的写入方向”，不同于 V2/V3 的 state angle。每个 stage 保存 mean、median、P90 与 standard deviation；primary representative 为 `median_layer_update_turning_angle`。

### V5. Vertical path metrics

\[
L_k^v=\sum_l\|d_{k,l}^v\|_2,
\qquad
N_k^v=\left\|\sum_ld_{k,l}^v\right\|_2,
\]

\[
S_k^v=\frac{N_k^v}{L_k^v+\epsilon},
\qquad
D_k^v=\log(L_k^v+\epsilon)-\log(N_k^v+\epsilon).
\]

Primary representative 为 stage 内 median `vertical_straightness`；\(L^v,N^v,D^v\) 同时报告。

### V6. Raw activation entropy

对未中心化的 hidden state \(y_{k,l}\in\mathbb R^D\)，直接将坐标平方作为能量：

\[
e_j=y_{k,l,j}^2,
\qquad
p_j=\frac{e_j}{\sum_m e_m+\epsilon},
\]

\[
H_{raw}(y_{k,l})
=
-\frac{\sum_jp_j\log(p_j+\epsilon)}{\log D}.
\]

这里不减 token 内 coordinate mean，也不使用 base 校准，因此保留持续高激活坐标的影响。每个 stage 保存 across-layer mean、median、P90；primary representative 为 `median_raw_activation_entropy`。

### V7. Centered state, layer-difference, and robust-z entropy

对任意向量 \(z\in\mathbb R^D\)，先做向量内部 coordinate mean centering：

\[
\tilde z_j=z_j-\frac{1}{D}\sum_mz_m.
\]

\[
e_j=\tilde z_j^2,
\qquad
p_j=\frac{e_j}{\sum_me_m+\epsilon},
\]

\[
H_c(z)
=
-\frac{\sum_jp_j\log(p_j+\epsilon)}{\log D}.
\]

分别计算：

\[
H_{centered\_state}(k,l)=H_c(y_{k,l}),
\]

\[
H_{\Delta v}(k,l)=H_c(d_{k,l}^v).
\]

state entropy 用于区分“状态本身展开”与“当前层新写入展开”。

#### Robust z-score entropy

在 base checkpoint evaluation cohort 上，按 representation、layer、coordinate 计算 label-blind 的 \(\mu^{base}_{r,l,j}\) 与 \(\sigma^{base}_{r,l,j}\)，冻结并复用于所有 checkpoint：

\[
z'_{k,l,j}
=
\operatorname{clip}\left(
\frac{y_{k,l,j}-\mu^{base}_{r,l,j}}
{\max(\sigma^{base}_{r,l,j},\sigma_{floor})},
-8,8
\right).
\]

直接以 \((z'_{k,l,j})^2\) 构造概率并计算归一化能量熵，不再做 token 内 coordinate mean centering。必须报告 low-variance coordinate fraction。

本家族的 primary representative 为 stage 内 `median(layer_difference_entropy)`。`centered_state_entropy` 与 `robust_z_entropy` 均完整保存：前者是 V6 raw entropy 的中心化对照，后者是 rogue-dimension 敏感性控制；二者不替代 V6 或 layer-difference entropy 进入主检验。

### V8. Vertical effective rank

Layer-state matrix：

\[
Z_k^v=
\begin{bmatrix}
y_{k,0}\\y_{k,1}\\\vdots\\y_{k,L}
\end{bmatrix}.
\]

Centered layer-state matrix：

\[
\bar y_k=\frac1{L+1}\sum_{l=0}^{L}y_{k,l},
\qquad
Z_{k}^{v,c}=Z_k^v-\mathbf1\bar y_k^\top.
\]

Layer-update matrix：

\[
Z_{k,\Delta}^v=
\begin{bmatrix}
d_{k,1}^v\\d_{k,2}^v\\\vdots\\d_{k,L}^v
\end{bmatrix}.
\]

`layer_state_ER` 对 \(Z_k^{v,c}\) 计算，消除所有层共有的 residual offset。`layer_update_ER` 对未中心化 \(Z_{k,\Delta}^{v}\) 计算，仍是 primary representative：相邻层差分已消除 common state offset，而继续减平均 update 会使每层完全相同的有效写入退化为零矩阵。

另存 centered update sensitivity：

\[
\bar d_k=\frac1L\sum_{l=1}^{L}d_{k,l}^v,
\qquad
Z_{k,\Delta}^{v,c}=Z_{k,\Delta}^{v}-\mathbf1\bar d_k^\top,
\]

并报告 `layer_update_ER_centered`。Primary representative 为 stage 内 `median(layer_update_ER)`；centered `layer_state_ER`、`layer_state_ER_uncentered_sensitivity` 与 centered update ER 均为解释性输出。

---

## 8. 16 个预注册代表量

完整 scalar 全部保存，但多重检验的主 discovery family 只使用下列 16 个代表量。H1-H7 在四个冻结 `anchor_layer` 上分别形成预注册子检验；H8 只有最终层进入主 family。

| ID | Axis | Family | Representative | Depth scope | Stage mode |
|---|---|---|---|---|---|
| H1 | horizontal | movement | median relative movement | L3/L12/L24/Lfinal | local |
| H2 | horizontal | path | straightness | L3/L12/L24/Lfinal | local primary；cumulative sensitivity |
| H3 | horizontal | turning | median turn angle | L3/L12/L24/Lfinal | local |
| H4 | horizontal | angular velocity | P90 absolute angular velocity | L3/L12/L24/Lfinal | local |
| H5 | horizontal | centered state ER dynamics | centered ERV | L3/L12/L24/Lfinal | cumulative |
| H6 | horizontal | directional spectrum | directional ER | L3/L12/L24/Lfinal | cumulative |
| H7 | horizontal | length-angle coupling | weighted turning | L3/L12/L24/Lfinal | local |
| H8 | horizontal | token entropy dynamics | median token time-diff coordinate entropy | Lfinal primary | local |
| V1 | vertical | layer-update norm | P90 relative layer-update norm | all layers | local |
| V2 | vertical | raw state angle | median raw state angle | all layers | local |
| V3 | vertical | demean state angle | median state angle demean | all layers | local |
| V4 | vertical | layer-update turning | median layer-update turning angle | all layers | local |
| V5 | vertical | path | median vertical straightness | all layers | local |
| V6 | vertical | raw activation entropy | median raw activation entropy | all layers | local |
| V7 | vertical | centered/difference/robust entropy | median layer-difference entropy | all layers | local |
| V8 | vertical | vertical spectrum | median layer-update ER | all layers | local |

其他 mean/median/P90、raw/relative、state/update、ER/ERA、\(L/N/D\) 是解释性输出，不得替代代表量进入主统计后再选择最显著者。

---

## 9. 强制控制量

每条 rollout、checkpoint、stage 同时保存：

- `response_token_count`；
- `stage_token_count`；
- `trajectory_point_count`；
- `anchor_layer`（H1-H7 使用 `L3/L12/L24/Lfinal`；H8 primary 使用 `Lfinal`）；
- `valid_displacement/turn/omega_count`；
- `valid_vertical_update/state_angle/update_turn_count` 与 update-turn 排除率；
- `policy_entropy_mean` 与 stage-local policy entropy；
- `token_logprob_mean`；
- `token_time_diff_coordinate_entropy / token_state_coordinate_entropy / token_state_raw_energy_entropy` 的窗口与 stage coverage；
- `hidden_norm_mean`；
- `answer_reward / format_reward / is_correct`；
- `finish_reason / truncated`；
- MATH `level / subject`；
- GRPO checkpoint progress；
- representation；
- metric-specific coverage flag。

Policy entropy 是词表分布熵，与 coordinate energy entropy 必须使用不同字段名和图例，不得统称为 entropy。

---

## 10. 统计分析

### 10.1 Base-calibrated 标准化

对每个代表量、representation、`anchor_layer`（若适用）和 stage，只使用 base checkpoint non-truncated rollout 估计：

\[
zM_{sqi,b,l}
=
\frac{M_{sqi,b,l}-\mu^{base}_{M,r,b,l}}
{\sigma^{base}_{M,r,b,l}+\epsilon}.
\]

该 calibrator 冻结后用于所有 checkpoint，不按 checkpoint 重新归一化，否则会抹掉训练漂移。

### 10.2 主趋势模型

对 V1-V8、H8(final) 及每个 H1-H7 `metric x anchor_layer` 预注册子检验分别拟合：

\[
zM
\sim
training\_progress
+ stage
+ training\_progress\times stage
+ \log(1+response\_length)
+ policy\_entropy
+ correctness
+ level
+ (1|question).
\]

- `training_progress` 取 \(0,0.2,0.4,0.6,0.8,1.0\)。
- `stage` 是四水平 categorical variable。
- 对 H1-H7 额外拟合联合深度模型 `... + anchor_layer + training_progress x anchor_layer + stage x anchor_layer`，用于报告信号随网络深度形成的方式；四个 anchor 不得按效应大小筛选后单独宣告发现。
- question 作为随机截距；若 mixed model 数值失败，使用 question fixed effects 并按 question cluster-robust SE。
- 主要问题是 `training_progress x stage`，即不同 response 阶段的训练趋势是否不同。

### 10.3 非参数与可视化估计

- 每个 checkpoint x stage 计算 question-equal mean/median。
- 使用 4,000 次 question bootstrap 计算 95% CI；同一题的全部 rollout、representation 和 stage 一起重采样。
- 对每个 stage 报告 Spearman correlation(metric, training progress)。
- 不强制拟合单调关系；六个 checkpoint 的原始曲线必须完整展示。

### 10.4 Correct/wrong outcome 分析

这是 secondary analysis，不替代训练趋势：

- 每 checkpoint x stage 计算 correct-minus-wrong standardized gap；
- 在同题同时存在 correct/wrong rollout 时计算 within-question AUROC；
- 报告 question-equal AUC、pair-weighted AUC 和各自 question-bootstrap CI；
- 建立 `length + policy_entropy` baseline 与 `baseline + hidden metric` 模型，使用按 question 分组的 out-of-fold 预测比较增量 AUC。

### 10.5 多重检验

- 主 discovery family 包含 V1-V8、H8(final) 与 H1-H7 的四个冻结 anchor 子检验，共 \(8+1+7\times4=37\) 个 `training_progress x stage` omnibus p 值；使用 Benjamini-Hochberg FDR，阈值 `q <= 0.10`。
- correct/wrong outcome family 单独做 BH-FDR，不与训练趋势 family 混合。
- H1-H7 的 `mean` 是主规格，`last` 不单独用于宣告发现，只检验方向和曲线形状是否一致；H8 为 `representation=token`，不参加 mean/last 比较。
- 其余 scalar、单 cell、单 checkpoint 结果均为 descriptive exploratory，不进入主发现计数。

---

## 11. Process-reward 候选晋级标准

某指标只有同时满足以下条件，才从 discovery 晋级到独立多 seed replication：

1. **Non-terminal：** 在 B1、B2 或 B3 至少一个阶段出现，不只是 B4。
2. **Training relevance：** stage-specific training trend 或 `training_progress x stage` 在主 family 中通过 `q <= 0.10`。
3. **Control robustness：** 加入 response length 与 policy entropy 后，效应方向不翻转，且调整后 CI 不完全覆盖一个接近零的宽区间。
4. **Outcome consistency：** correct/wrong gap 与随训练改善的方向相容；若二者相反，标记为 policy diagnostic，不进入 reward shaping。
5. **Representation robustness：** H1-H7 的 mean primary 与 last sensitivity 的主要趋势方向一致；last 可以更弱，但不能稳定反向。H8 是 representation-independent。
6. **Depth reporting：** H1-H7 必须报告全部四个冻结 anchor；后续若冻结单层候选，必须明确其是 discovery localization，不能表述为全层普遍规律。
7. **Coverage：** 结论不是由少量超长 response 或少数有效 cell 驱动；按 trajectory-point count 与 response length 分层后方向稳定。
8. **Question robustness：** leave-one-question-out 与 question bootstrap 不由单题主导。

候选分类：

| Training trend | Outcome relation | 定位 |
|---|---|---|
| 无 | 无 | 淘汰 |
| 有 | 无 | policy dynamics diagnostic |
| 无 | 有 | static rollout-quality diagnostic |
| 有 | 有 | process-reward replication candidate |

本轮不直接使用如下 reward。只有候选通过后续独立验证，才考虑：

\[
r^{hidden}_{i,b}=Q_{i,b}-Q_{i,b-1},
\]

\[
A_{i,t}
=
A_i^{outcome}
+\lambda r^{hidden}_{i,b(t)}.
\]

---

## 12. 主要图表与表格

### 12.1 主图

1. **Training x response-stage heatmap：** 每个代表量一张 `checkpoint x B1-B4` base-standardized heatmap；H1-H7 按四个冻结 anchor 分面。
2. **Stage curves：** B1/B2/B3/B4 四条线随 checkpoint 的变化，question-bootstrap 95% CI；H1-H7 同图展示 L3/L12/L24/Lfinal 四条深度曲线。
3. **Correct/wrong stage curves：** 每 checkpoint 的 correct 与 wrong 曲线，以及 standardized gap。
4. **Two-axis summary：** 8 个 horizontal 与 8 个 vertical 代表量的 adjusted training-stage interaction effect heatmap；H1-H7 显示四 anchor 的完整小面板，H8 同时给出全层探索 profile。
5. **Control comparison：** raw trend、控制 length 后、再控制 policy entropy 后的系数并列图。
6. **Horizontal-vertical coupling：** 16 个代表量的 residual correlation matrix，先回归掉 checkpoint、stage、length、policy entropy；H1-H7 的四 anchor correlation 另作深度分面，不从中挑最大值作为结论。

### 12.2 诊断图

- response length、policy entropy、mean logprob、reward 随训练曲线；
- 每 checkpoint 的 response length 分布与 truncation rate；
- 每 metric x stage 的有效 question/rollout 数；
- mean 与 last 的效应方向比较；
- 指标按 MATH level/subject 分层；
- trajectory point count 与指标的关系；
- 每个 checkpoint 随机抽取固定 audit questions 的单 rollout spaghetti curves。

### 12.3 机器可读输出

计划输出到 `Experiment_2E/server_results/<run_id>/`：

```text
manifest/
  train_manifest.parquet
  eval_manifest.parquet
  checkpoint_manifest.json
  rollout_seed_manifest.parquet
  frozen_analysis_spec.json
rollouts/
  rollout_status.parquet
metrics/
  horizontal_metrics.parquet
  vertical_metrics.parquet
  controls.parquet
  base_calibrators.npz
analysis/
  primary_effects.csv
  outcome_effects.csv
  coverage.csv
  candidate_decisions.json
figures/
  primary/
  diagnostics/
REPORT.md
REPORT.html
AUDIT.json
```

所有 Parquet 至少包含：`run_id, checkpoint, question_id, rollout_id, representation, anchor_layer, stage, metric, value, coverage, is_correct, response_length, policy_entropy`。H8 的 `representation=token`，并在 `anchor_layer` 中记录实际层；H1-H7 只出现四个冻结 anchor。

---

## 13. 资源与存储

- 正式 GRPO 与 hidden extraction 分开运行；discovery 阶段不在训练 loss 内打开 `output_hidden_states`。
- 六个 checkpoint 的评估 rollout 总数固定为 12,288；同一 rollout 的 mean/last 与全部层指标来自一次离线 forward。
- hidden tensors 只驻留显存/内存直到标量归约完成；不永久保存全 token x layer x hidden-dimension 数据。
- 永久保存：标量 Parquet、coverage、base common/calibrator、固定少量 audit question 的 pooled vectors、模型与 tokenizer revision、完整 frozen config。
- H200 正式任务必须运行在命名 tmux 中，日志定期 flush，支持按 checkpoint 断点恢复。
- 上机前通过 hidden smoke 实测“每 100 条 rollout 的 forward 时间、峰值显存、标量输出大小”，再据此给出正式 ETA；计划阶段不根据 long-CoT 旧实验速度外推。
- mean/last 两种表示与四个 horizontal anchor 不会导致额外模型 forward；16 个代表量和 H8 全层 scalar 主要增加在线归约、SVD 与 Parquet 体积，不应按多倍 GPU forward 估算。hidden smoke 必须单独报告 token-level H8 的归约时间。

---

## 14. 数值与统计单元测试

正式上机前，指标实现必须通过以下 toy cases：

- 完全直线路径：\(L=N\)、straightness = 1、log-detour = 0、turn angle = 0。
- 走出再返回起点：\(L>0\)、\(N\approx0\)、straightness 接近 0、log-detour 为有限大值且无 Inf。
- 两个等长正交方向：directional ER = 2（数值容差内）。
- 所有 movement 同一方向：directional ER = 1。
- 单坐标能量：raw activation entropy、centered entropy 与 layer-difference entropy 均为 0；所有坐标等能量时 normalized entropy = 1。
- 连续两次相同的非零 layer update：layer-update turning angle = 0；两次正交 update：该角为 \(\pi/2\)（数值容差内）。
- 对全部 hidden state 加同一平移：horizontal displacement/path/turn 不变。
- 对全部 hidden state 乘正比例常数：angle、straightness、directional ER、normalized entropy 不变；raw movement/path 按比例变化。
- H5 centered ER：对每条 prefix 的全部轨迹点加同一向量，centered ER/ERV/ERA 不变；uncentered sensitivity 允许变化。
- V8：对 layer state 加同一向量不改变 centered `layer_state_ER`；完全相同的非零 layer updates 使 raw `layer_update_ER` 接近 1，而 centered update ER 按 coverage 规则记缺失。
- H8：含至少两个 token 的恒定 hidden trajectory，其 time-diff entropy 按零能量规则记 0；没有相邻 token 的 window 才记缺失。coordinate-centered state entropy 与未中心化 raw-energy entropy 在公共偏移存在时可不同；token scalar 不因 `mean/last` 重复写出。
- 零位移：角度记缺失并增加 exclusion count，不产生 NaN 传播到无关指标。
- 每种最小 coverage 边界：point count 恰好达到与少于门槛时状态正确。
- stage boundary：endpoint 恰好等于 25%、50%、75%、100% 时只进入一个冻结 stage。
- question bootstrap：同一道题的 rollout、stage、representation 始终作为整体重采样。
- label permutation：correct/wrong 效应在多次置换后以 0 为中心。

---

## 15. 执行阶段与检查点

### Task 1: 冻结数据与运行协议

**Deliverable:** train/eval manifest、seed manifest、reward audit、frozen config。

- [x] 从 MATH train/test 构造互斥 manifest，校验题目与 prompt hash 无交集。
- [x] 审计数学答案解析与等价判断的抽样准确率。
- [x] 冻结 GRPO hyperparameters、prompt template、decoding 和 seeds。
- [x] 运行 length-only smoke，按第 3.1 节规则冻结 stride。
- [x] 将最终决策写入 `frozen_analysis_spec.json` 并计算 SHA256。

### Task 2: GRPO smoke 与正式单 seed 训练

**Deliverable:** 可恢复的标准 GRPO run 与六个冻结 checkpoint。

- [x] 验证 mixed-reward groups、非零 advantage、KL 与 loss 数值稳定。
- [x] 验证 checkpoint 保存/恢复模型、optimizer、RNG 与 scheduler state。
- [ ] 启动正式 GRPO tmux，按冻结训练进度保存 checkpoint。
- [ ] 训练结束后校验六个 checkpoint 的权重、config、tokenizer 与日志完整性。

### Task 3: 固定 cohort rollout

**Deliverable:** 12,288 条 rollout、正确性标签与全部控制量。

- [ ] 对 base checkpoint 运行固定 256 x 8 rollout 并冻结输出 schema。
- [ ] 依次运行 20% 到 final checkpoints，不改变任何 decoding 参数。
- [ ] 逐 checkpoint 校验题数、rollout slot、seed、truncation 与 reward。
- [ ] 生成 `rollout_status.parquet` 和 generation audit。

### Task 4: Hidden extraction 与在线归约

**Deliverable:** horizontal/vertical scalar Parquet 与 base calibrators。

- [ ] 通过 2 x 2 x 2 hidden smoke 验证 response 边界和层数。
- [ ] 先处理 base checkpoint，冻结 label-blind common、按 `anchor_layer` 的 z-score stats 与 stage calibrators。
- [ ] 处理其余五个 checkpoint，禁止重新拟合 common/calibrator。
- [ ] 每批释放 full hidden tensors，校验无 OOM、NaN、Inf 或重复 rollout。
- [ ] 合并 metric-specific coverage，验证两种 representation 与四个 horizontal anchor 来自同一 forward，并验证 H8 token scalar 未按 representation 重复。

### Task 5: 主统计与可视化

**Deliverable:** 16 个代表量的主趋势、四层 horizontal anchor 分析、outcome secondary analysis 和完整诊断图。

- [ ] 校验 16 个 representative、37 个 primary discovery tests 与 frozen spec 完全一致。
- [ ] 计算 base-standardized checkpoint x stage 汇总与 4,000 次 question bootstrap。
- [ ] 拟合 training-progress x stage 主模型并做 BH-FDR。
- [ ] 运行 correct/wrong、within-question AUC 和 length/policy-entropy 增量分析。
- [ ] 输出主图、诊断图、coverage 和 representation sensitivity。

### Task 6: 候选决策与后续边界

**Deliverable:** discovery report、candidate cards 和多 seed replication 建议；不启动 reward shaping。

- [ ] 按第 11 节逐项审核每个代表量，不根据故事偏好放宽门槛。
- [ ] 将候选分类为淘汰、policy diagnostic、static outcome diagnostic 或 process-reward candidate。
- [ ] 对每个晋级候选冻结 stage、方向、aggregation、controls 和预期 replication sample size。
- [ ] 明确记录 single-seed 限制，并将任何 post-hoc 图降级为 appendix。
- [ ] 在用户审阅 discovery report 前，不编写或启动 hidden reward/advantage 实验。

---

## 16. 完成标准

本 discovery 实验只有在以下条件全部满足时才算完成：

- 六个冻结 checkpoint 与 12,288 条固定 cohort rollout 完整；
- 16 个代表量在四 stages 上均有 coverage 报告；H1-H7 在四个 anchor 和两种 representations 上报告，H8 的 final primary 与逐层探索 profile 分开报告；
- base common/calibrator 只由 base checkpoint、label-blind 数据构造并在后续冻结；
- 主趋势经过 length 与 policy entropy 控制并完成 BH-FDR；
- correct/wrong 分析同时报告 question-equal、pair-weighted 和 OOF length increment；
- 主图没有按单 cell 最大效应自动选图；
- 结论明确区分 training diagnostic、outcome diagnostic 与 process-reward candidate；
- 没有在本轮启动任何 hidden-state reward shaping；
- 报告包含机器可读 audit、全部 frozen decision 和 single-seed 限制。

---

## 17. 本地实现状态（2026-08-05）

已完成但尚未在 H200 上执行：

- formal run/checkpoint/roll8 seed manifest 的冻结与审计代码；
- 正式 GRPO、六 checkpoint roll8、2 题 × 2 rollout × 2 checkpoint hidden smoke、正式 hidden extraction 的 tmux 入口；
- H1-H8、V1-V8、base common/z-score calibrator、在线标量归约与 coverage 输出；
- base-standardized 趋势、question bootstrap、question-equal/pair-weighted AUC、按题 OOF length+policy-entropy 对照、BH-FDR、主图与 HTML 报告代码；
- 本地 synthetic/unit tests 与 shell syntax checks。

明确未执行：正式 manifest 冻结、正式 GRPO、12,288 条 rollout、任何真实模型 hidden forward。下一边界是连接 H200 后先审计 GPU/tmux/路径，再依次冻结 manifest、运行正式 GRPO、生成并审计 roll8 cohort，最后运行 hidden smoke；hidden smoke 通过前不得启动正式 hidden extraction。
