# ER Long-CoT 实验计划（Qwen3-VL Thinking）

本文档记录在 **long chain-of-thought** setting 下重测 ER.md 现有 hidden-state 轨迹指标的实验计划。与 ER.md 的区别只有一个：rollout 来自 **Qwen3-VL Thinking** 模型，response 显著更长，并且包含 rethinking（"wait / actually / let me reconsider"）这类反思转折。

---

## 0. 动机

ER.md 第 0 节的核心立论是：

```text
把整条 response 压成一个全局 scalar 的方法，在 long-CoT / multi-step reasoning 下会失效；
应该改成 chunk-level 的 semantic trajectory diagnostic。
```

但 ER.md 全部结果来自 **Instruct 模型、response <= 1536 token 的短 CoT**。在这个 setting 下：

```text
平均 num_chunks ~= 4
几乎没有 rethinking
path_length（全局 sum）= 0.71，是最强单指标
方向类指标弱（实验 D cos_mean=0.58，实验 G primary gcos_mean~=0.53）
```

也就是说，ER.md 赖以立论的"long-CoT 下全局标量失效"这个前提**从未被真正压力测试过**，因为数据根本不 long。本实验用 Thinking 模型生成 long response，第一次在真正的 long-CoT 上检验这个动机。

### 主 claim（本实验聚焦）

```text
(a) 随着 response 变长，全局聚合指标（path_length 这种 sum、mean ER）的
    within-question 判别力会衰减；而 late-phase / 动态 / position-calibrated
    指标相对更抗长度。
    => 直接验证 ER.md "全局标量在 long-CoT 下失效，需要 chunk-level 轨迹" 的动机。
```

### 次要探索（加层，不作第一版主 claim）

```text
(b) rethinking 几何：long-CoT 第一次提供大量真实"轨迹转折"。
    "wait/actually/reconsider" 在 hidden 空间表现为方向急转 + 大位移。
    探索：正确 rollout 是"健康 rethink 一次就收敛"，
          错误 rollout 是"反复 rethink、震荡不收敛"？
    如果成立，极端转向 / late 不稳定指标在 long-CoT 上会比短 CoT 强得多
    （解释短 CoT 上方向类指标为何一直弱：根本没有转折事件可测）。
```

---

## 1. 与 ER.md 的关系

```text
ER.md（短 CoT, Instruct）：基线
ER_long.md（long CoT, Thinking）：本实验

设计原则：controlled comparison
  - 同一批 MathVerse mixed MCQ 题
  - 同尺寸模型（8B），只换 Instruct -> Thinking
  - 指标套件原样搬运（A / D / E / G），不新造
  - 差异归因于 long-CoT + thinking，而非模型大小或题目难度
```

现有指标套件（全部重算）：

```text
实验 A：local ER（er_raw）+ ERV/ERA（erv_adj_late_min 等）
实验 E：path dynamics（path_length, d_late_mean, d_ratio_late_early, pv_late_max, d_hist_late_min）
实验 D：local chunk angular（cos_mean, cos_p10, spike_rate_90 等）— 已知弱，作对照
实验 G：cross-chunk angular（gcos_*, gstep_* 的粒度 sweep）— 重点看 rethinking 是否让它复活
实验 B：cross-rollout ER（可选，question-level）
```

---

## 2. 模型与数据

### 模型

```text
模型：Qwen3-VL Thinking（确认确切 repo 名，建议用 8B 与现有 Instruct 同尺寸）
精度：bfloat16
rollout 引擎：vLLM
hidden 抽取引擎：transformers（AutoModelForImageTextToText, output_hidden_states）
主层：layer 24, layer 36（与 ER.md 一致）
```

### Decoding

```text
按 Thinking 模型官方推荐采样配置设定（可能不是 temp=0.7），但必须固定下来并记录。
max_tokens：给足，建议先在 smoke 阶段量截断率再定（初值 8192，可能需要 16384）。
rollouts/question：8（与短 CoT 一致，保证 mixed-correctness 题数）。
```

### 数据

```text
题集：同一批 MathVerse mixed MCQ（与 ER.md 对照）
先 smoke：100-300 题 x 8 rollout，首要目的是量长度分布 + 截断率
再决定是否全量
```

---

## 3. Response 结构与分段（核心新增）

Thinking 模型输出带 `<think>...</think>` 推理段 + 最终答案段。必须在 **token 层面**（不是字符串匹配）把 response 切成结构化三段：

```text
think 段   ：<think> 与 </think> 之间的 token —— 主分析对象，所有轨迹指标算在这里
answer 段  ：</think> 之后的 token —— 通常很短，单独标记，只用来定对错，不算轨迹指标
（可选）rethinking 标记：think 段内识别反思转折点（见第 6 节）
```

分段原则：

```text
1. 用 chat template 的 token id 定位 <think> / </think>，不要用字符串切分
   （tokenization 会把标记拆开，字符串匹配不可靠）。
2. 轨迹指标（path / ER / angular）只在 think 段 token 上计算。
3. 若 think 段为空或过短（不足 2 chunk / 不足 micro-window），标记并单列，不混入主结果。
4. answer 段长度单独记录（answer_length），用于 overthinking 诊断。
```

为什么主用 think 段：

```text
think 段才是"推理轨迹"，ER.md 整个 process-diagnosis 故事讲的就是推理过程。
answer 段太短，且是结论而非过程，几何意义弱。
```

---

## 4. 截断处理（long-CoT 头号陷阱）

long response 容易撞 max_tokens 被截断。这是 long-CoT 评测最大的混淆源：

```text
被截断 -> 答案没生成完 -> 判错
但它的"未收敛轨迹"是被砍出来的，不是推理质量差。
如果不处理，path_length / late 指标的"区分力"会有一大半其实是
"截断 vs 未截断"，而不是"对 vs 错"。
```

处理规则：

```text
1. 每条 rollout 记录 finish_reason / truncated 标志（vLLM 的 stop reason）。
   truncated = (finish_reason == "length") 或 response 末尾无 </think> 闭合 / 无 answer 段。
2. 主分析：排除 truncated rollout（clean subset）。
3. 单列报告：截断率（按长度桶、按对错），以及 truncated rollout 的指标分布。
4. 敏感性分析：把 truncated 当作单独一类，验证主 claim 在 clean subset 与
   full set 上方向一致（排除"结论全靠截断"）。
```

必报字段：

```text
truncation_rate_overall
truncation_rate_by_length_bin
truncation_rate_correct vs incorrect
```

---

## 5. Chunk 划分（long setting 下重新设计）

短 CoT 下 num_chunks ~= 4；long CoT 下 think 段可能 20-100+ chunk。这对不同指标影响相反，必须区别对待：

```text
chunk_size：保持 256 token（便于和短 CoT 直接对照）
            但 think 段长度作为自变量，num_chunks 从 ~4 涨到几十正是本实验看点。
```

### 5.1 绝对位置 vs 相对位置

```text
绝对 chunk index：长短 response 不可比，long 下后期 chunk 只有超长 rollout 才到达
                  （selection bias 比短 CoT 严重得多）。
相对位置（relative_pos = chunk_start / think_length）：long 版主对齐方式，刚需。

所有 "late phase" 定义改用相对位置：
  late = 相对位置后 25%（不是固定 chunk 3）
  early = 相对位置前 25%
```

### 5.2 Calibration 按 long 分布重估

```text
position-conditioned calibration（zER, zPAD 等）的 mean_k / std_k
必须在 long-CoT 数据上、按相对位置桶重新统计，
不能复用短 CoT 的 calibration 基线。
```

### 5.3 指标对长度的预期反应（这就是主 claim 要测的）

```text
path_length = sum(d_k)        ：随 think 长度直接累加 -> 预期判别力衰减（主假设）
d_mean = path_length / n      ：长度归一化 -> 预期更抗长度
d_late_mean（相对后 25%）      ：位置归一化 -> 预期更抗长度
erv_adj_late_min              ：动态 + late -> 预期更抗长度
mean ER（全局平均）            ：全局聚合 -> 预期判别力衰减
```

---

## 6. Rethinking 标记层（次要探索 b）

long-CoT 第一次提供大量真实轨迹转折。本层把"文本层反思"对到"hidden 层几何转折"，验证角度 spike 是否就是 rethinking 的几何足迹。

### 6.1 文本层识别 rethinking 转折点

```text
在 think 段 token 序列中标记反思触发词的位置（token offset）：
  "wait", "actually", "hmm", "let me reconsider", "but", "on second thought",
  "alternatively", "re-examine", "I made a mistake" 等（建立一个 trigger 词表）。
记录：
  rethink_count        ：触发次数
  rethink_positions    ：每次触发的 token offset / 相对位置
  rethink_density      ：rethink_count / think_length
```

### 6.2 几何层对齐

```text
对每个 rethinking 触发点，定位它落在哪个 chunk / 哪个中等粒度点，
检查该位置附近的 hidden 轨迹是否出现：
  - 角度 spike（gcos_min 局部下降 / theta 局部上升）
  - 位移 spike（d_k / gstep 局部上升）
对齐方式：把 rethink 位置映射到最近的 angular / path 点序列 index。
```

### 6.3 探索性问题

```text
1. rethink 触发点处，hidden 轨迹的角度/位移是否显著高于非触发点（事件对齐分析）？
2. rethink_count / rethink_density 本身能否区分对错？
   假设：适度 rethink（一次性纠错）-> 正确；过度 rethink（反复震荡）-> 错误。
   预期非单调：太少（没检查）和太多（纠结）都可能偏错。
3. 在 rethink 多的 rollout 子集上，方向类指标（gcos_min, cos_p10, spike_rate）
   的 AUROC 是否明显高于 rethink 少的子集？
   => 验证"方向信号需要转折事件才有用武之地"。
```

### 6.4 注意

```text
- trigger 词表是启发式，会有漏报/误报；rethinking 标记仅作探索，不进第一版主 claim。
- 多语言/公式场景触发词可能失效，需人工抽查若干样本校准词表。
- 文本触发 != 一定有几何转折；事件对齐分析本身就是要检验这个对应关系。
```

---

<LONG_PLACEHOLDER>

## 7. 指标重算清单

全部在 think 段、clean subset（非截断）上计算，layer 24 + 36。

```text
实验 A（local ER）：
  er_raw（per chunk）
  erv_adj_late_min, erv_hist_late_min, era_adj_late_min（late 用相对位置定义）

实验 E（path dynamics）：
  path_length, d_mean, d_late_mean, d_ratio_late_early, pv_late_max, d_hist_late_min

实验 D（local chunk angular，对照）：
  cos_mean, cos_p10, spike_rate_90

实验 G（cross-chunk angular，重点）：
  粒度 sweep w in {32, 64, 128}，pool in {mean, last}
  gcos_mean, gcos_p10, gcos_min, gcos_late_mean, gstep_mean, gstep_late_mean
  重点看：rethinking 是否让 gcos_min / gstep 在 long 上比短 CoT 强

实验 B（cross-rollout ER，可选 question-level）：
  ER_group early/mid/late vs question accuracy
```

数据存储遵循 ER.md 8.6：只存 chunk-level / 中等粒度池化点的标量指标，**不存全 token hidden**（long response forward 显存与磁盘成本高）。

---

## 8. 核心分析：把长度当自变量

这是本实验区别于"重刷一遍 AUROC"的关键。

### 8.1 长度分桶判别力曲线（主 claim 的核心图）

```text
按 think 段长度分桶：<2k / 2-4k / 4-8k / 8k+ token
在每个长度桶内，分别算 within-question AUROC：
  - path_length（全局 sum）
  - d_late_mean（相对 late）
  - erv_adj_late_min（动态）
  - mean ER（全局平均）

观察：随长度增加，
  全局聚合指标（path_length, mean ER）AUROC 是否衰减？
  late / 动态指标是否保持？
```

```text
positive（支持 ER.md 动机）：
  path_length 在长桶里 AUROC 下降，d_late_mean / erv 保持 -> 全局标量失效，局部/动态更鲁棒
negative（动机需重写）：
  path_length 在 long 上照样 ~0.71 -> "全局标量失效" 不成立，ER.md motivation 要改
```

### 8.2 跨模型对照（短 CoT vs long CoT）

只比 AUROC（尺度不变），**不比 raw 数值**（Thinking 与 Instruct 的 hidden norm 不同，raw path_length 不可比）。

| 指标 | 短 CoT (Instruct) | long CoT (Thinking) | 解读 |
|---|---:|---:|---|
| path_length | 0.71 | ? | 衰减了吗 |
| d_late_mean | 0.72 | ? | |
| erv_adj_late_min | 0.69 | ? | |
| cos_mean (D) | 0.58 | ? | rethinking 是否让方向信号变强 |
| gcos_min (G) | ~0.62 | ? | 同上，重点 |
| 平均 num_chunks | ~4 | ~20-100 | |
| 截断率 | ~0 | ? | 必报 |
| 平均 think 长度 | n/a | ? | |

### 8.3 控制变量

```text
同一批题、同尺寸模型、同 chunk_size、同 N=8 rollout。
clean subset（排截断）做主分析，full set 做敏感性对照。
overthinking 噪声：MCQ 下 Thinking 可能纠结后蒙对，
  可用多 rollout 一致性筛高置信样本做 robustness。
mixed-correctness 可用 n：long CoT 正确率可能升高 -> mixed 题变少 ->
  within-question AUROC 可用题数下降，需在 smoke 阶段确认够用。
```

---

## 9. 主要图表

```text
图 L1：截断率报告（overall / by length bin / correct vs incorrect）
图 L2：think 段长度分布（vs 短 CoT 长度分布对照）
图 L3：长度分桶判别力曲线（主图）—— path_length / d_late_mean / erv / mean ER 的 AUROC vs 长度桶
图 L4：跨模型对照表可视化（短 CoT vs long CoT AUROC）
图 L5：Correct vs Incorrect 的 d_k / gcos 曲线（按相对位置对齐）
图 L6（探索）：rethink 事件对齐——触发点 vs 非触发点的角度/位移分布
图 L7（探索）：rethink_density vs correctness（看非单调）
图 L8（探索）：rethink 多/少子集上方向类指标 AUROC 对比
```

---

## 10. 执行顺序

```text
Step 1：Thinking smoke（100 题 x 8 rollout）
  首要目的：量 think 段长度分布 + 截断率 + mixed-correctness 题数
  据此定 max_tokens、是否够分长度桶、是否够 within-question AUROC

Step 2：确认 <think>/</think> token 边界能干净切分（人工抽查若干样本）

Step 3：clean subset 上跑现有 path/ER 套件
  先看两件事：AUROC 是否还在量级 + 长度分桶衰减曲线（主 claim）

Step 4：据 Step 3 决定是否全量

Step 5（探索）：加 rethinking 标记层，做事件对齐 + density 分析

Step 6：写跨模型对照表，回填 ER.md motivation 是否被支持
```

---

## 11. 可能结论

### 主 claim (a) — 理想 positive

```text
As long-CoT response length increases, the within-question AUROC of global
aggregate metrics (path_length sum, mean ER) decays, while position-relative
late-phase and dynamic metrics (d_late_mean, erv_adj_late_min) remain stable.
This directly supports the ER.md premise that global scalar summaries fail
under long-CoT reasoning and chunk-level trajectory diagnostics are needed.
```

### 主 claim (a) — negative（同样有价值）

```text
Global path_length stays around 0.71 even on long-CoT.
Then "global scalar fails on long CoT" is not supported, and the ER.md
motivation must be reframed: the sum is a robust summary even at long length.
```

### 探索 (b) — rethinking 几何

```text
理想：rethink trigger 点对齐显著的 hidden 角度/位移 spike；
      在 rethink 多的 rollout 上方向类指标（gcos_min, cos_p10, spike_rate）
      AUROC 明显高于短 CoT，解释了短 CoT 上方向信号为何一直弱
      （无转折事件可测）。
      rethink_density 与 correctness 呈非单调（过度反思 -> 错）。
negative：rethink 文本触发与几何转折不对齐，或 rethink 多寡不区分对错；
          则 long-CoT 的方向信号仍弱，幅度（path/displacement）仍是主信号。
```

### 截断相关（必报）

```text
若主 claim 在 clean subset 与 full set 上方向不一致，
说明结论部分由截断驱动，需以 clean subset 为准并显式说明。
```

