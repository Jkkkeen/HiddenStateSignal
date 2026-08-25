# Qwen3-1.7B-Base DeepMath GRPO Hidden-Dynamics Discovery Plan

**目标：**在单张 H200 上对 `Qwen/Qwen3-1.7B-Base` 进行一轮可审计的全参数 GRPO。先用冻结 held-out 测试集判断数学能力是否提升，再分析 H1-H9、V1-V9 与 V4 weighted companion 是否随有效训练形成稳定的横向/纵向 hidden-state 动力学变化。

**实验性质：**单 seed discovery。训练 reward 仅作为 telemetry；主要能力结论来自 held-out `pass@1`。hidden 曲线不能单独证明因果机制或 reward-shaping 有效。

**面板规则：**完全复用 `Experiment_2E/cut_logic.md`。训练期间每一步追加 correct/wrong 曲线点，并异步更新同一批约 141 个固定 PNG key；每 25 steps 计算一次固定测试集 AUROC。不得根据中途图形修改指标公式、layer、stage、reducer、数据难度或 checkpoint。

---

## 1. 研究问题与结论边界

本实验依次回答：

1. `Qwen3-1.7B-Base + GRPO` 是否提升未参与训练的数学题表现；
2. 若 held-out 能力改善，H1-H9、V1-V9 与 V4C 是否随训练出现可复现的阶段性变化；
3. correct/wrong rollout 的 hidden 差异是否随着训练扩大、缩小或发生阶段转移；
4. hidden 变化是否仍能在控制 response length、policy entropy、trajectory-point count、难度和 correctness 后成立；
5. 横向动力学与纵向动力学是否呈现不同的训练时序，而不是同一个长度或尺度效应的重复表达。

允许的结论：

- held-out 能力改善时，可表述为“伴随有效 GRPO 的内部动力学变化”；
- held-out 能力未改善时，只能表述为 policy/optimization drift diagnostic；
- 单 seed、单模型、单算法不能证明普适机制；
- 本轮不启用 hidden reward shaping，不把任何 H/V 指标加入 reward。

---

## 2. 冻结模型、Prompt 与 verifier

### 2.1 模型

- 模型：`Qwen/Qwen3-1.7B-Base`；
- 不是后训练版 `Qwen/Qwen3-1.7B`；
- 不设置或宣称 `enable_thinking=True`；
- 全参数训练，不使用 LoRA；
- 保存 model revision、tokenizer revision、generation config、权重 SHA256 和 tokenizer SHA256；
- 记录 Qwen3 decoder 层数，并据此解析 `Lfinal`；横向主 anchor 固定为 `L3/L12/Lfinal`。

### 2.2 Prompt

Base 模型使用冻结的数学 prompt，必须明确要求：

1. 展示逐步推理过程；
2. 最终答案写入 `\boxed{}`；
3. 最终答案后不再继续生成；
4. 不依赖 `<think>...</think>` 作为 response 边界。

hidden 分析中的 response 指完整生成 response token。B1-B4 按 response token 的相对进度切分，不按 `<think>` 标签切分。

### 2.3 Reward 与 verifier

主 reward 为答案正确性二值量：

\[
r_{q,i}\in\{0,1\}.
\]

- 数学等价性由冻结 rule-based verifier 判断；
- boxed rate、parse rate、format correctness 单独记录；
- format 不默认加入 reward；
- calibration 结束后不得根据训练曲线改 verifier 或 reward；
- calibration 的 parse rate 低于 `0.90` 或 boxed rate 低于 `0.70` 时，不启动正式训练，先判定 Prompt/verifier 协议未通过。

---

## 3. 数据方案与冻结规则

### 3.1 两个候选数据源

| 数据 | 角色 | 初始范围 |
|---|---|---|
| DeepMath-103K | 主训练集候选 | Level 3-6 |
| SimpleRL-Zoo | 稳定基线/备用候选 | MATH Level 3-5 |

本轮优先验证 DeepMath Level 3-6。SimpleRL-Zoo Level 3-5 用于判断 Base 模型在成熟 zero-RL 难度带上的基础可训练性。两个数据源不能混成一个训练集，也不能根据 hidden 结果临时切换。

### 3.2 Calibration 样本

分别抽取：

```text
DeepMath Level 3-6: 256 questions x roll8
SimpleRL Level 3-5: 256 questions x roll8
```

calibration：

- 不更新模型参数；
- 使用相同 Prompt、verifier、sampling 和 tokenizer；
- 先用 `max_response_length=12288` 生成；
- 同时模拟 `4096/8192/12288` 三个候选 cap；
- 所有 calibration 题永久排除出正式训练和正式测试。

每个候选集记录：

- empirical `pass@1`；
- 每题 `k_correct in [0,8]`；
- all-wrong、all-correct、mixed 和 useful-mixed；
- boxed/parse/format rate；
- response token mean/P50/P90/P95；
- natural stop 与各 cap 下的 truncation；
- difficulty/topic 分层；
- verifier failure 类型。

定义：

\[
\text{mixed}=\mathbf 1(1\le k_{correct}\le7),
\]

\[
\text{useful-mixed}=\mathbf 1(2\le k_{correct}\le6).
\]

### 3.3 数据集 GO/NO-GO

候选训练集必须同时满足：

\[
0.15\le \operatorname{pass@1}\le0.50,
\]

\[
\text{mixed fraction}\ge0.35,
\]

\[
\text{useful-mixed fraction}\ge0.20,
\]

\[
\text{non-format truncation}\le0.05.
\]

选择顺序：

1. DeepMath Level 3-6 通过时，作为本轮主训练集；
2. DeepMath 未通过、SimpleRL 通过时，停止 DeepMath 主实验并为 SimpleRL 建立单独冻结 manifest；
3. 两者都不通过时，本轮正式 GRPO NO-GO；
4. 不能使用训练 reward 或 hidden 曲线事后修改难度范围。

SimpleRL 数据规模与 DeepMath 不同。如果切换到 SimpleRL，不能机械沿用 `9,600 accepted groups`；须以官方去重题池能够支持的无重复训练规模重新冻结。本文后续的 `9,600 accepted groups` 仅对应 DeepMath 主路线。

### 3.4 Response cap

在 `4096/8192/12288` 中选择满足以下要求的最小 cap：

- non-format truncation `<=5%`；
- 相对 `12288` 的正确率下降不超出 question-bootstrap 不确定性；
- boxed/parse rate 无显著下降；
- 不造成明显的终局答案被截断。

预期冻结 `8192`，但正式值只由 calibration 决定。训练、在线 hidden、checkpoint 测试和最终评估共享同一 cap。

---

## 4. 数据划分与防泄漏

DeepMath 主路线固定：

```text
calibration: 256 questions
held-out primary test: 256 questions
training candidate pool: 其余满足冻结难度带的题
```

规则：

- 标准化题面后计算 question hash；
- calibration、training、held-out 三者交集必须为 0；
- held-out 按 difficulty/topic 分层；
- 同源近重复题需在切分前聚类，整个近重复簇只能进入一个 split；
- held-out 永久不进入 group-filter 补采池；
- 保存完整 manifest、source row id、difficulty、topic、answer hash 和 split hash；
- MATH500 只在 base/final 做外部行为 sanity check，不参与 checkpoint 选择或 hidden 主分析。

固定分析单位：

- 训练单位：question + 完整 roll8 group；
- 行为统计单位：question；
- correct/wrong hidden 比较：先题内，再 question-equal 汇总；
- bootstrap：以 question 为重采样单位。

---

## 5. 正式 GRPO 配置

### 5.1 硬件与训练方式

| 项目 | 冻结值/规则 |
|---|---:|
| GPU | `1 x H200` |
| 模型 | `Qwen/Qwen3-1.7B-Base` |
| 训练 | 全参数 GRPO |
| 框架 | 当前冻结 veRL 环境 |
| distributed | 单 rank；当前栈要求时使用 world-size 1 FSDP2 |
| rollout engine | vLLM，单卡 hybrid/serial 调度 |
| CPU offload | 默认关闭，仅 smoke 证明 OOM 后统一开启 |
| gradient checkpointing | 开启 |
| remove padding | 开启 |
| dynamic token batching | 开启 |

### 5.2 Batch 与优化

| 项目 | 冻结值 |
|---|---:|
| outer steps | `300` |
| accepted groups | `9,600` |
| `train_batch_size` | `32 prompts/groups` |
| `rollout.n` | `8` |
| responses/outer step | `256` |
| `ppo_mini_batch_size` | `8 prompts = 64 responses` |
| optimizer mini-batches/step | `4` |
| `ppo_epochs` | `1` |
| micro-batch | 每次 1 条长 response |
| actor learning rate | `1e-6` |
| optimizer | AdamW，`beta=(0.9,0.95)` |
| weight decay | `0.01` |
| warmup | 前 `5%`，约 15 steps |
| scheduler | warmup 后 cosine decay |
| clip ratio | `0.2` |
| KL loss | `use_kl_loss=True`，`coef=1e-4`，`low_var_kl` |
| KL in reward | `False` |
| entropy coefficient | `0` |
| loss aggregation | `token-mean` |
| max prompt length | `2048` |
| max response length | Phase A 冻结 |
| sampling | temperature `0.8`，top-p `0.95`，top-k `20` |
| rollout tensor parallel | `1` |
| GPU memory utilization | smoke 从 `0.55-0.60` 冻结 |

运行时必须验证 `ppo_mini_batch_size` 的实际口径是 prompt 还是 response，日志应明确显示：

\[
32\ \text{prompts}\times8=256\ \text{responses/step},
\]

\[
8\ \text{prompts}\times8=64\ \text{responses/mini-batch},
\]

\[
256/64=4\ \text{optimizer mini-batches/step}.
\]

### 5.3 Advantage 与 group filtering

对 question `q` 的完整 roll8：

\[
A_{q,i}=\frac{r_{q,i}-\bar r_q}{s_q+\epsilon}.
\]

动态 group filtering 保留：

\[
1\le k_{correct}\le7.
\]

规则：

- 保留 group 时保留完整 8 条 rollout；
- rejected all-wrong/all-correct group 不进入梯度；
- 使用新的 prompt 补足 32 个 accepted groups；
- 不对同一题反复生成直到 mixed；
- 记录 raw prompts、accepted prompts、all-wrong、all-correct、acceptance rate 和 refill rounds；
- `2<=k<=6` 只作为 useful-mixed 诊断；
- rolling 20-step acceptance rate 低于 `0.25` 时暂停并审计，不静默改难度或过滤规则。

由于过滤会放大实际生成量，必须同时报告 accepted groups 和 raw generated groups。

---

## 6. Checkpoint 与固定测试

### 6.1 Checkpoint

保存：

```text
0, 25, 50, 75, 100, 125, 150,
175, 200, 225, 250, 275, 300
```

共 13 个 checkpoint。每个 checkpoint 保存：

- actor weights；
- optimizer 与 scheduler；
- global step；
- RNG state；
- resolved config；
- code commit；
- model/data manifest hash；
- online hidden schema version。

`step 300` 为预先指定的主报告点。中间 checkpoint 只用于趋势分析，不得根据训练 reward 挑选“最佳模型”作为主结果。

### 6.2 每 25 steps 固定测试

每个 checkpoint 对同一批 256 道 held-out 题生成固定 roll8：

\[
13\times256\times8=26,624\ \text{test rollouts}.
\]

固定：

- question order；
- rollout-slot seed；
- sampling config；
- response cap；
- verifier；
- prompt template。

每个 checkpoint 计算：

- empirical `pass@1`；
- unbiased `pass@4/pass@8`；
- answer accuracy/reward；
- boxed/parse/format rate；
- response length mean/P50/P90；
- truncation；
- policy/token entropy；
- all-wrong/all-correct/mixed/useful-mixed；
- H1-H9、V1-V9、V4C 的 held-out within-question AUROC 与 coverage。

若一题 roll8 中有 `c` 条正确：

\[
\operatorname{pass@k}
=1-\frac{\binom{8-c}{k}}{\binom{8}{k}},
\qquad k\in\{1,4,8\}.
\]

单张 H200 上测试与训练串行执行。checkpoint 先落盘，测试失败时可从该 checkpoint 恢复，不允许测试进程破坏 optimizer state。

SwanLab 在线显示 point estimate；4,000 次 question-bootstrap CI 在该 checkpoint 测试完成后计算或在最终离线分析统一计算。

---

## 7. 每步在线 Hidden-State 计算

### 7.1 核心实现边界

主方案必须复用 actor training forward 中已经产生的 layer activations，不在每个 outer step 后额外重跑一次完整 hidden forward。

数据流：

```text
actor training forward
-> layer hook / activation capture
-> response-only chunk pooling
-> rollout-level H/V scalar
-> discard pooled vectors
-> question-equal correct/wrong/AUROC
-> append local scalar history
-> submit latest-only plotting task
-> overwrite current 141 local PNGs
-> asynchronously upload fixed 141 SwanLab image keys
```

scalar 历史仍然必须保存到本地 Parquet/CSV，供统计分析和重新出图使用；SwanLab hidden 面板只展示固定 image key 的最新 PNG。PNG 生成和上传不得在训练主线程中同步阻塞。

禁止永久保存完整：

```text
response x token x layer x hidden_dim
```

允许每个 micro-batch 临时保留：

```text
response x chunk x selected/all-layer x hidden_dim
```

处理完当前 micro-batch 后，只保留 rollout-level scalar、coverage 和 exclusion count。

### 7.2 在线计算安全要求

- hook 输出必须从 autograd graph `detach`，不能改变训练梯度；
- 不能逐层频繁同步 GPU 到 CPU；先在 GPU 做 pooling，再批量转移小矩阵或 scalar；
- gradient checkpointing 的 backward recomputation 不能重复累计指标；
- remove-padding 后必须恢复每条 response 的 token 边界；
- reward/correctness label 与 rollout id 必须一一对应；
- 任一在线 probe 异常不得静默写入 0，必须写为 missing 并计入 coverage；
- hidden probe 失败不能改变训练 batch、reward 或 optimizer update。

### 7.3 在线资源门槛

先在永久排除的 smoke run 中比较无 probe 与有 probe 的 5 个稳定 steps：

```text
step_time_without_probe
step_time_with_probe
probe_gpu_ms
probe_cpu_ms
peak_gpu_memory
online_metric_coverage
```

冻结规则：

- 单步增幅 `<=30%`：每步计算完整 32 accepted groups；
- 单步增幅 `(30%,50%]`：每步按冻结 hash 顺序选 16 个 mixed groups；
- 单步增幅 `>50%`：正式训练改为每 5 steps 计算一次，不能为了保留每步曲线而牺牲训练稳定性；
- 任一方案都必须记录实际 probed question/rollout 数。

### 7.4 指标与表示

指标公式直接继承冻结的 `experiment_2e/metrics.py` 与 `experiment_2e/reduction.py` 语义，本计划不修改算法定义。

横向：

| ID | primary |
|---|---|
| H1 | movement |
| H2 | straightness/path |
| H3 | turning angle |
| H4 | angular velocity |
| H5 | centered ER dynamics |
| H6 | directional ER |
| H7 | weighted turning |
| H8 | token time-difference coordinate entropy |
| H9 | trajectory polynomial complexity/ED |

纵向：

| ID | primary |
|---|---|
| V1 | relative layer-update norm |
| V2 | raw state angle |
| V3 | demean state angle |
| V4 | layer-update turning |
| V5 | vertical straightness/path |
| V6 | raw activation entropy |
| V7 | layer-difference entropy |
| V8 | layer-update effective rank |
| V9 | layer-update trajectory ED |

Companion：

```text
V4C = weighted_layer_update_turning_angle
```

统一：

- response window `128`；
- stride 由当前冻结 metric schema 继承，正式训练前写入 resolved config；
- chunk representation：`chunkMean/chunkLast`；
- horizontal anchors：`L3/L12/Lfinal`；
- vertical：`allLayers`；
- stages：B1 `0-25%`、B2 `25-50%`、B3 `50-75%`、B4 `75-100%`；
- stage reducer/native 严格使用 `cut_logic.md`。

---

## 8. SwanLab 面板与数据来源

### 8.1 图像数量

完整 hidden 面板按 `cut_logic.md` 去重后为约 141 张图：

\[
93\ \text{horizontal}+48\ \text{vertical}=141.
\]

checkpoint 数和训练 step 数不再与 141 相乘。面板固定使用约 141 个永久 image key；每个 key 对应一张包含从 step 0 到当前 step 完整历史的最新 PNG。不能为每一步创建新 key，也不保存逐 step PNG 快照。

固定 key 示例：

```text
hidden/H1_movement__chunkMean_L12_stageMedian
hidden/H5_centeredERV__chunkLast_Lfinal_stageNative
hidden/V4_layerUpdateTurning__chunkMean_allLayers_stageMedian
hidden/V4C_weightedTurning__chunkMean_allLayers_stageMedian
```

key 必须使用稳定 ASCII 命名并完整包含 metric、representation、layer/allLayers 和 reducer。step 变化只能更新既有 key，不能创建带 step 编号的新 key；只有 driver/rank 0 注册和上传 image key。

SwanLab 页面默认显示每个 key 的最新图片；服务端可能保留同一 key 的历史媒体版本，但不会增加面板卡片数量。

### 8.2 每张图

每张 hidden 图固定四个子图 B1-B4。为避免混淆，图例明确标注两种来源：

- 蓝线：当前训练 step 的 online question-equal correct mean；
- 红线：当前训练 step 的 online question-equal wrong mean；
- 灰色虚线：当前训练 step 的 online within-question AUROC；
- 黑色 checkpoint marker：每 25 steps 的 held-out within-question AUROC。

黑色 held-out marker 与蓝/红在线训练曲线来自不同 question cohort，不能把它解释为蓝红差异的直接函数。held-out correct/wrong mean 另存表格，不额外复制 141 张图。

每个图同时显示或关联：

- probed questions；
- correct/wrong rollout 数；
- mixed question 数；
- coverage；
- exclusion rate。

### 8.3 PNG 绘图与上传队列

每个训练 step 完成 question-equal 聚合后：

1. 将当前 step 的 scalar 追加到本地历史数组；
2. 向独立绘图队列提交一个“最新状态”任务；
3. 绘图进程根据最新历史覆盖本地当前 141 张 PNG；
4. 由 driver/rank 0 异步上传到固定的 141 个 SwanLab image key。

队列必须合并过期任务：如果 step 100 的图尚未完成而 step 101、102 已到达，只保留 step 102 的待绘状态，不依次渲染过期图片。训练主线程不等待 PNG 渲染或上传；上传失败只记录 telemetry，保留本地最新 PNG，下一次只上传最新状态。

本地只保留当前 141 张临时 PNG 和 scalar 历史，不保存每个 step 的 PNG 副本。每个 key 的最后成功上传 step、失败次数以及 `plot_render_ms`、`plot_upload_ms` 必须写入 `system/` telemetry。

PNG 队列过慢时只允许降低图片渲染/上传频率或合并任务，不能改变已由第 7.3 节冻结的 hidden probe 频率、样本选择或指标计算结果。

### 8.4 训练 telemetry

每一步至少上传：

```text
train/reward/*
train/groups/*
train/optimization/*
train/generation/*
hidden_online/*
system/*
```

其中 `hidden_online/*` 只包含 `probed_n`、`mixed_n`、`correct_n`、`wrong_n`、coverage、exclusion 和 probe/plot timing 等诊断 scalar，不把 H1-H9、V1-V9、V4C 的全部 metric value 再注册成 SwanLab 原生曲线。H/V metric value 保存到本地历史并进入固定 141 张 PNG。

包括：

- reward/score mean、min、max；
- `k_correct` 分布；
- all-wrong/all-correct/mixed/useful-mixed；
- filter acceptance 与 refill rounds；
- zero-advantage fraction；
- KL、policy entropy、grad norm、clip fraction、learning rate；
- response length、truncation、tokens/s、step time、GPU memory；
- hidden probe GPU/CPU 时间与 coverage。

SwanLab 只保存图表与标量。永久原始结果写入 Parquet/CSV/JSON，不上传或保存完整 hidden tensor。

---

## 9. 统计分析

### 9.1 行为主判据

预先指定：

\[
\Delta pass@1
=pass@1_{step300}-pass@1_{base}.
\]

主成功条件：

- point estimate 为正；
- question-bootstrap 95% CI 下界大于 0。

`pass@4/pass@8`、中间 checkpoint 最佳值、长度和格式为辅助结果。不能用中间最优 checkpoint 替代 step 300 主判据。

### 9.2 Hidden 趋势

online 曲线用于 discovery，至少报告：

- training progress 主效应；
- correctness 主效应；
- training progress x correctness 交互；
- stage 交互；
- correct/wrong 分离随训练的趋势；
- held-out AUROC 的 checkpoint 轨迹。

控制项至少包括：

- `log1p(response_length)`；
- policy/token entropy；
- trajectory point/chunk count；
- difficulty；
- correctness；
- question random intercept 或 question-grouped inference。

多重比较：

- H1-H9/V1-V9 primary family 使用 BH-FDR；
- V4C 标记为 companion，不替换 V4 primary；
- 报告 effect size、95% CI、q value、coverage 和方向；
- 训练 online AUROC 与 held-out AUROC 分开报告。

### 9.3 解释规则

- held-out 能力提升且 hidden 指标改变：候选“伴随有效训练的内部动力学”；
- hidden 指标改变但 held-out 不提升：policy drift diagnostic；
- 指标变化被 length/policy entropy/point count 吸收：昂贵 nuisance proxy；
- online 有趋势但 held-out AUROC 不复现：训练分布或 batch composition 信号；
- held-out AUROC 有变化但 correct/wrong mean 同步漂移：需优先检查尺度与 calibration drift。

---

## 10. 执行阶段与停止条件

### Phase A：数据与 verifier calibration

输出：

- 两个候选集的 calibration 表；
- 难度与数据源冻结决策；
- response cap；
- Prompt/verifier 审计；
- train/calibration/test manifest hashes。

未通过第 3 节 GO 条件时停止。

### Phase B：系统与 hidden smoke

依次验证：

1. 0-step generation/verifier；
2. 5-step online hidden timing；
3. 20-step optimizer smoke；
4. checkpoint 保存与恢复；
5. 固定 141 image key、最新 PNG 覆盖、异步队列和失败恢复；
6. 单个 held-out checkpoint 的 pass@k/AUROC 端到端流程。

smoke 权重和题目永久排除出正式结果。

### Phase C：正式 300-step GRPO

- tmux 持久运行；
- 每步上传训练 telemetry，追加在线 hidden scalar，并提交最新 PNG 绘图任务；
- 由独立绘图/上传队列异步更新固定 141 个 image key；
- 每 25 steps 保存 checkpoint；
- 保存后串行执行固定 held-out 测试；
- 测试完成后恢复训练；
- 任一 crash 从最后完整 checkpoint 恢复，不跳过测试点。

### Phase D：最终统计与报告

- 4,000 次 question bootstrap；
- pass@k checkpoint 曲线；
- 141 张 hidden correct/wrong/AUROC 图；
- controlled trend models；
- BH-FDR；
- coverage、length、truncation、policy entropy sensitivity；
- base/final MATH500 sanity check；
- final audit JSON 与实验报告。

---

## 11. 资源与预计耗时

### 11.1 生成量

不考虑过滤时，正式训练生成：

\[
300\times32\times8=76,800\ \text{responses}.
\]

若 mixed acceptance rate 为 `a`，预期 raw generation 为：

\[
N_{raw}\approx\frac{76,800}{a}.
\]

示例：

| acceptance | raw responses |
|---:|---:|
| 0.60 | 128,000 |
| 0.50 | 153,600 |
| 0.40 | 192,000 |
| 0.35 | 219,429 |

因此 acceptance 和平均 response length 是主要时间决定因素，不能只根据模型参数量估时。

### 11.2 时间预算

假设平均 response 为 1.5k-3k token，acceptance 为 0.4-0.6：

| 阶段 | 预计时间 |
|---|---:|
| 双数据 calibration | 3-8 小时 |
| 5-step hidden + 20-step optimizer smoke | 3-8 小时 |
| 300-step 正式训练，不含 online probe | 30-70 小时 |
| 复用 forward 的每步 H/V probe | 增加约 15%-40% |
| 13 个固定测试点 roll8 + hidden | 8-24 小时 |
| CPU 统计、bootstrap 与报告 | 1-3 小时 |

总预算：

\[
\boxed{55\text{-}120\ \text{hours}}
\]

即约 `2.5-5 days`。若平均 response 超过 4k token或 acceptance 低于 0.4，可能超过 120 小时。20-step smoke 后必须使用实测 tokens/s、step time、probe overhead 和 acceptance rate 重算正式 ETA。

### 11.3 存储

永久保存：

- 13 个 checkpoint；
- manifests 与 hashes；
- 训练 telemetry；
- rollout text/token ids/reward/status；
- rollout-level H/V scalar Parquet；
- held-out pass@k/AUROC 表；
- 图表与最终审计。

不永久保存：

- 全量 token x layer hidden tensors；
- online hook 的中间 activation；
- 重复的逐 step/逐 checkpoint PNG 快照；只保留当前 141 张最新 PNG。

正式启动前必须审计 checkpoint 预计体积和 H200 对应挂载盘剩余空间。

---

## 12. 最终交付物

```text
resolved_config.yaml
environment_audit.json
model_tokenizer_hashes.json
calibration_deepmath.csv
calibration_simplerl.csv
dataset_selection_decision.json
train_manifest.parquet
heldout_manifest.parquet
checkpoint_manifest.json
training_telemetry.parquet
online_hidden_metrics.parquet
heldout_behavior_metrics.parquet
heldout_hidden_metrics.parquet
passk_bootstrap_summary.csv
hidden_effects_fdr.csv
coverage_summary.csv
figures/
final_audit.json
EXPERIMENT_REPORT.html
```

`final_audit.json` 至少确认：

- 模型确为 `Qwen/Qwen3-1.7B-Base`；
- 数据源、难度带和 cap 与 calibration 决策一致；
- train/calibration/test 无 hash 泄漏；
- 13 个 checkpoint 是否完整；
- 每个测试点是否恰有 `256 x 8` 个预期 rollout；
- online probe 是否改变训练梯度或 batch；
- H1-H9/V1-V9/V4C schema 与 `cut_logic.md` 一致；
- 约 141 个固定 image key 是否去重且没有 step 派生 key；
- 所有 coverage、exclusion、SwanLab run id 和输入 hash 是否齐全。

---

## 13. 冻结后的执行顺序

```text
环境与磁盘审计
-> 模型/tokenizer/prompt/verifier 审计
-> DeepMath 与 SimpleRL 双 calibration
-> 冻结数据源、难度带、response cap
-> 生成 train/test manifests 并做 hash 防泄漏
-> 5-step online hidden timing smoke
-> 20-step optimizer/checkpoint/SwanLab smoke
-> 正式 300-step GRPO
-> 每 25 steps checkpoint + held-out roll8/pass@k/AUROC
-> 4,000 次 question bootstrap 与 controlled hidden analysis
-> SwanLab 141 图表与最终 HTML 报告
-> final audit
```

任何正式实验配置变化都必须发生在 calibration/smoke 结束、正式 step 0 开始之前，并写入新的 resolved config 和 manifest hash。正式训练启动后不再根据 reward、pass@k 或 hidden 曲线调整数据、指标、layer、reducer、reward、学习率或 checkpoint 位置。
