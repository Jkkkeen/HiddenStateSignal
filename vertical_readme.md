# Vertical Hidden-State Toolkit

这份教程用于在已经保存好 hidden states 的服务器上，离线计算 Qwen3-8B-Base、MiMo-7B-Base/SFT 的逐层纵向指标。正常流程只使用 CPU，不加载模型权重、不重新生成回答，也不执行模型 forward。

当前稳定流程计算 V1、V3、V4、V6、V7，并输出 correct-wrong、within-question AUROC、depth summaries 和标准图。扩展指标 V2、rolling/cumulative V5/V8/V9 以及 bootstrap、cluster permutation、nuisance controls 使用同一数据契约。

## 1. Environment

推荐环境：Linux、Python 3.11、足够容纳源 hidden states 和输出 Parquet 的本地磁盘，以及 `tmux`。

```bash
git clone https://github.com/Jkkkeen/HiddenStateSignal.git
cd HiddenStateSignal
git checkout vertical-toolkit

python3.11 -m venv .venv-vertical
source .venv-vertical/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vertical.txt
python -m pytest tests -q
```

先不要修改真实数据，也不要把真实数据复制进 Git 仓库。

## 2. Synthetic Quickstart

先生成极小的 response-only 示例，确认环境、adapter 和分析依赖正常：

```bash
python examples/synthetic_hidden/make_fixture.py \
  --output examples/synthetic_hidden/generated
```

生成内容包含：

- Qwen3 Base raw token hidden states；
- MiMo Base/SFT raw token hidden states，并带独立 MTP 标记；
- 一个已经池化的 NPZ 示例；
- `synthetic_qwen.yaml` 和 `synthetic_mimo.yaml`。

这些文件位于 Git ignore 范围内。

## 3. Input Contract

### Raw mode

canonical 形状为：

```text
[response_token, hidden_state_position, hidden_dimension]
```

如果 tensor 还包含 prompt 或 padding，metadata 必须提供精确的 `response_start` 和 `response_stop`。工具不会猜 response 边界。

### Pooled mode

canonical 形状为：

```text
[trajectory_endpoint, hidden_state_position, hidden_dimension]
```

必须提供 `endpoints` 或 `progress`。如果两者同时存在，它们必须一致。

### Required metadata

每条 record 至少需要：

```text
question_id
rollout_id
response_token_count
num_decoder_layers
```

推荐同时保留：

```text
record_id
is_correct
checkpoint
global_step
training_progress
reward
format_correct
difficulty
layer_kind
```

`hidden_states[0]` 必须是 embedding output，随后是 decoder block outputs。MiMo 的 MTP/non-decoder state 使用 `layer_kind: mtp` 标记，不进入 decoder 主 profile。

## 4. Configuration

从示例配置复制一份本地配置：

```bash
cp configs/mimo_7b.example.yaml .vertical-local-mimo.yaml
```

只需要修改 `source`、adapter 选项和本地 `output_root`。本地配置已被 `.gitignore` 排除。

常见 axis 声明：

```yaml
axis_order: token,layer,dim
```

不要使用 `auto` 猜 axis order。`decoder_layers: auto` 只有在每条 metadata 已保存 `num_decoder_layers` 时才可用。

## 5. Preflight

Preflight 是只读检查，不会修改 hidden-state 文件：

```bash
bash scripts/inspect_vertical_data.sh \
  --config .vertical-local-mimo.yaml \
  --output ./runs/mimo_vertical/input_audit.json \
  --sample-limit 16
```

只有 `passed: true` 才能继续。重点检查：

- tensor shape、dtype、finite rate；
- axis order；
- question/rollout ID；
- response-only 或精确 response slice；
- embedding + decoder 层数；
- MTP candidate positions；
- duplicate record IDs；
- Base/SFT 是否使用可对齐的问题和回答。

## 6. Frozen Pooling

raw response states 使用：

- `mean_w128_s32`：每个 endpoint 前最多 128 个 response tokens 的均值；
- `last_s32`：同一 endpoint 的最后一个 response token；
- endpoint stride 为 32，response 最后一个 token 永远补为最终 endpoint。

B1-B4 定义为：

```text
B1 [0.00, 0.25)
B2 [0.25, 0.50)
B3 [0.50, 0.75)
B4 [0.75, 1.00]
```

## 7. Base Calibration

Base calibration 必须 label-blind：correctness 不参与拟合。

- Qwen3 Base 拟合 Qwen3 calibrator；
- MiMo Base 拟合 MiMo calibrator；
- MiMo Base 与 MiMo SFT 共用 MiMo Base calibrator；
- Qwen3 与 MiMo calibrator 不能互换；
- `mean_w128_s32` 与 `last_s32` 分别拟合。

公共分量使用 rollout-equal weighting；coordinate mean/sigma 使用 chunk-equal weighting。

## 8. Stable Metrics

主 profile 的 `layer_index=l` 表示目标 state `h_l`。对 update 指标，它表示 `h_l - h_(l-1)` 的目标层。

- V1 raw/relative layer-update norm：layer 1..L；
- V3 Base-common 去均值后的相邻 state angle：layer 1..L；
- V4 相邻 layer updates turning angle：layer 2..L；
- V6 raw activation coordinate-energy entropy：layer 0..L；
- V7 centered layer-difference coordinate-energy entropy：layer 1..L。

结构上没有定义的位置写 `NaN`，并写独立 coverage count，绝不写成 0。

## 9. Smoke

Smoke 对每个 dataset 只取少量 record，但必须包含 Base、目标 condition、两种 representation 和尽可能多的 mixed-outcome questions。

```bash
bash scripts/run_vertical_smoke.sh \
  --config .vertical-local-mimo.yaml \
  --output-root ./runs/mimo_vertical_smoke
```

继续 formal 前确认：

```text
smoke_approval.json -> status: passed
audit/input_audit.json -> passed: true
profiles/*.audit.json -> passed: true
analysis/analysis_audit.json -> passed: true
figures/*.png -> 非空
```

## 10. Formal tmux

正式计算必须从 detached tmux session 启动：

```bash
bash scripts/launch_vertical_formal_tmux.sh \
  --config .vertical-local-mimo.yaml \
  --output-root ./runs/mimo_vertical_formal \
  --smoke-approval ./runs/mimo_vertical_smoke/smoke_approval.json
```

查看和进入 session：

```bash
tmux ls
tmux capture-pane -pt vertical_formal_<run_id> -S -80
tmux attach -t vertical_formal_<run_id>
```

离开但不停止任务：按 `Ctrl-b`，再按 `d`。

正式完成后，tmux session 可以自然退出；`run_status.json` 和 final audit 才是完成依据。

## 11. Restart And Resume

同一配置和 output root 可以重新运行。只有同时存在通过的 audit、匹配的输入 hash 和完整 Parquet 时，已有 partition 才会复用。

不要删除 `.tmp` 文件来伪造完成。失败 partition 不会被当成有效输出。

## 12. Outputs

```text
runs/<run_id>/
  manifest.json
  run_status.json
  smoke_approval.json
  audit/
  calibrators/
  profiles/
  analysis/
  figures/
  logs/
```

需要返回给项目负责人的最小文件：

- `input_audit.json`；
- `final_audit.json`；
- 脱敏后的 manifest/config；
- policy/outcome/AUROC/depth summary；
- 关键 figures；
- 失败时的日志末尾。

## 13. Interpretation

Qwen3-8B 只有 Base 时，可以报告层 profile 和 outcome separation，不能称为训练动态。

MiMo Base/SFT 如果使用各自生成的不同 response，差异包含权重变化与文本变化。只有相同 prompt/response 在两种 condition 下的 hidden states 才支持 representation-only 解释。

跨模型只比较 relative depth、family 内 Base-standardized change、normalized entropy 和效应方向。不要直接相减 Qwen3 与 MiMo 的 hidden vectors 或 raw norms。

## 14. Troubleshooting

### `axis_order` error

检查真实 tensor 的三个轴，并在配置中明确声明。不要通过反复试排列来选择“看起来正常”的结果。

### `response_token_count` error

确认 tensor 是否 response-only。如果包含 prompt，补充精确的 `response_start/response_stop`。

### calibrator family mismatch

确认 MiMo SFT 使用 MiMo Base calibrator，Qwen3 使用 Qwen3 Base calibrator，并且 representation 相同。

### no mixed-outcome questions

这不是计算故障。correct-wrong 和 AUROC 会标记 unsupported；增加同题 rollout 或更换 cohort 后再计算。

### tmux session already exists

先运行 `tmux ls` 检查旧任务，不要直接覆盖同名 session。确认旧任务结束后使用新的 run ID。

## 15. Do Not Push

Do not push 以下内容到 GitHub：

- 原始 hidden states；
- 模型权重和 checkpoints；
- 真实 pooled caches 和 calibrators；
- 完整 formal Parquet/figures/logs；
- 私有服务器绝对路径；
- 包含敏感 prompt/response 的 manifest。

只提交代码、测试、教程、脱敏配置、审计摘要和经过确认的小型图表。
