# H10-P / H10-O 计算方案

## 1. 目标

H10 在固定 layer、固定 response、固定 chunk representation 上，将回答过程中每一步 hidden-state 位移分解为：

- **H10-P（parallel progress）**：沿该 response 最终净方向的有符号推进比例；
- **H10-O（orthogonal expansion）**：垂直于该 response 最终净方向的展开比例。

H10-P/H10-O 保留 B1 -> B2 -> B3 -> B4 的回答阶段顺序，用于观察模型在回答前期和后期分别更偏向侧向展开还是向最终状态推进。

H10 使用完整 response 的最终状态定义参考方向，因此是离线诊断指标，不是因果的在线指标。H10-P 高只表示朝该 response 自己的最终状态收束，不代表最终答案正确。

## 2. 输入

对一条 response，在一个固定 layer 上取得按回答顺序排列的 chunk hidden states：

\[
h_0,h_1,\ldots,h_K, \qquad h_k\in\mathbb{R}^{D}
\]

对应的 response 相对进度为：

\[
0<t_0<t_1<\cdots<t_K\le 1
\]

其中：

- \(h_0\) 是当前轨迹中第一个有效 chunk 的 hidden state，不是 prompt 末尾状态；
- \(h_K\) 是最后一个有效 chunk 的 hidden state；
- \(t_k\) 是第 \(k\) 个 chunk 终点在 response 中的相对位置；
- H10-P/H10-O 必须分别在每个 layer、每种 chunk representation 上计算，不能先跨层平均 hidden state。

## 3. 相邻位移和路径长度

对相邻 chunk 计算位移：

\[
d_k=h_k-h_{k-1}, \qquad k=1,\ldots,K
\]

局部路径长度为：

\[
r_k=\|d_k\|_2
\]

如果 \(r_k\) 小于预先冻结的数值阈值，则该步没有可靠方向，应从 H10-P/H10-O 的分子和分母中同时排除，并记录 `excluded_zero_norm`。

## 4. 最终参考方向

完整 response 轨迹的净位移为：

\[
g=h_K-h_0=\sum_{k=1}^{K}d_k
\]

将其单位化：

\[
\hat g=\frac{g}{\|g\|_2}
\]

\(\hat g\) 表示从第一个观测 chunk 指向最后一个观测 chunk 的最终净方向。

如果 \(\|g\|_2\) 小于预先冻结的数值阈值，则该 response 的最终方向不可靠，四个 stage 的 H10-P/H10-O 均标记为无效。

## 5. 单步 H10-P

第 \(k\) 步在最终方向上的有符号投影长度为：

\[
a_k=d_k^\top\hat g
\]

用该步路径长度归一化：

\[
p_k=\frac{a_k}{r_k}
=\frac{d_k^\top\hat g}{\|d_k\|_2}
=\cos\theta_k
\]

其中 \(\theta_k\) 是 \(d_k\) 与最终方向 \(\hat g\) 的夹角，因此：

\[
-1\le p_k\le 1
\]

解释：

- \(p_k=1\)：完全沿最终方向推进；
- \(0<p_k<1\)：部分沿最终方向推进；
- \(p_k=0\)：该步完全垂直于最终方向；
- \(p_k<0\)：该步背离最终方向。

本方案不单独计算 H10-B。回退信息保留在 H10-P 的负值中。

## 6. 单步 H10-O

先从第 \(k\) 步位移中去掉沿最终方向的投影：

\[
q_k=d_k-a_k\hat g
=d_k-(d_k^\top\hat g)\hat g
\]

\(q_k\) 与 \(\hat g\) 正交。其归一化长度定义为：

\[
o_k=\frac{\|q_k\|_2}{r_k}
=\frac{\|d_k-(d_k^\top\hat g)\hat g\|_2}{\|d_k\|_2}
=|\sin\theta_k|
\]

因此：

\[
0\le o_k\le 1
\]

解释：

- \(o_k=0\)：该步与最终方向平行，可能正向推进，也可能反向回退；
- \(0<o_k<1\)：该步同时包含平行移动和垂直展开；
- \(o_k=1\)：该步完全垂直于最终方向。

H10-O 只保留垂直分量的大小，不保留其具体侧向方向。它是回答内侧向展开的几何代理量，不能单独等同于有效 exploration。

对每个有效单步都有：

\[
p_k^2+o_k^2=1
\]

## 7. Stage 划分

为与现有横向 H2 的 stage 口径保持一致，每个位移 \(d_k=h_k-h_{k-1}\) 按其终点进度 \(t_k\) 分配到 stage：

\[
I_{B1}=\{k\mid 0<t_k<0.25\}
\]

\[
I_{B2}=\{k\mid 0.25\le t_k<0.50\}
\]

\[
I_{B3}=\{k\mid 0.50\le t_k<0.75\}
\]

\[
I_{B4}=\{k\mid 0.75\le t_k\le1\}
\]

如果实际代码中现有 `stage_mask` 对 0.25 边界采用不同归属，则 H10 必须直接复用现有 `stage_mask(progress[1:], stage)`，避免 H2 和 H10 的边界口径不一致。

跨越 stage 边界的位移整体归入其终点所在 stage，不对单个位移按 token 比例拆分。

## 8. Stage-local H10-P

stage \(s\) 的有效路径长度为：

\[
L_s=\sum_{k\in I_s}r_k
\]

H10-P 定义为单步有符号平行比例的路径长度加权平均：

\[
\boxed{
P_s=
\frac{\sum_{k\in I_s}r_kp_k}
{\sum_{k\in I_s}r_k}
}
\]

代入 \(r_kp_k=d_k^\top\hat g\)，可写为：

\[
\boxed{
P_s=
\frac{\sum_{k\in I_s}d_k^\top\hat g}
{L_s}
}
\]

取值范围：

\[
-1\le P_s\le1
\]

解释：

- \(P_s\) 较高：该 stage 的路径主要朝最终状态推进；
- \(P_s\) 接近 0：正向和反向移动相互抵消，或路径主要沿垂直方向展开；
- \(P_s<0\)：该 stage 整体背离最终状态。

## 9. Stage-local H10-O

H10-O 定义为单步垂直比例的路径长度加权平均：

\[
\boxed{
O_s=
\frac{\sum_{k\in I_s}r_ko_k}
{\sum_{k\in I_s}r_k}
}
\]

代入 \(r_ko_k=\|q_k\|_2\)，可写为：

\[
\boxed{
O_s=
\frac{
\sum_{k\in I_s}
\|d_k-(d_k^\top\hat g)\hat g\|_2
}{L_s}
}
\]

取值范围：

\[
0\le O_s\le1
\]

解释：

- \(O_s\) 较高：该 stage 的路径主要表现为相对最终方向的侧向展开；
- \(O_s\) 较低：该 stage 的路径主要与最终方向平行，但仅凭 O 无法区分正向推进和反向回退，必须联合查看 P。

由于 \(P_s\) 和 \(O_s\) 是多个单步值的路径长度加权平均，一般不满足：

\[
P_s^2+O_s^2=1
\]

也不满足：

\[
P_s+O_s=1
\]

## 10. 顺序输出

每条 response 必须分别保留四个 stage 的结果：

\[
(P_{B1},P_{B2},P_{B3},P_{B4})
\]

\[
(O_{B1},O_{B2},O_{B3},O_{B4})
\]

不能只输出四个 stage 的总平均，否则会再次丢失回答前期到后期的顺序变化。

可额外输出两个前后转换 companion：

\[
P_{early}=\frac{P_{B1}+P_{B2}}{2},
\qquad
P_{late}=\frac{P_{B3}+P_{B4}}{2}
\]

\[
O_{early}=\frac{O_{B1}+O_{B2}}{2},
\qquad
O_{late}=\frac{O_{B3}+O_{B4}}{2}
\]

\[
T_P=P_{late}-P_{early}
\]

\[
T_O=O_{early}-O_{late}
\]

其中：

- \(T_P>0\)：后期比前期更偏向最终状态推进；
- \(T_O>0\)：垂直展开更多发生在前期而非后期。

上述 early/late companion 不替代 B1-B4 主输出。

## 11. 有效性与覆盖条件

每条 response、每个 layer、每种 chunk representation 的 H10-P/H10-O 应执行以下检查：

1. hidden-state 轨迹必须是有限值，并至少包含两个有效位移；
2. \(\|h_K-h_0\|_2\) 必须高于冻结的最终方向阈值；
3. 每个 stage 至少包含一个 \(r_k\) 高于局部位移阈值的有效步；
4. stage 的 \(L_s\) 必须高于数值阈值；
5. 无效 stage 输出 `coverage=false`，不得用 0 填充；
6. 同时保存 `n_valid`、`excluded_zero_norm` 和 `stage_path_length`，用于覆盖率审计。

## 12. 绘图口径

H10-P 与 H10-O 各自生成一张四 panel 图：

```text
H10_parallelProgress   | chunkMean/chunkLast | layer | stageNative
H10_orthogonalExpansion| chunkMean/chunkLast | layer | stageNative
```

每张图保持现有横向指标格式：

- B1、B2、B3、B4 分 panel；
- 横轴为 training step；
- 左轴为 correct/wrong 的 question-balanced metric mean；
- 右轴为 checkpoint-matched within-question AUROC；
- correct/wrong/AUROC 使用完全相同的 rollout、正确性标签和覆盖过滤；
- AUROC 同时保留原始方向，不用 `max(AUC, 1-AUC)` 替代主值。

## 13. 性质与限制

H10-P/H10-O 具有以下性质：

- 对所有 hidden state 加同一个平移向量不变；
- 对 hidden state 做全局正比例缩放不变；
- 对 hidden coordinate 做正交旋转不变；
- 使用完整 response 的终点，因此不能直接作为严格因果的 token-level 在线奖励；
- P/O 描述的是相对自身最终状态的几何过程，错误 response 也可能有很高的 late P；
- O 高可能是有效探索、纠错、犹豫或噪声，必须结合 stage、正确性 AUROC、训练动态和后续能力变化解释。

