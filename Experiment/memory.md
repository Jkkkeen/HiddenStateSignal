给我详细讲解SHEAR和stALT两篇文章具体的方法和实验






星期日00:46






Worked for 13m 20s



先给结论：
SHEAR 是一个真正进入 RLVR/GRPO 训练的 credit assignment 方法：利用正确与错误 rollout 的 hidden-state 分布差异，决定哪些 token 应该被更强地奖励或惩罚。
StALT 是一个无需训练的轨迹诊断指标：同时观察 token 方向和 layer 方向的 hidden-state 变化，给一条完整推理生成一个标量分数。论文没有证明把它加入 RL 后能提升训练。
SHEAR
SHEAR 的全名是 Span-level Hidden state Enabled Advantage Reweighting。它解决的是 GRPO 的一个核心问题：一条 rollout 最终答错后，GRPO 会给这条 rollout 的所有 token 相同的负 advantage，即使前 80% 的推理其实是正确的。
标准 GRPO 中，第 \(i\) 条 rollout 的 advantage 近似为：
\[
A_i=\frac{r_i-\bar r}{\sigma_r+\epsilon}
\]所有 token 都使用 \(A_i\)：
\[
\nabla J\approx
\sum_t A_i\nabla\log\pi_\theta(y_{i,t}\mid y_{i,<t},x)
\]SHEAR 把它改成：
\[
\widetilde A_{i,t}=A_i\omega_{i,t}
\]其中 \(\omega_{i,t}\) 是从 hidden state 计算出的 token 权重。
1. 把 rollout 切成 span
对 rollout \(i\)，使用长度 \(w\)、步长 \(s\) 的滑动窗口：
\[
S_k^{(i)}=[1+(k-1)s,\;1+(k-1)s+w-1]
\]默认：
\[
w=100,\qquad s=25
\]论文只取 LM head 前的最后一层 hidden state：
\[
h_{i,t}\in\mathbb R^d
\]一个 span 不再被表示成均值，而是其中所有 token hidden state 构成的经验分布：
\[
\widehat P_{i,k}
=
\frac1{|S_k|}
\sum_{t\in S_k}\delta_{h_{i,t}}
\]这点很重要：它想保留均值、方差和分布形状。不过它把 span 视为无序点集，因此窗口内部的 token 顺序没有直接进入距离。
2. 找 opposing group
对于正确 rollout，比较对象是同一问题下的错误 rollout；对于错误 rollout，比较对象是正确 rollout：
\[
O(i)=
\begin{cases}
G^-,&r_i=1\\
G^+,&r_i=0
\end{cases}
\]然后计算当前 span 到 opposing group 中所有 span 的最小 Wasserstein 距离：
\[
d_{i,k}
=
\min_{j\in O(i)}
\min_q
W_\epsilon(\widehat P_{i,k},\widehat P_{j,q})
\]实际使用 Sinkhorn 近似计算 entropic Wasserstein-1 距离。
取最小值的含义是：只要错误 span 还能和某一种正确推理路径相似，就不要强烈惩罚它。只有它和所有正确路径都不相似，距离才会很大。
3. 从 span 权重还原到 token 权重
先用整个 rollout group 的平均 hidden-state norm 归一化：
\[
\bar n=
\frac1{N_G}
\sum_{i,t}\|h_{i,t}\|_2
\]一个 token 通常属于多个重叠 span，SHEAR 使用 max pooling：
\[
\omega_{i,t}
=
\frac1{\bar n}
\max_{k:t\in S_k^{(i)}}d_{i,k}
\]最终：
\[
\widetilde A_{i,t}=A_i\omega_{i,t}
\]因此：
正确 rollout 中，和错误轨迹高度分离的 token 获得更强正更新。
错误 rollout 中，和所有正确轨迹高度分离的 token 获得更强负更新。
与 opposing group 仍然相似的部分，更新较弱。
注意它不是把“错误发生之前”的 token 变成正 advantage，只是减弱这些 token 所承受的负更新。
4. SHEAR 的诊断实验
论文先在 MATH500、Qwen2.5-Math-7B 上验证 Wasserstein 信号：
每题采样 \(G=8\) 条 rollout。
只保留同时有正确和错误 rollout 的问题，占 MATH500 的 83.8%。
窗口 \(w=100\)，stride \(s=25\)。
排除含最终答案的 span，避免答案泄漏。
从每个截断 prefix 再生成 16 次，用正确比例估计 continuation success。
结果：
随推理推进，continuation success 下降、Wasserstein 距离上升，聚合曲线 Spearman \(\rho=-0.96\)。
同一条轨迹相邻 stride 的局部变化中，\(\Delta W\) 与 \(\Delta Accuracy\) 的 Spearman \(\rho=-0.42\)。
共分析 23,704 个原始正确 span 和 21,352 个原始错误 span。
Wasserstein 较低时，正确/错误 span 的区分 AUC 约 0.65；距离最高的区间 AUC 超过 0.97。
这里的 AUC 不是用 Wasserstein 直接预测某条 rollout 是否正确，而是在不同距离区间中分析正确与错误 span 的 continuation accuracy 分离程度。
5. 理论结果
论文假设正确和错误轨迹在某个位置 \(\tau\) 前来自相同 hidden-state 分布，之后错误轨迹从 \(P_t\) 偏移到 \(Q_t\)。
对于 span \(S\)，定义总体分布差异：
\[
D(S)=W_1(\bar P_S^-,\bar P_S^+)
\]有限样本噪声为：
\[
\eta(n,d)=2\widetilde C_dMn^{-1/d}
\]论文证明：
\[
\mathbb E[W_1(\widehat P_S^-,\widehat P_S^+)]
\leq \eta(n,d)
\]对于分叉前 span，而分叉后：
\[
\mathbb E[W_1(\widehat P_S^-,\widehat P_S^+)]
\geq D(S)-\eta(n,d)
\]所以如果：
\[
D(S)>2\eta(n,d)
\]分叉后的距离期望严格大于分叉前。
但这是条件性证明，不是说任何错误都一定能被检测。特别是 hidden dimension 很高时，\(n^{-1/d}\) 的经典 Wasserstein 收敛界很弱；实际有效性主要仍由实验证据支持。
6. RL 实验
数学训练集是 MATH 8.5K，代码训练集是 Eurus-RL-Code 25K。模型包括 Qwen2.5-Math-7B、Qwen2.5-14B-Base、Llama3.1-8B-Instruct 和 Qwen2.5-Coder-7B。
数学五个 benchmark 的平均成绩：
模型	GRPO	SHEAR	提升
Qwen2.5-Math-7B	49.6	51.2	+1.6
Llama3.1-8B	24.7	26.2	+1.5
Qwen2.5-14B	45.4	46.9	+1.5

代码五个 benchmark：
模型	GRPO	SHEAR
Qwen2.5-Coder-7B	63.1	63.9
Llama3.1-8B	54.7	55.8
Qwen2.5-14B	67.1	67.7

训练设置包括 batch size 512、8 rollouts、学习率 \(10^{-6}\)、最大 3K response tokens、训练温度 1.0，不使用 KL 和 entropy loss。
主要消融结论：
Wasserstein 最好，但 Chamfer 和 MMD 接近，说明“分布级距离”可能比 Wasserstein 本身更关键。
span mean cosine 低于 GRPO：0.4837 对 0.4884。
每条 rollout 单独归一化仍优于 GRPO，说明主要收益确实来自 rollout 内部的 token credit。
更小 stride 更好，窗口长度 80、100、120 差异不大。
\(G=16\) 比 \(G=8\) 更有利，因为 opposing set 更丰富。
额外训练时间为 7.4%–15.5%。
需要谨慎看待：论文报告的是在五个评测 benchmark 上选择的 best-observed checkpoint，不是固定 final checkpoint；这可能使数字偏乐观。论文也只明确说明诊断阶段排除了答案 span，训练算法部分没有同样明确排除答案区域。
StALT
StALT 的全名是 Spatiotemporal Amplitude of Latent Transition。它从完整 hidden-state 张量出发：
\[
H\in\mathbb R^{T\times(L+1)\times D}
\]其中 \(t\) 是生成 token 位置，\(l\) 是层。
它区分两种变化。
1. 横向 token 变化
固定层 \(l\)，比较相邻 token 位置：
\[
\Delta_{\text{time}}[t,l]
=
\|h_t^l-h_{t-1}^l\|_2
\]这就是你所说的“横向 hidden state 语义变化”。但严格说，它比较的是两个相邻 token 的表示，不是同一个 token 的状态随真实时间演化。
2. 纵向 layer 变化
固定 token 位置 \(t\)，比较相邻层：
\[
\Delta_{\text{layer}}[t,l]
=
\|h_t^l-h_t^{l-1}\|_2
\]它表示这个 token 在第 \(l\) 层经历了多大的 residual transformation。
论文发现：
正确推理的 \(\Delta_{\text{time}}\) 差异广泛分布在多个中高层。
\(\Delta_{\text{layer}}\) 的正确性差异集中在少数层。
因此横向变化是主要信号，纵向变化更像“选择哪些层值得观察”。
3. StALT 聚合
论文去掉 embedding 层的横向变化，因为 embedding 差异主要反映 token identity；再去掉 layer change 的第一行，使两个矩阵都变成 \((T-1)\times L\)。
对每个 token 位置，使用纵向变化为不同层分配 softmax 权重：
\[
W[t,l]
=
\frac{
\exp(\Delta_{\text{layer}}[t,l]/\tau)
}{
\sum_{k=1}^{L}
\exp(\Delta_{\text{layer}}[t,k]/\tau)
}
\]然后读取这些层上的横向变化：
\[
\operatorname{StALT}
=
\frac1{T-1}
\sum_{t=2}^{T}
\sum_{l=1}^{L}
W[t,l]\Delta_{\text{time}}[t,l]
\]直观上：如果当前 token 在第 24 层发生了最大的纵向更新，StALT 就重点观察第 24 层上当前 token 与前一个 token 的横向差异。
所以它不是：
\[
\Delta_{\text{time}}+\Delta_{\text{layer}}
\]而是：
\[
\text{横向运动幅度}\times\text{纵向层选择权重}
\]4. 大规模实验
模型包括：
Reasoning models：Qwen3-1.7B/4B/8B、SmolLM3-3B、gpt-oss-20b。
Non-reasoning baselines：Qwen2.5-Math-7B、Llama3.1-8B-Instruct。
数据集包括 GSM8K、MATH500、GPQA-Diamond、MMLU-Pro STEM 和 Non-STEM。生成温度 0.7、top-p 0.95、最长 16K token，结果平均五次运行。
Qwen3 模型组的平均 AUROC：
指标	GSM8K	MATH500	GPQA	MMLU-STEM	Non-STEM
负生成长度	.83	.73	.64	.75	.68
StALT	.83	.74	.67	.79	.69

StALT 在 Qwen3 上有效，但相对生成长度的提升其实不大。对于其他 LRM，生成长度甚至普遍强于 StALT；对于非推理模型，StALT 通常只有接近随机的效果。
以 Qwen3-4B 为例：
\[
\text{MATH500}=0.69,\quad
\text{GSM8K}=0.84,\quad
\text{GPQA}=0.70,\quad
\text{MMLU-STEM}=0.79
\]论文在 Qwen3-4B/MATH500 上做了长度分层，短、中、长三组内 StALT 仍有预测力。但这个 length-control 只展示了一个模型和一个数据集，不能完全排除其他条件下的长度混淆。
5. 干预实验
论文用 Qwen3-4B 和 s1K-1.1 的 472 道整数题做了三组干预。
Reasoning amplification：
Base 模型准确率 8.1%。
reasoning-tuned 但关闭 thinking：27.5%。
开启 thinking：65.4%。
StALT 随这三个条件单调上升。
CoT scaffolding：
不提供推理：65.4%。
提供一半 CoT：74.7%。
提供完整 CoT：86.8%。
完整 CoT 加答案：88.1%。
随着推理被放进 prompt，正确组和错误组各自的 StALT 都下降，说明模型需要自行执行的推理减少。总体平均 StALT 后期看起来回升，是因为正确样本比例增加造成的组成效应。
SFT intervention：
在相同 s1K 推理轨迹上训练 Qwen3-4B 十个 epoch。
准确率轻微上升。
StALT 持续下降。
生成长度也下降，但比 StALT 更缓慢。
作者解释为：模型逐渐参数化或记住这些推理模式，不再需要同等幅度的在线内部更新。不过训练集和评测题相同，所以这更像受控记忆实验，不是泛化证据。
6. StALT 消融
反向聚合，也就是用横向变化选择时间、累计纵向变化，在 Qwen3 上经常接近或低于随机。
L2 明显优于 cosine，说明信号主要在运动幅度，不仅是方向。
thinking-only 基本等于 full-trace，answer-only 较弱。
前 25% 已有信号，前 75% 通常接近完整轨迹。
\(\tau=0.01\) 到 10 几乎不敏感；完全均匀层权重通常变差。
温度不敏感也表明 raw layer L2 的 softmax 很可能已经接近硬层选择。
两篇论文的本质区别
维度	SHEAR	StALT
使用层	最后一层	所有层
横向/纵向	span 内横向分布	横向变化 + 纵向层选择
是否需要正确标签	需要 group outcome	不需要
输出	每个 token 的 advantage 权重	每条 rollout 一个标量
是否真正做 RL	是	否
对照对象	同题正确/错误 rollout	单条轨迹自身相邻 token/层
主要风险	opposing set、答案泄漏、非平稳表示	长度/norm 混淆、只能事后评分

对你的目标而言，SHEAR 更接近“怎么把 hidden state 用于 RL”，StALT 更接近“应该观察 hidden-state 张量中的哪些层和哪些 token”。
最合理的理解不是直接把 StALT 当额外 reward，而是把它保留成局部量：
\[
s_t^{\text{StALT}}
=
\sum_l W[t,l]\Delta_{\text{time}}[t,l]
\]然后让它决定 SHEAR/ER 信号在哪些 token 或哪些层可信。因为单纯奖励高 StALT，模型可能通过生成更长、hidden norm 更大或无效振荡的轨迹来获得奖励。
另外，你现有的 h_chunk[K,37,4096] 只能计算 chunk-level StALT 近似，不能复现论文的 token-level StALT；chunk mean 已经丢掉了 SHEAR 所依赖的 span 内 hidden-state 分布。当前把 chunk × layer 直接 flatten 的 M1 还会产生 chunk0-layer36 → chunk1-layer0 的人工跳变，它既不是 SHEAR，也不是干净的 StALT。


星期日00:59






1 annotation





这里的“正确性信号”不是说某一层直接存着“正确/错误”标签，而是统计上：
\[
\mathbb E[\Delta\mid\text{correct}]
-
\mathbb E[\Delta\mid\text{incorrect}]
\]明显不为零，因此只看这个量，就能一定程度区分正确和错误轨迹。
设 hidden state 是二维网格：
\[
H[t,l]=h_t^l
\]其中横轴是 layer，纵轴是生成 token 位置。
横向变化为什么在很多高层都有信号
StALT 所说的横向变化实际是固定层、比较相邻 token：
\[
\Delta_{\text{time}}[t,l]
=
\|h_t^l-h_{t-1}^l\|_2
\]论文发现，在正确推理中，中高层的很多 \(l\) 都表现出更大的 \(\Delta_{\text{time}}\)。意思是：
随着推理从 token \(t-1\) 推进到 token \(t\)，正确轨迹在许多中高层都产生了更明显的表示变化；错误轨迹的表示变化相对更弱、更平坦或更重复。

例如：
             layer 10   layer 15   layer 20   layer 25   layer 30
正确轨迹       小          大          大          大          大
错误轨迹       小          中          中          小          中
正确-错误      弱          强          强          强          强
因此热力图上会出现一片跨越多个高层的区域，而不是只有一个亮点。
这不意味着每个高层都独立完成了一次推理。因为高层 hidden state 包含了之前所有层的累计结果。
纵向变化为什么只集中在少数层
纵向变化固定同一个 token，比较相邻层：
\[
\Delta_{\text{layer}}[t,l]
=
\|h_t^l-h_t^{l-1}\|_2
\]Transformer 的 residual stream 可以近似写成：
\[
h_t^l=h_t^{l-1}+F_l(h_{\le t}^{l-1})
\]所以：
\[
\Delta_{\text{layer}}[t,l]
=
\|F_l(h_{\le t}^{l-1})\|_2
\]它测量的是第 \(l\) 层对当前 token “新写入了多少东西”。
如果正确推理和错误推理的关键区别主要由第 18、24、29 层完成，那么：
layer         15   16   17   18   19   20 ... 24 ... 29
纵向差异       小    小    小    大    小    小     大     大
热力图就表现为少数 layer hotspot。
但第 18 层写入的结果会继续传给第 19、20、21……层。因此，同一个关键计算可以造成：
在 \(\Delta_{\text{layer}}\) 中：只在第 18 层出现局部峰值；
在 \(\Delta_{\text{time}}\) 中：第 18 层之后的多个高层都携带这个变化。
数学上，横向差异近似包含截至当前层的所有累计更新：
\[
h_t^l-h_{t-1}^l
\approx
(h_t^0-h_{t-1}^0)
+
\sum_{j=1}^{l}
\left[
F_j(t)-F_j(t-1)
\right]
\]而纵向差异只看当前一层：
\[
h_t^l-h_t^{l-1}=F_l(t)
\]所以可以这样理解：
纵向变化定位“计算主要在哪些层发生”，横向变化观察“这些计算如何随着推理 token 推进，并在后续层中持续表现出来”。

一个类比是流水线：少数工位真正执行了关键加工，所以“工位前后变化”集中在少数位置；但加工结果会被后面很多工位一直携带，因此观察连续产品时，后面很多工位都能看到正确产品和错误产品的差异。
这也解释了 StALT 的设计：
\[
W[t,l]=\operatorname{softmax}_l
\left(\Delta_{\text{layer}}[t,l]/\tau\right)
\]先用纵向变化找出当前 token 上计算活跃的层，再读取这些层上的横向变化：
\[
s_t=
\sum_l W[t,l]\Delta_{\text{time}}[t,l]
\]最后对 token 平均得到 StALT。
需要注意三个边界：
“很多高层有信号”不表示这些层都在表达相同语义，只表示它们的表示运动幅度与正确性相关。
“少数层有热点”不表示只有这些层参与推理，只表示正确与错误之间新增更新的差异在那里最明显。
这是相关性，不是因果性。要证明某层真的产生关键推理，需要对 hotspot 层做 activation patching、替换或定向干预。
因此，纵向变化更适合叫“计算位置/层深定位”，而不是直接叫“思考深度”；横向变化更像推理过程随 token 展开的语义轨迹。


星期日21:00






1 annotation


span wasserstain是什么






星期日21:42






这里的 span Wasserstein 应该理解为：
比较两个 token span 中 hidden-state 分布差异的 Wasserstein 距离。

它不是一个模型，而是一个距离指标。SHEAR 使用它来判断“这段局部推理是否已经偏离了另一类轨迹”。论文是 Hidden States Know Where Reasoning Diverges。
1. Span 是什么？
一个 span 就是连续的一段 token。
例如，假设 rollout 是：
t1 t2 t3 ... t100 t101 ... t200
窗口大小 \(w=100\)，stride \(s=25\)，那么会得到：
\[
S_1=[t_1,\ldots,t_{100}]
\]\[
S_2=[t_{26},\ldots,t_{125}]
\]\[
S_3=[t_{51},\ldots,t_{150}]
\]这些 span 相互重叠，用来避免错误刚好落在两个窗口之间。
对每个 token \(t\)，提取最后一层 hidden state：
\[
h_t\in\mathbb R^d
\]因此一个 span 不再是一个向量，而是一组向量：
\[
H_{i,k}
=
\{h_{i,t}:t\in S_{i,k}\}
\]例如 \(w=100\) 时：
\[
H_{i,k}\in\mathbb R^{100\times d}
\]这里 \(i\) 表示第几条 rollout，\(k\) 表示第几个 span。
2. Span 的 hidden-state 分布
SHEAR 把一个 span 中的 hidden states 看成一个经验分布：
\[
\widehat P_{i,k}
=
\frac{1}{|S_{i,k}|}
\sum_{t\in S_{i,k}}\delta_{h_{i,t}}
\]其中 \(\delta_{h_{i,t}}\) 表示在 hidden-state 点 \(h_{i,t}\) 上放置一个质量。
所以：
不是只看 span 的平均 hidden state；
不是只看某一个 token；
而是看这个窗口内所有 hidden states 在向量空间中的整体分布。
例如，下面两个 span 的均值可能相同：
\[
H^+=[-1,0,1]
\]\[
H^-=[0,0,0]
\]但它们的分布形状不同。仅比较均值会认为它们相同，而 Wasserstein 距离仍然可以发现它们的 spread 不同。
3. Wasserstein 距离是什么？
Wasserstein 距离可以理解成：
把一个分布“搬运”成另一个分布，最小需要付出多少运输成本。

对于两个离散分布：
\[
P=\sum_a p_a\delta_{x_a},
\qquad
Q=\sum_b q_b\delta_{y_b}
\]需要寻找一个运输方案 \(\pi_{ab}\)，表示从 \(x_a\) 搬多少质量到 \(y_b\)。
一阶 Wasserstein 距离是：
\[
W_1(P,Q)
=
\min_{\pi\in\Pi(P,Q)}
\sum_{a,b}\pi_{ab}\|x_a-y_b\|_2
\]其中 \(\Pi(P,Q)\) 表示所有满足边缘分布约束的运输方案。
直观上：
两个分布很接近，可以用很短的距离搬过去，\(W_1\) 小；
两个分布分离很远，需要大量搬运，\(W_1\) 大。
SHEAR 实际使用的是 一阶 Wasserstein 距离的 Sinkhorn 近似，不是严格意义上的 \(W_2\)。所以你看到的概念公式如果写成：
\[
u_{q,k}
=
W_2\left(
P(H_{q,k}\mid correct),
P(H_{q,k}\mid wrong)
\right)
\]可以作为一般性写法，但对于 SHEAR，更准确的是：
\[
u_{q,k}
=
W_1\left(
P(H_{q,k}\mid correct),
P(H_{q,k}\mid wrong)
\right)
\]实际训练时使用的是正则化版本：
\[
W_\varepsilon(P,Q)
=
\min_{\pi\in\Pi(P,Q)}
\left[
\sum_{a,b}\pi_{ab}\|x_a-y_b\|_2
+
\varepsilon
\,\mathrm{KL}(\pi\|P\otimes Q)
\right]
\]Sinkhorn 算法就是高效求这个运输问题的近似解。
4. 你的公式是什么意思？
你给出的公式是：
\[
u_{q,k}
=
W_2\left(
P(H_{q,k}\mid correct),
P(H_{q,k}\mid wrong)
\right)
\]逐项解释：
\(q\)：第 \(q\) 道题；
\(k\)：第 \(k\) 个 span；
\(H_{q,k}\)：这道题对应 span 的 hidden-state 集合；
\(P(H_{q,k}\mid correct)\)：正确 rollout 中该 span 的 hidden-state 分布；
\(P(H_{q,k}\mid wrong)\)：错误 rollout 中该 span 的 hidden-state 分布；
\(u_{q,k}\)：这个 span 的“正确/错误分布差异”。
如果 \(u_{q,k}\) 很小，表示：
正确和错误 rollout 在这个局部 span 的 hidden states 仍然很相似。

如果 \(u_{q,k}\) 很大，表示：
这个局部 span 在 hidden-state 空间中已经明显区分了正确路径和错误路径。

因此可以把 \(u_{q,k}\) 看成：
hidden-state 分布的偏离程度；
局部 reasoning divergence；
潜在的错误发生区域；
需要加强 credit assignment 的位置。
但它本身不说明“正确方向是什么”，只说明两类轨迹在这里分开了。
5. SHEAR 中的实际计算更细一些
你给出的公式是在比较“正确组分布”和“错误组分布”，这是概念化写法。
SHEAR 实际上对每一条 rollout 的每个 span 单独计算。
对于第 \(i\) 条 rollout 的第 \(k\) 个 span：
\[
d_{i,k}
=
\min_{j\in O(i)}
\min_r
W_\varepsilon
\left(
\widehat P_{i,k},
\widehat P_{j,r}
\right)
\]这里：
\(O(i)\) 是与 rollout \(i\) 结果相反的 rollout 集合；
\(j\) 是 opposing rollout；
\(r\) 是 opposing rollout 中的 span；
对所有 opposing span 取最小距离。
为什么取最小值？
假设一题有三个正确推理路径：
正确路径 A：代数解法
正确路径 B：枚举解法
正确路径 C：几何解法
错误路径 D：前半段和 A 相似，后半段偏离
如果只拿错误路径 D 与正确路径 A 比较，可能会误判它整条都很远。但拿它和 A、B、C 全部比较，只要它仍然和某条正确路径相似，就保留较小距离。
因此：
\[
d_{i,k}\text{ 大}
\]意味着这个 span 和所有可用的相反结果路径都不相似，而不是只和某一条路径不相似。
6. 从 span 分数变成 token advantage
你的第二个公式是：
\[
A'_{i,t}=A_i\cdot w(u_{q,k(t)})
\]含义如下：
\(A_i\)：第 \(i\) 条 rollout 的 GRPO advantage；
\(t\)：该 rollout 中的第 \(t\) 个 token；
\(k(t)\)：覆盖 token \(t\) 的 span；
\(w(\cdot)\)：把 Wasserstein 距离转成 token 权重的函数；
\(A'_{i,t}\)：最终用于训练该 token 的 advantage。
SHEAR 的具体版本是：
\[
\omega_{i,t}
=
\frac{1}{\bar n}
\max_{k:t\in S_{i,k}}d_{i,k}
\]然后：
\[
\widetilde A_{i,t}
=
A_i\omega_{i,t}
\]为什么是 max pooling？
因为一个 token 可能落在多个重叠窗口中：
span 1: t1 ... t100
span 2: t26 ... t125
span 3: t51 ... t150
如果 token \(t=80\) 同时属于三个 span，就会有三个距离：
\[
d_{i,1},d_{i,2},d_{i,3}
\]SHEAR 取：
\[
\omega_{i,80}
=
\max(d_{i,1},d_{i,2},d_{i,3})/\bar n
\]这样可以保留最明显的局部 divergence 信号。
7. 正确和错误 rollout 中分别发生什么？
错误 rollout
错误 rollout 的 advantage 通常是：
\[
A_i<0
\]如果某个 token 的 Wasserstein 距离很大：
\[
\widetilde A_{i,t}
=
A_i\omega_{i,t}
\]会变得更加负。
例如：
\[
A_i=-0.5,\qquad
\omega_{i,t}=2
\]则：
\[
\widetilde A_{i,t}=-1.0
\]这个 token 的概率会被更强地压低。
正确 rollout
正确 rollout 的 advantage 通常是：
\[
A_i>0
\]如果某个 token 的 hidden-state 分布特别不同：
\[
A_i=0.5,\qquad
\omega_{i,t}=2
\]则：
\[
\widetilde A_{i,t}=1.0
\]模型会更强地强化这个局部模式。
所以 SHEAR 的核心不是简单地说：
Wasserstein 越大越好。

而是：
在正确 rollout 中，大 divergence 区域得到更强正更新；在错误 rollout 中，大 divergence 区域得到更强负更新。

8. 和普通 cosine/mean 距离的区别
如果只比较两个 span 的均值：
\[
\|\mu^+-\mu^-\|_2
\]会忽略：
分布方差；
多峰结构；
hidden-state 点云的形状；
一部分 token 是否已经移动而其他 token 没有移动。
Wasserstein 允许不同 hidden-state 点以不同运输方案匹配，因此更适合比较一个窗口内部的整体分布。
但它也有局限：
span 内 token 顺序被忽略；
高维 Wasserstein 计算昂贵；
最小匹配可能把不同推理阶段的 span 错配；
hidden-state norm 的尺度变化可能影响距离；
如果一个 group 全部正确或全部错误，就没有 opposing group。
9. 对你现有数据的一个关键影响
如果你的文件里只有：
\[
h_{\text{chunk}}
\in\mathbb R^{K\times L\times D}
\]也就是每个 chunk 的平均 hidden state，那么不能严格复现 SHEAR。
因为 SHEAR 需要一个 span 内的 token-level 点云：
\[
\{h_{t_1},h_{t_2},\ldots,h_{t_w}\}
\]而 chunk mean 只有：
\[
\bar h_{\text{chunk}}
=
\frac1w\sum_{t=1}^w h_t
\]这会丢掉：
span 内部方差；
多峰分布；
token-level divergence；
Wasserstein 的真正分布信息。
用 chunk mean 计算的只能叫：
\[
\text{chunk-level Wasserstein approximation}
\]不能直接称为原始 SHEAR。
另外，SHEAR 还需要同一个问题下多条正确和错误 rollout 的 hidden states，才能建立 opposing group。单条轨迹没有正确/错误对照时，不能直接计算这个信号。


星期日21:42






1 annotation





这个公式表示：把同一道题中所有正确 rollout 在“答案前的关键位置”上的 hidden state 拉得更接近。
\[
L_{\text{align}}
=
\frac{1}{|G^+|}
\sum_{i\in G^+}
d(h_i^{\text{anchor}},\bar h_+^{\text{anchor}})
\]它可以拆成四部分。
1. \(G^+\) 是什么？
\[
G^+=\{i:r_i=1\}
\]表示同一道题的一组 rollout 中，最终答案正确的那些 rollout。
例如，对同一个问题采样 8 条：
rollout 1: 正确
rollout 2: 错误
rollout 3: 正确
rollout 4: 错误
rollout 5: 正确
...
那么：
G+ = {rollout 1, rollout 3, rollout 5, ...}
这里必须是同一个问题内部的正确 rollout，不能把不同问题的 hidden state 混在一起平均。不同问题的语义和答案空间不同，直接对齐没有意义。
2. \(h_i^{\text{anchor}}\) 是什么？
它是第 \(i\) 条正确 rollout 在某个固定关键位置的 hidden state。
Hidden-Align 中的 anchor 通常是：
最终答案标记之前的最后一个 reasoning token 的 hidden state。

例如：
<think>
...
所以答案应该是 17
</think>
答案是：17
anchor 可以取 <think> 结束前，或者答案标记前的那个 token。
于是：
\[
h_i^{\text{anchor}}\in\mathbb R^d
\]表示第 \(i\) 条 rollout 在“已经形成最终答案、但尚未输出答案”的内部状态。
为什么选这里？
因为这个位置通常已经包含了：
对问题的最终判断；
推理结果的收敛状态；
答案选择信息；
但还没有太多最终答案格式、标点和文字差异。
所以它比整条轨迹的平均 hidden state 更适合作为“最终正确状态”。
3. \(\bar h_+^{\text{anchor}}\) 是什么？
它是所有正确 rollout anchor hidden state 的平均向量：
\[
\bar h_+^{\text{anchor}}
=
\frac{1}{|G^+|}
\sum_{i\in G^+}
h_i^{\text{anchor}}
\]假设有 3 条正确 rollout：
\[
h_1=(1,0)
\]\[
h_2=(0.8,0.2)
\]\[
h_3=(1.1,-0.1)
\]那么它们的中心是：
\[
\bar h_+
=
\frac{h_1+h_2+h_3}{3}
=
(0.97,0.03)
\]这就是该问题下“正确 rollout anchor 状态”的中心。
4. \(d(\cdot,\cdot)\) 是什么？
\(d\) 是 hidden-state 距离函数，常见选择有：
欧氏距离
\[
d(h_i,\bar h)=\|h_i-\bar h\|_2
\]表示向量位置距离中心有多远。
平方欧氏距离
\[
d(h_i,\bar h)=\|h_i-\bar h\|_2^2
\]通常梯度更平滑。
Cosine distance
\[
d(h_i,\bar h)
=
1-
\frac{
h_i^\top\bar h
}{
\|h_i\|_2\|\bar h\|_2
}
\]如果两个向量方向一致，cosine distance 接近 0；方向不同，距离更大。
因此：
\[
L_{\text{align}}
\]就是每个正确 rollout 到“正确中心状态”的平均距离。
训练时最小化它，就会让：
\[
h_i^{\text{anchor}}
\rightarrow
\bar h_+^{\text{anchor}}
\]也就是让正确 rollout 的最终 hidden state 更集中。
5. 这个 loss 的具体含义
如果：
\[
L_{\text{align}}\text{ 很小}
\]说明同一道题的多个正确 rollout 在答案前已经收敛到相似的 hidden-state 区域。
如果：
\[
L_{\text{align}}\text{ 很大}
\]说明虽然这些 rollout 最终都答对了，但它们的内部最终状态差异较大。
所以它不是在判断某一条 rollout 是否正确，而是在约束：
同一道题的多条正确推理，最终应该在某个 anchor 位置形成更稳定的 hidden-state 几何结构。

6. 为什么不直接对齐错误 rollout？
因为错误 rollout 可能有很多种不同的错误方式：
错误路径 A：代数步骤错
错误路径 B：计算错
错误路径 C：误读题意
错误路径 D：答案格式错
如果把所有错误 rollout 也压到一起，可能会得到一个没有明确意义的“错误中心”。
Hidden-Align 的思路是：
正确 rollout：应该在最终状态上适度收敛；
早期 reasoning：应该保留多样性；
错误 rollout：不强行构造一个错误 manifold。
因此它通常只对正确组的 anchor hidden state 做 alignment。
7. 它和 pairwise cosine 的关系
你看到的中心形式是：
\[
\frac1{|G^+|}
\sum_i d(h_i,\bar h_+)
\]而 Hidden-Align 论文的实际实现更接近正确 rollout 之间的 pairwise cosine alignment，例如：
\[
L_{\text{pair}}
=
-\frac{1}{|G^+|(|G^+|-1)}
\sum_{i\neq j}
\cos(h_i,h_j)
\]这里加负号是因为训练目标要最小化 loss，而我们希望 cosine 越大越好。
对于平方欧氏距离，中心形式和 pairwise dispersion 有严格关系：
\[
\sum_i\|h_i-\bar h\|_2^2
=
\frac{1}{2|G^+|}
\sum_{i,j}\|h_i-h_j\|_2^2
\]所以“靠近中心”和“彼此靠近”在平方欧氏距离下本质上是等价的。
但是对于 cosine distance，这两个形式不完全等价，因此：
中心形式更容易解释；
pairwise cosine 更贴近 Hidden-Align 的实际实现；
不应该把两者当成完全相同的公式。
8. 一个数值例子
假设两个正确 rollout 的 anchor cosine 相似度为：
\[
\cos(h_1,h_2)=0.6
\]则 cosine alignment loss 可以看成：
\[
1-0.6=0.4
\]训练后相似度提高到：
\[
\cos(h_1,h_2)=0.9
\]则 loss 变成：
\[
1-0.9=0.1
\]模型会被鼓励让正确 rollout 的最终状态更相似。
如果总损失是：
\[
L
=
L_{\text{DAPO}}
+
\lambda L_{\text{align}}
\]其中 \(\lambda\) 很小，例如 \(0.001\)，那么：
\(L_{\text{DAPO}}\) 仍然负责主要的 RL 优化；
\(L_{\text{align}}\) 只提供一个 hidden-state 几何约束；
它不会完全替代 outcome reward。
9. 它和 SHEAR 的区别
这个 alignment loss 测的是：
\[
\text{正确 rollout 彼此是否收敛}
\]SHEAR 测的是：
\[
\text{当前 span 与 opposing rollout 的 hidden-state 分布有多远}
\]因此：
Hidden-Align 是“正确状态之间的收敛约束”；
SHEAR 是“正确/错误状态之间的局部分离信号”；
Hidden-Align 不直接告诉你错误发生在哪个 token；
SHEAR 可以生成 token-level advantage weight。
另外，如果一组 rollout 只有一个正确样本：
\[
|G^+|=1
\]那么 alignment 没有有效的组内比较；如果全部错误，也无法计算正确中心。这是它依赖 mixed group 的地方。


星期日21:44






1 annotation





这两个公式的核心思想是：
把“当前推理状态离正确推理轨迹有多远”变成一个分数；当模型生成下





Reconnecting /5






这两个公式想表达的是：
先用 hidden state 到“正确推理区域”的距离，定义模型当前推理状态的好坏；然后只奖励模型是否从当前一步向正确区域靠近，而不是直接奖励某个 hidden state 数值。

先看第一个公式：
\[
\Phi(s_t)=-d(h_t,\mathcal M_t^+)
\]各个符号如下。
1. \(t\)：当前生成位置
\(t\) 表示模型已经生成到第几个 token 或第几个 reasoning span。
例如：
t=0：只有题目
t=1：Let us analyze...
t=2：First, we know...
t=3：Therefore x=...
实际实现中，最好把 \(t\) 定义成语义 span 或归一化推理进度，而不是单个 token，因为不同 rollout 长度不同。
2. \(s_t\)：当前 RL 状态
在语言模型中，状态 \(s_t\) 通常就是：
\[
s_t=(x,y_1,\ldots,y_t)
\]其中：
\(x\)：题目或 prompt；
\(y_1,\ldots,y_t\)：当前已经生成的 token；
\(s_t\)：题目加目前为止的完整生成前缀。
例如：
题目：求方程 x+3=7
当前前缀：We subtract 3 from both sides
这整个上下文就是 \(s_t\)。
模型接下来选择动作：
\[
a_t=y_{t+1}
\]即生成下一个 token。生成后状态从 \(s_t\) 变为 \(s_{t+1}\)。
3. \(h_t\)：当前状态的 hidden representation
[
h_t=f_{\text{hidden}}(s




Reconnecting /5






这两个公式的目的，是把“当前 hidden state 是否正在靠近正确推理区域”变成每一步都能获得的 RL 奖励。
每个符号
\[
\Phi(s_t)=-d(h_t,\mathcal M_t^+)
\]符号	含义
\(t\)	当前生成步骤或 reasoning span
\(s_t\)	RL 状态，即题目和当前已生成前缀
\(h_t\)	状态 \(s_t\) 对应的 hidden-state 表示
\(\mathcal M_t^+\)	正确 rollout 在相同推理进度上的 hidden-state 集合
\(d(\cdot,\cdot)\)	当前 hidden state 到正确集合的距离
\(\Phi(s_t)\)	状态的 potential，即这个状态有多接近正确区域
\(r_t\)	原始奖励，例如最终答案是否正确
\(r'_t\)	加入 hidden-state shaping 后的新奖励
\(\gamma\)	discount factor，通常接近 1

1. 状态 \(s_t\) 是什么？
在语言模型 RL 中：
\[
s_t=(x,y_1,\ldots,y_t)
\]其中：
\(x\) 是题目；
\(y_1,\ldots,y_t\) 是当前已经生成的 token；
下一步动作是生成 \(y_{t+1}\)。
例如：
题目：求方程 x+3=7
当前前缀：移项可得 x =
整个“题目 + 当前前缀”就是状态 \(s_t\)。
生成下一个 token 4 后，得到：
\[
s_{t+1}=(x,y_1,\ldots,y_t,y_{t+1})
\]2. \(h_t\) 是什么？
\[
h_t=f(s_t)
\]表示模型读取当前题目和推理前缀后得到的 hidden-state 向量。
它可以是：
最后一层、当前 token 的 hidden state；
当前 reasoning span 的平均 hidden state；
多层 hidden state 的组合；
anchor token 的 hidden state。
你的实验如果按 chunk 分割，那么更实际的定义是：
\[
h_t=\operatorname{Pool}(H_{\text{chunk }t})
\]此时 \(t\) 表示第几个 reasoning chunk，而不一定是单个 token。
3. \(\mathcal M_t^+\) 是什么？
\(\mathcal M_t^+\) 读作“第 \(t\) 个进度位置上的 success manifold”。
假设同一道题有多条正确 rollout：
正确轨迹1：建立方程 → 移项 → 计算
正确轨迹2：代入选项 → 排除 → 验证
正确轨迹3：化简条件 → 求解 → 检查
提取它们在相近推理进度上的 hidden states：
\[
\mathcal M_t^+
=
\{h_{1,t}^+,h_{2,t}^+,\ldots,h_{N,t}^+\}
\]它不一定是真正严格的数学流形，也可以实现成：
正确 hidden-state 样本库；
若干聚类中心；
一个低维子空间；
高斯分布；
正确 rollout 的局部轨迹集合。
之所以写成 \(\mathcal M_t^+\)，是为了表达：
正确推理可能有多条不同路径，不应该只有一个正确中心点。

下标 \(t\) 很重要。推理开头的 hidden state 应该和正确轨迹开头比较，推理后期应和正确轨迹后期比较，不能把不同进度直接混在一起。
4. 距离 \(d(h_t,\mathcal M_t^+)\)
最简单的定义是最近邻距离：
\[
d(h_t,\mathcal M_t^+)
=
\min_{z\in\mathcal M_t^+}
\|h_t-z\|_2
\]也可以用 cosine distance：
\[
d(h_t,\mathcal M_t^+)
=
\min_{z\in\mathcal M_t^+}
\left(1-\cos(h_t,z)\right)
\]如果表示的是一个 span 分布，还可以使用 Wasserstein：
\[
d(h_t,\mathcal M_t^+)
=
\min_{P^+\in\mathcal M_t^+}
W_1(P_t,P^+)
\]距离越小，表示当前状态越接近某种已知的正确推理状态。
5. 为什么 potential 要加负号？
定义：
\[
\Phi(s_t)=-d(h_t,\mathcal M_t^+)
\]假设距离分别为：
状态	距离 \(d\)	potential \(\Phi=-d\)
很远	8	-8
较近	3	-3
位于正确区域	0	0

所以：
\[
\Phi(s_t)\text{ 越大}
\]表示状态越接近正确区域。potential 的最大值是 0。
6. shaping reward 是什么？
\[
r'_t
=
r_t+
\gamma\Phi(s_{t+1})-\Phi(s_t)
\]新增的 hidden-state 奖励是：
\[
F_t=
\gamma\Phi(s_{t+1})-\Phi(s_t)
\]如果先取 \(\gamma=1\)，代入 \(\Phi=-d\)：
\[
F_t
=
-d_{t+1}+d_t
=
d_t-d_{t+1}
\]因此它实际测量的是：
生成下一个 token 后，到正确 hidden-state 区域的距离减少了多少。

靠近正确区域
假设：
\[
d_t=5,\qquad d_{t+1}=3
\]那么：
\[
F_t=5-3=2
\]如果原始中间奖励 \(r_t=0\)，则：
\[
r'_t=2
\]这一步获得正奖励。
远离正确区域
假设：
\[
d_t=5,\qquad d_{t+1}=7
\]那么：
\[
F_t=5-7=-2
\]这一步获得负奖励。
没有明显变化
\[
d_t=5,\qquad d_{t+1}=5
\]则：
\[
F_t=0
\]所以它奖励的不是“当前距离近”，而是“这一步让推理变得更接近正确区域”。
7. 原始奖励 \(r_t\)
在普通 RLVR 中，中间 token 通常没有奖励：
\[
r_t=0,\qquad t<T
\]最终答案处才有：
\[
r_T=
\begin{cases}
1,&\text{答案正确}\\
0,&\text{答案错误}
\end{cases}
\]potential shaping 将稀疏的最终奖励重新分配到过程：
生成步骤      原始奖励       hidden-state变化       新奖励
步骤1            0           靠近正确区域            正
步骤2            0           靠近正确区域            正
步骤3            0           远离正确区域            负
最终答案          1           最终状态变化            1 + shaping
这就是它可能改善 credit assignment 的原因。
8. 为什么不会简单改变原任务？
把整条轨迹的 shaping reward 加起来：
\[
\sum_{t=0}^{T-1}
\gamma^t
\left[
\gamma\Phi(s_{t+1})-\Phi(s_t)
\right]
=
-\Phi(s_0)+\gamma^T\Phi(s_T)
\]中间 potential 会两两抵消。
如果：
所有 rollout 从同一个初始状态开始；
terminal potential 被设置成相同值，例如 \(\Phi(s_T)=0\)；
discount 和 reward 定义保持一致；
那么总 shaping return 只相差一个与动作无关的常数。理论上它不会改变最优策略，只会重新分配学习信号。
对于语言模型这种有限长度任务，最简单的实现通常是：
\[
\gamma=1,\qquad \Phi(s_{\text{terminal}})=0
\]但终止状态和截断状态必须统一处理，否则“保持最优策略”的保证可能不成立。
9. “Frozen potential”是什么意思？
如果用正在训练的 actor 自己产生 hidden state 和 success manifold，那么模型参数每次更新后：
\(h_t\) 在变化；
\(\mathcal M_t^+\) 也在变化；
奖励标准本身不断漂移；
actor 可能学会操纵 hidden norm 来获取奖励。
更稳定的方案是使用冻结的 reference model：
\[
h_t^{\text{ref}}
=
f_{\text{ref}}(s_t)
\]并提前用它建立：
\[
\mathcal M_t^{+,\text{ref}}
\]训练时，actor 生成前缀，但把前缀输入冻结模型计算距离：
\[
\Phi(s_t)
=
-d\left(
h_t^{\text{ref}},
\mathcal M_t^{+,\text{ref}}
\right)
\]这样几何坐标系不会随 actor 更新而变化，所以叫 frozen potential。
它不是说 actor 被冻结，而是说“用于评分的 hidden-state 编码器和正确 manifold”被冻结。
10. 对你实验的实际含义
你现在最需要先验证的不是马上做 RL，而是：
\[
d_t-d_{t+1}
\]是否真的和 continuation success 的变化相关。




Reconnecting /5






Worked for 23m 24s




这两个公式的目标是：把“hidden state 正在靠近正确推理区域”转化成每一步都能使用的稠密 RL 奖励。
先把所有符号展开。
第一个公式
\[
\Phi(s_t)=-d(h_t,\mathcal M_t^+)
\]符号	含义
\(t\)	当前生成到第几个 token 或第几个 reasoning span
\(s_t\)	当前 RL 状态，即“题目 + 已生成的推理前缀”
\(h_t\)	模型对当前状态 \(s_t\) 计算出的 hidden state
\(\mathcal M_t^+\)	在相同推理进度上，正确轨迹形成的









stream disconnected before completion: Transport error: network error: error decoding response body






1 annotation





Worked for 5m 51s



这个公式本身不是在计算均值，而是在把一个 span 内的每个 token hidden state 都当成一个“带质量的点”：
\[
\widehat P_{i,k}
=
\frac1{|S_k|}
\sum_{t\in S_k}\delta_{h_{i,t}}
\]1. 每个符号是什么
假设第 \(i\) 条 rollout 的第 \(k\) 个 span 是：
\[
S_{i,k}=\{101,102,\ldots,200\}
\]包含 100 个 token。
每个 token 的最后一层 hidden state 是：
\[
h_{i,t}\in\mathbb R^d
\]那么整个 span 的 hidden-state 矩阵为：
\[
H_{i,k}
=
\begin{bmatrix}
h_{i,101}\\
h_{i,102}\\
\vdots\\
h_{i,200}
\end{bmatrix}
\in\mathbb R^{100\times d}
\]如果模型 hidden dimension 是 4096，就是：
\[
H_{i,k}\in\mathbb R^{100\times4096}
\]\(\delta_{h_{i,t}}\) 叫 Dirac point mass，意思是在向量 \(h_{i,t}\) 的位置放置一个质量点。
因为一共有 100 个 token，每个点的质量是：
\[
\frac1{100}
\]全部质量相加等于 1，所以形成一个经验概率分布。
2. 计算时不需要显式构造 \(\delta\)
代码中不会真的创建“Dirac 分布对象”。实际只保存：
hidden_states.shape == [num_tokens, hidden_dim]
weights.shape == [num_tokens]
例如：
x = hidden_i[101:201]  # [100, 4096]
a = torch.full((100,), 1.0 / 100)
x 是分布中的 100 个点，a 表示每个点的概率质量都是 \(1/100\)。
另一个错误或正确 span：
y = hidden_j[76:176]   # [100, 4096]
b = torch.full((100,), 1.0 / 100)
现在需要比较：
\[
\widehat P_X
=
\frac1{100}\sum_{a=1}^{100}\delta_{x_a}
\]和：
\[
\widehat P_Y
=
\frac1{100}\sum_{b=1}^{100}\delta_{y_b}
\]3. 先计算 token 两两之间的距离
构造一个 \(100\times100\) 的 cost matrix：
\[
C_{ab}
=
\|x_a-y_b\|_2
\]其中：
第 \(a\) 行对应第一个 span 的第 \(a\) 个 token；
第 \(b\) 列对应第二个 span 的第 \(b\) 个 token；
\(C_{ab}\) 表示两个 token hidden state 相距多远。
代码近似是：
cost = torch.cdist(x, y, p=2)  # [100, 100]
例如只有三个 token：
\[
X=\{x_1,x_2,x_3\},\qquad
Y=\{y_1,y_2,y_3\}
\]得到：
\[
C=
\begin{bmatrix}
\|x_1-y_1\| & \|x_1-y_2\| & \|x_1-y_3\|\\
\|x_2-y_1\| & \|x_2-y_2\| & \|x_2-y_3\|\\
\|x_3-y_1\| & \|x_3-y_2\| & \|x_3-y_3\|
\end{bmatrix}
\]4. 再寻找最小运输方案
定义运输矩阵：
\[
\Pi\in\mathbb R_{\geq0}^{n\times m}
\]其中 \(\Pi_{ab}\) 表示：从 \(x_a\) 向 \(y_b\) 搬运多少概率质量。
它必须满足：
\[
\sum_b\Pi_{ab}=\frac1n
\]\[
\sum_a\Pi_{ab}=\frac1m
\]即第一个分布中每个点的质量都必须被运走，第二个分布中每个点也必须接收到对应质量。
Wasserstein-1 距离是：
\[
W_1(P_X,P_Y)
=
\min_{\Pi}
\sum_{a=1}^{n}\sum_{b=1}^{m}
\Pi_{ab}C_{ab}
\]也就是找到总运输成本最小的匹配方式。
5. 一个简单的一维例子
假设两个 span 的 hidden state 简化成一维：
\[
X=\{0,2\}
\]\[
Y=\{1,5\}
\]每个点质量都是 \(1/2\)。
距离矩阵：
\[
C=
\begin{bmatrix}
|0-1| & |0-5|\\
|2-1| & |2-5|
\end{bmatrix}
=
\begin{bmatrix}
1&5\\
1&3
\end{bmatrix}
\]最优方案是：
把 \(0\) 匹配到 \(1\)，成本为 1；
把 \(2\) 匹配到 \(5\)，成本为 3。
因为每个点质量为 \(1/2\)：
\[
W_1
=
\frac12\times1+\frac12\times3
=
2
\]交叉匹配的成本是：
\[
\frac12\times5+\frac12\times1=3
\]所以不会选择交叉方案。
6. 为什么它不等于比较均值？
考虑两个分布：
\[
X=\{-1,1\}
\]\[
Y=\{0,0\}
\]它们的均值都是 0：
\[
\mu_X=\mu_Y=0
\]所以比较均值会得到：
\[
\|\mu_X-\mu_Y\|=0
\]但把 \(-1\) 和 \(1\) 都运输到 0，需要成本：
\[
W_1(X,Y)
=
\frac12|-1-0|
+
\frac12|1-0|
=
1
\]因此 Wasserstein 可以发现：
两个分布虽然均值相同；
但第一个分布更分散；
第二个分布集中在中心。
这就是“保留分布形状”的具体含义。
7. Sinkhorn 实际怎么算
精确求最优运输成本比较慢，因此 SHEAR 使用带熵正则的 Sinkhorn 近似：
\[
W_\varepsilon(P_X,P_Y)
=
\min_{\Pi}
\left[
\sum_{a,b}\Pi_{ab}C_{ab}
+
\varepsilon
\operatorname{KL}(\Pi\|ab^\top)
\right]
\]构造：
\[
K_{ab}=\exp(-C_{ab}/\varepsilon)
\]然后迭代更新缩放向量：
\[
u\leftarrow a/(Kv)
\]\[
v\leftarrow b/(K^\top u)
\]最后：
\[
\Pi=\operatorname{diag}(u)K\operatorname{diag}(v)
\]距离为：
\[
W_\varepsilon
=
\sum_{a,b}\Pi_{ab}C_{ab}
\]实际实现通常使用 log-domain Sinkhorn，避免高维距离导致指数下溢。
8. 为什么说它是无序点集？
假设原 span 顺序是：
\[
[h_1,h_2,h_3]
\]打乱成：
\[
[h_3,h_1,h_2]
\]经验分布仍然是：
\[
\frac13(\delta_{h_1}+\delta_{h_2}+\delta_{h_3})
\]所以 Wasserstein 距离完全不变。
它保存了：
hidden states 位于什么区域；
分布是否集中或分散；
是否存在多个簇；
正确与错误 span 的整体几何差异。
但它没有保存：
哪个 token 先出现；
推理方向；
\(h_1\to h_2\to h_3\) 的轨迹顺序。
SHEAR 通过滑动窗口的位置和重叠间接保留局部顺序，但窗口内部仍然被视为无序分布。
如果希望显式考虑 token 位置，可以修改 cost：
\[
C_{ab}
=
\|x_a-y_b\|_2
+
\lambda
\left|
\frac an-\frac bm
\right|
\]这样不仅要求 hidden state 相似，还会惩罚相距很远的 token 位置。不过这已经不是原始 SHEAR。
9. SHEAR 中完整的计算流程
对于当前 rollout 的 span \((i,k)\)：
取 hidden matrix \(X_{i,k}\in\mathbb R^{w\times d}\)。
找到所有相反结果 rollout。
提取每条相反 rollout 的所有 span。
对每一对 span 构造 \(w\times w\) cost matrix。
用 Sinkhorn 求 Wasserstein 距离。
对所有 opposing spans 取最小值：
\[
d_{i,k}
=
\min_{j\in O(i),r}
W_\varepsilon(
\widehat P_{i,k},
\widehat P_{j,r}
)
\]把 span 距离通过 max pooling 分配回 token：
\[
\omega_{i,t}
=
\frac1{\bar n}
\max_{k:t\in S_{i,k}}d_{i,k}
\]你本地的 chunk mean 数据只有每个 chunk 的一个平均向量，无法构造这里的 \(w\times w\) cost matrix。因此，要严格计算 span Wasserstein，需要保存每个 token 的最后一层 hidden state；不需要保存全部层。


12:23





















Full access

5.6 SolExtra High








Work locallyLocal