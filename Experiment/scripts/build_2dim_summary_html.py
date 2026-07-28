#!/usr/bin/env python3
"""Build a self-contained HTML summary of 2Dimension-02 / 03 experiments.

Figures are embedded as base64 so the HTML is fully portable.
Read-only over result artifacts; writes a single HTML file.
"""
import base64
from pathlib import Path

ROOT = Path(r"C:/Users/LENOVO/Desktop/A-G实验/AI-HiddenState/Experiment")
FIG_DIR = ROOT / "server_results/experiment02_replication96_20260723/results/figures"
OUT = ROOT / "server_results/2DIMENSION_02_03_SUMMARY.html"

FIGS = {
    "F1": ("E02_F1_progress_spaghetti.png",
           "E02_F1 · Progress spaghetti",
           "750 条 rollout 的 movement / activity 透明曲线，叠加 correct / wrong 均值。两组几乎重合，直观看出冻结信号没有复现。"),
    "F2": ("E02_F2_confirmatory_auc_curves.png",
           "E02_F2 · Confirmatory AUC curves",
           "movement / activity 在 bin0–9 的 within-Q AUC。红点是冻结的 bin5 / bin9；两者 95% CI 均跨 0.5。"),
    "F3": ("E02_F3_entropy_layer_progress.png",
           "E02_F3 · Entropy layer×progress heatmaps",
           "raw / z-score / update entropy 的 layer × progress correct-minus-wrong 热图。最重要发现是 raw entropy 在 L14–L19 / bin6 形成连续正 band。"),
    "F4": ("E02_F4_path_geometry_auc.png",
           "E02_F4 · Path geometry AUC",
           "L24 / L36 上 path length、net displacement、straightness、log-detour 随 progress 的 AUC。net displacement 在 bin6、straightness 在中后段有局部信号。"),
}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def fig_block(key: str) -> str:
    fname, title, cap = FIGS[key]
    p = FIG_DIR / fname
    if not p.exists():
        return f'<div class="fig missing"><h4>{title}</h4><p class="cap">[文件缺失: {fname}]</p></div>'
    data = b64(p)
    return (
        f'<figure class="fig">'
        f'<figcaption class="fig-title">{title}</figcaption>'
        f'<img src="data:image/png;base64,{data}" alt="{title}"/>'
        f'<figcaption class="cap">{cap}</figcaption>'
        f'</figure>'
    )


CSS = """
:root{--bg:#0f1115;--card:#171a21;--card2:#1e222b;--ink:#e6e9ef;--mut:#9aa3b2;
--line:#2a2f3a;--accent:#6ea8fe;--good:#4ec9a5;--bad:#f2708a;--warn:#e6c07b;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue","PingFang SC","Microsoft YaHei",sans-serif;
line-height:1.65;font-size:15px}
.wrap{max-width:960px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:28px;margin:0 0 6px;letter-spacing:.3px}
h2{font-size:22px;margin:44px 0 10px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:17px;margin:26px 0 8px;color:var(--accent)}
h4{margin:0 0 4px;font-size:14px}
p{margin:8px 0}
.sub{color:var(--mut);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;margin:16px 0}
.flow{background:linear-gradient(135deg,#1b2330,#171a21);border-left:3px solid var(--accent)}
.mut{color:var(--mut)}
code{background:var(--card2);padding:1px 6px;border-radius:5px;font-size:13px;
font-family:"SF Mono",Consolas,monospace}
.eq{background:var(--card2);border:1px solid var(--line);border-radius:8px;
padding:10px 14px;margin:10px 0;font-family:"SF Mono",Consolas,monospace;
font-size:13px;color:#cbd3e1;overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:var(--card2);color:var(--mut);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.pass{color:var(--good);font-weight:600}
.fail{color:var(--bad);font-weight:600}
.warn{color:var(--warn);font-weight:600}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;
border:1px solid var(--line);margin-left:6px;vertical-align:middle}
.tag.fail{background:rgba(242,112,138,.12);border-color:var(--bad);color:var(--bad)}
.tag.new{background:rgba(78,201,165,.12);border-color:var(--good);color:var(--good)}
.tag.run{background:rgba(230,192,123,.12);border-color:var(--warn);color:var(--warn)}
figure.fig{margin:18px 0;background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:14px}
.fig img{width:100%;height:auto;border-radius:8px;background:#fff}
.fig-title{font-weight:600;font-size:14px;margin-bottom:8px;color:var(--ink)}
.cap{color:var(--mut);font-size:12.5px;margin-top:8px}
.fig.missing{border-style:dashed;color:var(--mut)}
.concl{background:rgba(110,168,254,.07);border-left:3px solid var(--accent);
padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0}
.concl b{color:var(--accent)}
ul{margin:8px 0;padding-left:22px}
li{margin:4px 0}
.pill{display:inline-block;font-size:12px;color:var(--mut);border:1px solid var(--line);
border-radius:20px;padding:2px 10px;margin:2px 4px 2px 0}
hr{border:none;border-top:1px solid var(--line);margin:30px 0}
a{color:var(--accent);text-decoration:none}
"""

HEAD = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2Dimension-02 / 03 实验总结</title><style>{CSS}</style></head><body><div class="wrap">
<h1>2Dimension-02 / 03 实验总结</h1>
<p class="sub">Hidden-state 内部指标用于评估 rollout 质量 · Qwen3-VL-8B-Thinking · MathVerse long-response</p>
"""

OVERVIEW = """
<div class="card flow">
<h3 style="margin-top:0">整体脉络</h3>
<p><b>2Dimension-02</b> 用 96 道新题复现 movement / activity，同时探索 entropy 与路径几何。
两个旧信号<span class="fail">复现失败</span>，但发现了新的 <span class="pass">raw activation entropy 中层信号</span>。</p>
<p><b>2Dimension-03</b> 用 120 道<b>完全独立</b>的新题，对这个 entropy 信号做严格冻结复现，
<span class="warn">目前仍在生成数据</span>。</p>
<p class="mut" style="margin-bottom:0">一句话：movement / activity 原始候选在独立题上失败；路径几何只有弱局部趋势；
raw activation entropy 的 L14–L19 / bin6 band 是目前<b>唯一</b>同时具备较强 AUC、跨层连续性和部分长度增量证据的信号，
但仍属 exploratory。03 正在回答它在全新 120 题上能否真正复现。</p>
</div>
"""

E02_SETUP = """
<h2>2Dimension-02 · 复现 + 探索</h2>
<div class="card">
<h3 style="margin-top:0">实验设置</h3>
<span class="pill">Qwen3-VL-8B-Thinking</span><span class="pill">MathVerse long response</span>
<span class="pill">max_tokens = 16384</span><span class="pill">rollout / 题 = 8</span>
<ul>
<li>96 道 <code>≥2 correct + ≥2 wrong</code> 的新题，共 <b>750</b> 条有效 rollout</li>
<li>其中 <b>45</b> 道满足严格 <code>3 correct + 3 wrong</code></li>
<li>thinking 段均分为 10 个 relative-progress bins</li>
</ul>
<p class="mut">计划见 2dimension02.md，正式报告见 EXPERIMENT02_RESULTS.md。</p>
</div>

<h3>1 · 横向 movement 复现 <span class="tag fail">未复现</span></h3>
<p>在 L24 使用 <code>mean_w128_s64</code> span pooling，冻结检验 bin5：</p>
<div class="eq">M_{i,b} = median_{k∈b}  ‖ g^24_{i,k} − g^24_{i,k−1} ‖₂</div>
<table>
<tr><th>cohort</th><th>AUC</th><th>95% CI</th><th>参考</th></tr>
<tr><td>Discovery24（旧）</td><td class="n">≈0.591</td><td>—</td><td class="mut">原始观察</td></tr>
<tr><td>新 96 题 (main)</td><td class="n fail">0.4676</td><td>[0.4189, 0.5144]</td><td class="mut">点估计反向</td></tr>
<tr><td>严格 3+3</td><td class="n fail">0.4516</td><td>—</td><td class="mut">同向偏低</td></tr>
</table>
<div class="concl">movement + length AUC 0.5425 &lt; length-only 0.5573。
<b>结论：</b>bin5 的 movement 信号没有复现，点估计甚至反向，不能进入 RL。</div>

<h3>2 · 纵向 activity burst 复现 <span class="tag fail">未复现</span></h3>
<p>计算 L14→L15 的跨层更新，冻结检验 bin9：</p>
<div class="eq">A_{i,b} = Q_0.9 { ‖ h^15_{i,t} − h^14_{i,t} ‖₂ : t∈b }</div>
<table>
<tr><th>cohort</th><th>AUC</th><th>95% CI</th></tr>
<tr><td>Discovery（旧）</td><td class="n">≈0.661</td><td>—</td></tr>
<tr><td>新 96 题 (main)</td><td class="n fail">0.5188</td><td>[0.4674, 0.5701]</td></tr>
<tr><td>严格 3+3</td><td class="n fail">0.5204</td><td>—</td></tr>
</table>
<div class="concl">activity + length AUC 0.5488 &lt; length-only 0.5624。
<b>结论：</b>末段 activity 信号也没有复现；整体下降趋势稳定，但 correct / wrong 差异不稳定。</div>
"""

E02_ENTROPY = """
<h3>3 · Raw activation entropy <span class="tag new">最有希望</span></h3>
<p>对每个 token、layer 的 hidden coordinates 去均值后，将平方能量归一化再算 normalized coordinate-energy entropy：</p>
<div class="eq">H_raw(t,l) = − Σ_j p_{t,l,j} log p_{t,l,j} / log D</div>
<p>扫描所有 layer × progress 后，最连续的正向区域是 <b>L14–L19, bin6</b>（相对进程约 60–70%）。六层平均后：</p>
<table>
<tr><th>cohort</th><th>within-Q AUC</th><th>95% CI</th><th>正向题占比</th></tr>
<tr><td>新 96 题 (main)</td><td class="n pass">0.6275</td><td>≈[0.580, 0.674]</td><td class="n">68.8%</td></tr>
<tr><td>严格 3+3</td><td class="n pass">0.6108</td><td>—</td><td class="n">—</td></tr>
</table>
<p class="mut">length 增量：feature-only 比 length-only 高 0.0651（CI 下界略 &gt;0）；
feature+length 比 length-only 高 0.0395（CI [−0.0066, 0.0855]，下界略跨 0）。</p>
<div class="concl"><b>结论：</b>02 中最有希望的发现，但它来自 layer / bin 网格扫描，存在 winner's curse，
不能直接当作已确认结果——这正是 03 要独立复现的对象。</div>

<h3>4 · Z-score activation entropy <span class="tag fail">弱</span></h3>
<p>先在每条轨迹内部对每个 coordinate 沿时间 z-score，再算能量熵。最好区域约 L22 / bin6：AUC <b>0.5355</b>，CI ≈[0.486, 0.585]。</p>
<div class="concl"><b>结论：</b>z-score 版明显弱于 raw entropy。原始坐标尺度中可能确实包含有效信息，
而不只是 rogue dimensions 干扰——信号是<b>尺度依赖</b>的，含义提示 RL 应使用 raw 而非 z-score。</div>

<h3>5 · Update entropy <span class="tag fail">碎</span></h3>
<p>对跨层更新向量 <code>u_{t,l} = h^l_t − h^{l−1}_t</code> 的箱内坐标能量算 entropy。
L14 / bin6 等局部 cell 的 AUC 可达 ≈0.613，但 layer 热图较碎，没有 raw entropy 那样连续的 L14–L19 band，
因此未被选为下一轮主候选。</p>

<h3>6 · 路径几何 <span class="tag fail">弱 exploratory</span></h3>
<p>对 span movement <code>d_k = g_k − g_{k−1}</code> 计算：</p>
<div class="eq">L = Σ_k‖d_k‖,   N = ‖Σ_k d_k‖,   S = N/L,   D = log L − log N</div>
<p>分别对应 path length、net displacement、straightness、log-detour。探索结果：</p>
<ul>
<li>L36 / bin6 局部 net displacement AUC ≈ 0.600；L24 / bin6 ≈ 0.585；L36 / bin7 straightness ≈ 0.568</li>
<li>整条轨迹上，正确 rollout 倾向于 path 较短、straightness 稍高、detour 稍低</li>
<li>但所有 whole-trajectory path 特征均<b>未通过</b>相对 think length 的增量 gate</li>
</ul>
<div class="concl"><b>结论：</b>中段净位移有局部迹象，但路径几何目前只是弱 exploratory signal，尚未超越回答长度。
（注：与旧 EXPERIMENT_REPORT 一致——path 几何在 long-CoT / thinking 模型上退化为 think 长度代理。）</div>
"""

E02_FIGS_HEADER = """
<h3>2Dimension-02 · 正式输出图像</h3>
<p class="mut">正式结果目录实际只有以下四张图；其余结果保存在 CSV / parquet 中。</p>
"""

E03 = """
<h2>2Dimension-03 · 严格独立复现 <span class="tag run">生成中</span></h2>
<div class="card">
<p>03 不是新一轮扫描，而是对 02 中 raw entropy band 的<b>严格独立复现</b>。</p>
<h3 style="margin-top:6px">冻结指标</h3>
<div class="eq">E_i = (1/6) Σ_{l=14}^{19} mean_{t∈bin6} H_raw(i,t,l)</div>
<ul>
<li>固定 L14–L19；bin6（thinking 相对进程约 60–70%）</li>
<li>方向冻结：entropy <b>越高越预测 correct</b></li>
<li>不查看其他 layer / bin；不根据新数据改变方向</li>
</ul>
<h3>数据</h3>
<ul>
<li>完全排除旧 Thinking 500 题</li>
<li>先生成 600 道新题，每题 rollout 8</li>
<li>从冻结顺序中取最先满足 <code>2 correct + 2 wrong</code> 的 120 题；不足时才用预冻结的 50 题备用批次</li>
</ul>
<h3>两级判据</h3>
<ul>
<li>① entropy within-Q AUC 的 bootstrap CI 下界 <code>&gt; 0.5</code></li>
<li>② entropy + think_length 相对 think_length-only 的 OOF AUC 增量 CI 下界 <code>&gt; 0</code></li>
</ul>
<p class="mut">只有两级都通过，才进入 RL credit-assignment 实验。</p>
</div>

<div class="card">
<h3 style="margin-top:0">当前进度 <span class="tag run">rollout 生成阶段</span></h3>
<ul>
<li>tmux <code>entropy_band_confirm120</code> 正常运行</li>
<li>已生成 <b>69 / 600</b> 题，<b>552</b> 条 rollout，约 <b>179 万</b>输出 token</li>
<li>12 条达到长度上限，后续会被排除</li>
<li>尚未做正式 cohort 选择、hidden forward 或 entropy 统计</li>
</ul>
<p class="mut">预计总耗时约 10–18 小时（约 7 月 24 日 07:00–15:00）。因此 03 <b>暂无</b>正式结果与可解释图像；
两道旧题 smoke 曾生成 E03 图，但只用于验证代码，不作为科学结果。</p>
<h3>正式完成后将产生</h3>
<ul>
<li><b>E03_F1_question_correct_wrong.png</b>：每题 correct entropy 均值 vs wrong entropy 均值，观察是否普遍位于对角线上方。</li>
<li><b>E03_F2_question_auc.png</b>：120 道题各自 within-Q AUC，及总体均值与 bootstrap CI。</li>
</ul>
</div>
"""

FOOT = """
<hr>
<p class="mut" style="font-size:12px">本页由 build_2dim_summary_html.py 生成，图片以 base64 内嵌，可离线打开 ·
数据来源：experiment02_replication96_20260723 结果目录 + 2dimension03 生成进度快照。</p>
</div></body></html>
"""


def main():
    parts = [
        HEAD, OVERVIEW,
        E02_SETUP, E02_ENTROPY,
        E02_FIGS_HEADER,
        fig_block("F1"), fig_block("F2"), fig_block("F3"), fig_block("F4"),
        E03, FOOT,
    ]
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
