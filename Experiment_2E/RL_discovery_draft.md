# RL Discovery Draft：16 个指标的计算代码表达

> 本文件只用代码形式表达 [RL_discovery_plan.md](RL_discovery_plan.md) 第 6/7 节的 16 个指标家族怎么算。
> **不是可运行的实验脚本**，没有 IO、没有模型 forward、没有 CLI。
> H1-H7/V1-V8 的输入是已经 pool 好的 numpy 数组；H8 直接接收 response-only token hidden states 并在线归约。
> 公式编号对应 plan 的 §6（H1–H8）与 §7（V1–V8）；代表量对应 §8 的 16 行表格。

## 约定

| 符号 | 代码变量 | 形状 | 含义 |
|---|---|---|---|
| \(x_{k,l}\) | `traj[k, l, :]` | `(K, L+1, D)` | 单条 rollout 单一 representation 的全部轨迹点 × 全部层 |
| \(x_k^{(l)}\) | `traj[:, l, :]` | `(K, D)` | anchor layer \(l\) 的横向轨迹 |
| \(y_{k,l}\) | `traj[k, :, :]` | `(L+1, D)` | 纵向用的单 chunk 层栈 |
| \(h_{t,l}\) | `token_hidden[t, l, :]` | `(T, L+1, D)` | H8 的 response-only token hidden state |
| \(p_k\) | `progress[k]` | `(K,)` | \(e_k/T\) |
| \(\epsilon\) | `EPS` | | 数值保护 |
| norm 阈值 | `NORM_FLOOR` | | 低于此值的位移/更新不进角度计算 |

- `L+1` 组 hidden states（含 embedding 层输出），`L, D` 运行时从 config 读，不硬编码。
- 所有函数返回 `(value, coverage_ok, n_valid)`；`coverage_ok=False` 时 `value=np.nan`，绝不返回 0 冒充。
- `hidden_states[0]` 是 embedding 输出。H1-H7 的横向 anchor 固定为 Transformer block 输出 `L3/L12/L24/Lfinal`；纵向用全部 `l = 0..L`。
- H8 final layer 是 primary；锚点层和全层 H8 profile 是探索性输出。H8 不使用 `mean/last` representation，而是只计算一次 token scalar。

---

## 0. 共享原语

```python
from __future__ import annotations
import numpy as np

EPS = 1e-12
NORM_FLOOR = 1e-6          # 冻结阈值，plan §6 / §5.3
Z_CLIP = 8.0               # plan §7 V7 robust-z
SIGMA_FLOOR = 1e-4         # plan §7 V7

STAGES = ((0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00))
HORIZONTAL_ANCHOR_LAYERS = (3, 12, 24, "final")
WINDOW = 128


def stage_mask(progress: np.ndarray, b: int) -> np.ndarray:
    """plan §5：B1=[0,.25) B2=[.25,.5) B3=[.5,.75) B4=[.75,1]，边界点只进一个 stage。"""
    lo, hi = STAGES[b]
    if b == len(STAGES) - 1:
        return (progress >= lo) & (progress <= hi)      # 末段闭区间
    return (progress >= lo) & (progress < hi)


def cum_index(progress: np.ndarray, b: int) -> int | None:
    """plan §5.2：k_b = max{k : p_k <= (b+1)/4}；不足则 None（记缺失，不插值）。"""
    hi = STAGES[b][1]
    idx = np.nonzero(progress <= hi)[0]
    return int(idx[-1]) if idx.size else None


def resolve_anchor_layers(n_hidden_states: int) -> tuple[int, ...]:
    """Return frozen block-output indices L3/L12/L24/Lfinal.

    Transformers exposes embedding output at index 0, so block output Lq is
    `hidden_states[q]`. `Lfinal = n_hidden_states - 1`.
    """
    final = n_hidden_states - 1
    anchors = (3, 12, 24, final)
    if final < 24 or len(set(anchors)) != len(anchors):
        raise ValueError("model does not provide distinct frozen L3/L12/L24/Lfinal anchors")
    return anchors


def horizontal_anchor_points(traj: np.ndarray, layer: int) -> np.ndarray:
    """`traj: (K, L+1, D)` -> anchor trajectory `(K, D)` for H1-H7."""
    if traj.ndim != 3 or not 0 <= layer < traj.shape[1]:
        raise ValueError("invalid horizontal anchor layer")
    return traj[:, layer, :]
```

```python
def displacements(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """plan §6：d_k = x_k - x_{k-1}，r_k = ||d_k||，u_k = d_k / (r_k + eps)。

    points: (n, D) -> d: (n-1, D), r: (n-1,), u: (n-1, D)
    """
    d = np.diff(points, axis=0)
    r = np.linalg.norm(d, axis=1)
    u = d / (r[:, None] + EPS)
    return d, r, u


def turn_angles(u: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, int]:
    """Return position-aligned H3 angles and the number excluded for low norm.

    `theta[i - 1]` is the turn from displacement `i - 1` to `i`, so it aligns
    exactly with the endpoint `progress[i + 1]`.  A low-norm displacement makes
    that turn undefined: retain its position as NaN rather than compressing the
    array.  Downstream stage aggregation must apply `np.isfinite(theta)`.
    """
    theta = np.full(max(len(u) - 1, 0), np.nan, dtype=float)
    ok = r >= NORM_FLOOR
    for i in range(1, len(u)):
        if ok[i] and ok[i - 1]:
            theta[i - 1] = np.arccos(
                np.clip(float(u[i] @ u[i - 1]), -1.0, 1.0)
            )
    return theta, int((~np.isfinite(theta)).sum())


def spectral_effective_rank(matrix: np.ndarray, *, center_rows: bool = False) -> float:
    """p_j = sigma_j / sum sigma, ER = exp(-sum p log p).

    `center_rows=True` subtracts the feature-wise row mean before SVD. H5 and
    V8 layer-state ER use it; raw layer-update ER deliberately does not.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        return np.nan
    if center_rows:
        values = values - values.mean(axis=0, keepdims=True)
    sigma = np.linalg.svd(values, compute_uv=False)
    sigma = sigma[sigma > EPS]
    if sigma.size == 0:
        return np.nan
    p = sigma / (sigma.sum() + EPS)
    return float(np.exp(-np.sum(p * np.log(p + EPS))))


def historical_deviation(m: np.ndarray) -> np.ndarray:
    """plan §6 H5：delta_j = m_j - (1/(j-1)) sum_{a<j} m_a，j >= 2（0-indexed j >= 1）。"""
    out = np.full(m.shape, np.nan, dtype=float)
    for j in range(1, len(m)):
        out[j] = m[j] - m[:j].mean()
    return out


def normalized_energy_entropy(vec: np.ndarray, center: bool, *, zero_value: float = np.nan) -> float:
    """plan §7 V6/V7：e_j = z_j^2（可选先减 coordinate mean），H = -sum p log p / log D。"""
    z = vec - vec.mean() if center else vec
    e = z ** 2
    total = e.sum()
    if total <= EPS:
        return zero_value
    p = e / (total + EPS)
    return float(-np.sum(p * np.log(p + EPS)) / np.log(len(vec)))


def unit_directions(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """plan §7 V4：v_{k,l} = d_{k,l} / (||d_{k,l}|| + eps)，同时返回 norm 用于阈值过滤。"""
    n = np.linalg.norm(d, axis=1)
    return d / (n[:, None] + EPS), n
```

---

## 1. 横向 8 个家族（plan §6）

H1-H7 对 `resolve_anchor_layers(traj.shape[1])` 返回的四个 layer 分别调用，输入为
`pts = horizontal_anchor_points(traj, layer)`。每一行输出必须携带 `anchor_layer=layer`，四层全部报告，不得事后只保留最显著层。H8 直接使用 `token_hidden`，不经过这里的 chunk representation 调度。

### H1. Movement magnitude — 代表量 `median relative movement`（local）

```python
def h1_movement(pts: np.ndarray, progress: np.ndarray, b: int) -> dict:
    """plan §6 H1：保存 raw r_k、per-coordinate RMS r_k/sqrt(D)、
    以及 relative r_k / (||x_{k-1}|| + eps)。

    coverage（§5.3）：至少 1 个有效位移。
    """
    d, r, _ = displacements(pts)
    r_rms = r / np.sqrt(pts.shape[1])                      # ||d_k|| / sqrt(D)
    base_norm = np.linalg.norm(pts[:-1], axis=1)          # ||x_{k-1}||
    r_rel = r / (base_norm + EPS)
    sel = stage_mask(progress[1:], b)                      # 位移归属其终点所在 stage
    if sel.sum() < 1:
        return {"coverage_ok": False, "n_valid": 0}
    return {
        "coverage_ok": True,
        "n_valid": int(sel.sum()),
        "median_relative_movement": float(np.median(r_rel[sel])),   # ← PRIMARY
        "mean_relative_movement": float(np.mean(r_rel[sel])),
        "p90_relative_movement": float(np.percentile(r_rel[sel], 90)),
        "median_rms_movement": float(np.median(r_rms[sel])),
        "mean_rms_movement": float(np.mean(r_rms[sel])),
        "p90_rms_movement": float(np.percentile(r_rms[sel], 90)),
        "median_raw_movement": float(np.median(r[sel])),
        "mean_raw_movement": float(np.mean(r[sel])),
        "p90_raw_movement": float(np.percentile(r[sel], 90)),
    }
```

### H2. Path geometry — 代表量 `straightness`（local + cumulative）

```python
def h2_path(d_subset: np.ndarray) -> dict:
    """plan §6 H2：L = sum ||d_k||，N = ||sum d_k||，S = N/(L+eps)，Dlog = log(L+eps) - log(N+eps)。

    d_subset 由调用方按 local stage 或 cumulative prefix 切好。
    coverage（§5.3 path）：至少 1 个有效位移；cumulative 边界不足则记缺失。
    """
    if d_subset.shape[0] < 1:
        return {"coverage_ok": False, "n_valid": 0}
    path_length = float(np.linalg.norm(d_subset, axis=1).sum())
    net_disp = float(np.linalg.norm(d_subset.sum(axis=0)))
    return {
        "coverage_ok": True,
        "n_valid": int(d_subset.shape[0]),
        "straightness": net_disp / (path_length + EPS),                      # ← PRIMARY
        "path_length": path_length,                                          # L, 必须同时报告
        "net_displacement": net_disp,                                        # N, 必须同时报告
        "log_detour": float(np.log(path_length + EPS) - np.log(net_disp + EPS)),  # D, 必须同时报告
    }


def h2_path_local(pts: np.ndarray, progress: np.ndarray, b: int) -> dict:
    d, _, _ = displacements(pts)
    return h2_path(d[stage_mask(progress[1:], b)])


def h2_path_cumulative(pts: np.ndarray, progress: np.ndarray, b: int) -> dict:
    """plan §5.2：用不超过边界 b 的全部轨迹点。"""
    kb = cum_index(progress, b)
    if kb is None or kb < 1:
        return {"coverage_ok": False, "n_valid": 0}
    d, _, _ = displacements(pts[: kb + 1])
    return h2_path(d)
```

### H3. Turning angle — 代表量 `median turn angle`（local）

```python
def h3_turning(pts: np.ndarray, progress: np.ndarray, b: int) -> dict:
    """plan §6 H3。coverage（§5.3）：至少 2 个相邻有效位移，即至少 3 个轨迹点。"""
    d, r, u = displacements(pts)
    theta, excluded = turn_angles(u, r)
    # theta has one slot per original turn and is aligned to progress[2:].
    sel = stage_mask(progress[2:], b) & np.isfinite(theta)
    if theta.size == 0 or sel.sum() < 1:
        return {"coverage_ok": False, "n_valid": 0, "excluded_zero_norm": excluded}
    vals = theta[sel]
    return {
        "coverage_ok": True,
        "n_valid": int(sel.sum()),
        "excluded_zero_norm": excluded,
        "median_turn_angle": float(np.median(vals)),        # ← PRIMARY
        "mean_turn_angle": float(np.mean(vals)),
        "std_turn_angle": float(np.std(vals)),
        "p90_turn_angle": float(np.percentile(vals, 90)),
    }
```

### H4. Angular velocity — 代表量 `P90 absolute angular velocity`（local）

```python
def h4_angular_velocity(pts: np.ndarray, progress: np.ndarray, b: int) -> dict:
    """plan §6 H4：omega_k = theta_k - theta_{k-1}，同时保存 signed 与 abs。

    PRIMARY 用 P90(|omega|) 而非 mean(omega)：捕捉突然转向，避免正负抵消。
    coverage（§5.3）：至少 3 个相邻有效位移，即至少 4 个轨迹点。
    """
    d, r, u = displacements(pts)
    theta, excluded = turn_angles(u, r)
    if theta.size < 2:
        return {"coverage_ok": False, "n_valid": 0, "excluded_zero_norm": excluded}
    omega = np.diff(theta)
    # An angular velocity is valid only when its two original adjacent turns
    # are valid; np.diff preserves that invalid gap as NaN.
    omega_valid = np.isfinite(theta[1:]) & np.isfinite(theta[:-1])
    sel = stage_mask(progress[3:], b) & omega_valid & np.isfinite(omega)
    if sel.sum() < 1:
        return {
            "coverage_ok": False,
            "n_valid": 0,
            "excluded_zero_norm": excluded,
            "excluded_omega_gap": int((~omega_valid).sum()),
        }
    signed, absolute = omega[sel], np.abs(omega[sel])
    return {
        "coverage_ok": True,
        "n_valid": int(sel.sum()),
        "excluded_zero_norm": excluded,
        "excluded_omega_gap": int((~omega_valid).sum()),
        "p90_abs_angular_velocity": float(np.percentile(absolute, 90)),   # ← PRIMARY
        "median_abs_angular_velocity": float(np.median(absolute)),
        "mean_signed_angular_velocity": float(np.mean(signed)),
        "median_signed_angular_velocity": float(np.median(signed)),
    }
```

### H5. State-trajectory ER / ERV / ERA — 代表量 `centered_ERV`（cumulative）

```python
def er_dynamics_from_series(m: np.ndarray) -> tuple[float, float]:
    """Return VERL-style historical-deviation ERV and ERA for one ER series."""
    delta = historical_deviation(m)
    valid = delta[np.isfinite(delta)]
    erv = float(valid.mean()) if valid.size >= 1 else np.nan
    era = float(np.diff(valid).mean()) if valid.size >= 2 else np.nan
    return erv, era


def h5_er_dynamics(pts: np.ndarray, progress: np.ndarray, b: int) -> dict:
    """Compute centered prefix ER/ERV/ERA, with uncentered sensitivity outputs.

    Every prefix is centered independently before SVD, matching VERL Appendix F.2.
    coverage（§5.3）：ER 至少 2 个非重复状态点；ERV 至少 2 个 prefix 点；ERA 至少 3 个。
    """
    kb = cum_index(progress, b)
    if kb is None or kb < 1:
        return {"coverage_ok": False, "n_valid": 0}

    centered_m = np.asarray([
        spectral_effective_rank(pts[: j + 1], center_rows=True)
        for j in range(1, kb + 1)
    ])
    uncentered_m = np.asarray([
        spectral_effective_rank(pts[: j + 1], center_rows=False)
        for j in range(1, kb + 1)
    ])
    valid = np.isfinite(centered_m) & np.isfinite(uncentered_m)
    centered_m, uncentered_m = centered_m[valid], uncentered_m[valid]
    if centered_m.size < 2:
        return {"coverage_ok": False, "n_valid": int(centered_m.size)}

    centered_erv, centered_era = er_dynamics_from_series(centered_m)
    uncentered_erv, uncentered_era = er_dynamics_from_series(uncentered_m)

    return {
        "coverage_ok": True,
        "n_valid": int(centered_m.size),
        "centered_ERV": centered_erv,                            # ← PRIMARY
        "centered_ER_final": float(centered_m[-1]),              # 必须同时画出
        "centered_ERA": centered_era,                            # 必须同时画出
        "uncentered_ERV_sensitivity": uncentered_erv,
        "uncentered_ER_final_sensitivity": float(uncentered_m[-1]),
        "uncentered_ERA_sensitivity": uncentered_era,
    }
```

> H5 primary 现在与 VERL 附录 F.2 的 mean-centered Gram \(K=Z_cZ_c^\top\) 一致。
> 未中心化 dynamics 只用于判断公共 residual offset 对结果的影响，不进入主 family。

### H6. Length-weighted directional ER — 代表量 `directional ER`（cumulative）

```python
def h6_directional_er(pts: np.ndarray, progress: np.ndarray, b: int) -> dict:
    """plan §6 H6：G = (1/sum r_k) sum r_k u_k u_k^T，对其归一化特征谱取熵指数。

    实现按 plan 要求不构造 D x D 矩阵：对行向量 sqrt(r_k / sum r) * u_k 组成的
    (n, D) 小矩阵做 SVD，其平方奇异值即 G 的非零谱。
    """
    kb = cum_index(progress, b)
    if kb is None or kb < 1:
        return {"coverage_ok": False, "n_valid": 0}

    d, r, u = displacements(pts[: kb + 1])
    ok = r >= NORM_FLOOR
    if ok.sum() < 1:
        return {"coverage_ok": False, "n_valid": 0}

    w = r[ok] / (r[ok].sum() + EPS)
    rows = np.sqrt(w)[:, None] * u[ok]                 # (n_ok, D)
    sigma = np.linalg.svd(rows, compute_uv=False)
    lam = sigma ** 2                                    # G 的非零特征值
    lam = lam[lam > EPS]
    if lam.size == 0:
        return {"coverage_ok": False, "n_valid": int(ok.sum())}
    q = lam / (lam.sum() + EPS)

    return {
        "coverage_ok": True,
        "n_valid": int(ok.sum()),
        "directional_ER": float(np.exp(-np.sum(q * np.log(q + EPS)))),   # ← PRIMARY
        "companion_path_length": float(r[ok].sum()),   # 归一化丢掉了绝对路径长，必须与 H2 的 L 联合解释
    }
```

### H7. Length-weighted turning — 代表量 `weighted turning`（local）

```python
def h7_weighted_turning(pts: np.ndarray, progress: np.ndarray, b: int) -> dict:
    """plan §6 H7：r_bar_k = (r_k + r_{k-1})/2，T = sum r_bar_k theta_k / (sum r_bar_k + eps)。

    回答「大幅移动的位置是否也发生大转向」，与 H6「全局用了多少方向」不同。
    """
    d, r, u = displacements(pts)
    theta, excluded = turn_angles(u, r)
    if theta.size == 0:
        return {"coverage_ok": False, "n_valid": 0, "excluded_zero_norm": excluded}

    # r_bar and theta share the original turn index i - 1, hence both align
    # with progress[2:].  Invalid turns remain NaN and cannot shift stages.
    r_bar = 0.5 * (r[1:] + r[:-1])
    sel = stage_mask(progress[2:], b) & np.isfinite(theta) & np.isfinite(r_bar)
    if sel.sum() < 1:
        return {"coverage_ok": False, "n_valid": 0, "excluded_zero_norm": excluded}
    return {
        "coverage_ok": True,
        "n_valid": int(sel.sum()),
        "excluded_zero_norm": excluded,
        "weighted_turning": float((r_bar[sel] * theta[sel]).sum() / (r_bar[sel].sum() + EPS)),  # ← PRIMARY
    }
```

### H8. Token-level entropy dynamics — 代表量 `median token time-diff coordinate entropy`（local）

```python
def token_entropy_series(token_states: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the three H8 token scalar sequences for one layer.

    `token_states` contains response-only hidden states with shape `(T, D)`.
    The first time-difference entry is NaN because token 0 has no predecessor.
    Zero-energy vectors follow old Experiment04 coordinate_entropy and return 0.
    """
    if token_states.ndim != 2 or token_states.shape[0] < 1:
        raise ValueError("token_states must have shape (T, D) with T >= 1")

    state_coord = np.asarray([
        normalized_energy_entropy(v, center=True, zero_value=0.0)
        for v in token_states
    ])
    state_raw = np.asarray([
        normalized_energy_entropy(v, center=False, zero_value=0.0)
        for v in token_states
    ])
    time_diff = np.full(token_states.shape[0], np.nan, dtype=float)
    if token_states.shape[0] >= 2:
        time_diff[1:] = [
            normalized_energy_entropy(v, center=True, zero_value=0.0)
            for v in np.diff(token_states, axis=0)
        ]
    return {
        "token_time_diff_coordinate_entropy": time_diff,
        "token_state_coordinate_entropy": state_coord,
        "token_state_raw_energy_entropy": state_raw,
    }


def h8_token_entropy_stage(token_hidden: np.ndarray,
                           endpoints: np.ndarray,
                           progress: np.ndarray,
                           b: int,
                           layer: int) -> dict:
    """Aggregate H8 token scalars as window median -> stage median.

    `token_hidden`: `(T, L+1, D)` response-only states; `endpoints` are the
    plan's 1-based exclusive window ends. H8 is representation-independent.
    """
    if token_hidden.ndim != 3 or not 0 <= layer < token_hidden.shape[1]:
        raise ValueError("invalid token hidden states or layer")
    if endpoints.shape != progress.shape:
        raise ValueError("endpoints and progress must have identical shape")

    series = token_entropy_series(token_hidden[:, layer, :])
    selected_windows = np.nonzero(stage_mask(progress, b))[0]
    window_values = {key: [] for key in series}
    for k in selected_windows:
        end = int(endpoints[k])
        start = max(0, end - WINDOW)
        for key, values in series.items():
            finite = values[start:end]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                window_values[key].append(float(np.median(finite)))

    primary_values = window_values["token_time_diff_coordinate_entropy"]
    if not primary_values:
        return {"coverage_ok": False, "n_valid": 0, "anchor_layer": layer,
                "representation": "token"}

    out = {
        "coverage_ok": True,
        "n_valid": len(primary_values),
        "anchor_layer": layer,
        "representation": "token",
    }
    for key, values in window_values.items():
        if not values:
            continue
        values = np.asarray(values)
        out[f"median_{key}"] = float(np.median(values))
        out[f"mean_{key}"] = float(np.mean(values))
        out[f"p90_{key}"] = float(np.percentile(values, 90))
    # PRIMARY at layer = Lfinal: median_token_time_diff_coordinate_entropy
    return out
```

主调度只调用 `layer=Lfinal` 并将 `median_token_time_diff_coordinate_entropy` 放入 primary family。
`L3/L12/L24/Lfinal` 与全部层调用相同函数输出 exploratory profile；这些 profile 不得覆盖 final-layer primary。

---

## 2. 纵向 8 个家族（plan §7）

纵向量先在**单个 chunk** 上算（对 `layers = traj[k]`，形状 `(L+1, D)`），再在 stage 内跨 chunk 聚合。

```python
def vertical_stage_reduce(traj: np.ndarray, progress: np.ndarray, b: int,
                          per_chunk_fn, key: str, reducer=np.median) -> dict:
    """在 stage b 内对每个 chunk 调用 per_chunk_fn，取出 key，再跨 chunk 聚合。

    plan §8 的纵向代表量除 V1 用 P90 外，其余均为 median。
    """
    sel = np.nonzero(stage_mask(progress, b))[0]
    vals = []
    for k in sel:
        out = per_chunk_fn(traj[k])
        if out.get("coverage_ok") and np.isfinite(out.get(key, np.nan)):
            vals.append(out[key])
    if not vals:
        return {"coverage_ok": False, "n_valid": 0}
    return {"coverage_ok": True, "n_valid": len(vals), key: float(reducer(vals))}
```

### V1. Layer-update norm — 代表量 `P90 relative layer-update norm`（local）

```python
def v1_layer_update_norm(layers: np.ndarray) -> dict:
    """plan §7 V1：a_{k,l} = ||d^v_{k,l}||；a_rel = ||d^v_{k,l}|| / (||y_{k,l-1}|| + eps)。

    concentration C = max_l a / (sum_l a + eps)。
    coverage（§5.3）：至少 1 个有效 layer update。
    用 relative 作 primary 是为了消掉「残差流范数随深度单调增」这个平凡解释。
    """
    d_v = np.diff(layers, axis=0)                       # (L, D)
    if d_v.shape[0] < 1:
        return {"coverage_ok": False, "n_valid": 0}
    a = np.linalg.norm(d_v, axis=1)                     # (L,)
    base = np.linalg.norm(layers[:-1], axis=1)          # ||y_{k,l-1}||
    a_rel = a / (base + EPS)
    return {
        "coverage_ok": True,
        "n_valid": int(a.size),
        "p90_relative_layer_update_norm": float(np.percentile(a_rel, 90)),   # ← PRIMARY
        "mean_relative_layer_update_norm": float(a_rel.mean()),
        "median_relative_layer_update_norm": float(np.median(a_rel)),
        "mean_raw_layer_update_norm": float(a.mean()),
        "median_raw_layer_update_norm": float(np.median(a)),
        "p90_raw_layer_update_norm": float(np.percentile(a, 90)),
        "argmax_layer": int(np.argmax(a) + 1),                                # maximum layer
        "concentration": float(a.max() / (a.sum() + EPS)),                    # C^v_k
    }


def v1_stage(traj: np.ndarray, progress: np.ndarray, b: int) -> dict:
    """per-chunk 已取 P90(over layers)；stage 内跨 chunk 用 median 聚合。"""
    return vertical_stage_reduce(traj, progress, b, v1_layer_update_norm,
                                 "p90_relative_layer_update_norm", np.median)
```

### V2. Raw state angle — 代表量 `median raw state angle`（local）

```python
def v2_raw_state_angle(layers: np.ndarray) -> dict:
    """plan §7 V2：phi_raw = arccos(cos(y_{k,l}, y_{k,l-1}))。

    coverage（§5.3）：至少 2 个有效相邻 layer states。
    """
    if layers.shape[0] < 2:
        return {"coverage_ok": False, "n_valid": 0}
    norms = np.linalg.norm(layers, axis=1)
    angles = []
    for l in range(1, layers.shape[0]):
        if norms[l] < NORM_FLOOR or norms[l - 1] < NORM_FLOOR:
            continue
        cos = float(layers[l] @ layers[l - 1]) / (norms[l] * norms[l - 1] + EPS)
        angles.append(np.arccos(np.clip(cos, -1.0, 1.0)))
    if not angles:
        return {"coverage_ok": False, "n_valid": 0}
    angles = np.asarray(angles)
    return {
        "coverage_ok": True,
        "n_valid": int(angles.size),
        "median_raw_state_angle": float(np.median(angles)),      # ← PRIMARY
        "mean_raw_state_angle": float(angles.mean()),
        "p90_raw_state_angle": float(np.percentile(angles, 90)),
    }
```

### V3. State angle demean — 代表量 `median state angle demean`（local）

```python
def base_layer_common(all_base_traj: list[np.ndarray]) -> np.ndarray:
    """plan §7 V3：mu^base_{r,l} = (1/R) sum_i ( (1/K_i) sum_k y_{i,k,r,l} )。

    label-blind、rollout-equal（先在 rollout 内对 k 平均，再在 rollout 间平均）。
    只在 base checkpoint 上算一次，冻结后复用于全部 checkpoint。
    返回 (L+1, D)。
    """
    per_rollout = [t.mean(axis=0) for t in all_base_traj]     # 每条: (L+1, D)
    return np.mean(np.stack(per_rollout, axis=0), axis=0)


def v3_state_angle_demean(layers: np.ndarray, mu_base: np.ndarray) -> dict:
    """plan §7 V3：phi_demean = arccos(cos(y_{k,l} - mu_l, y_{k,l-1} - mu_{l-1}))。"""
    if layers.shape[0] < 2:
        return {"coverage_ok": False, "n_valid": 0}
    centered = layers - mu_base                               # (L+1, D)
    norms = np.linalg.norm(centered, axis=1)
    angles = []
    for l in range(1, centered.shape[0]):
        if norms[l] < NORM_FLOOR or norms[l - 1] < NORM_FLOOR:
            continue
        cos = float(centered[l] @ centered[l - 1]) / (norms[l] * norms[l - 1] + EPS)
        angles.append(np.arccos(np.clip(cos, -1.0, 1.0)))
    if not angles:
        return {"coverage_ok": False, "n_valid": 0}
    angles = np.asarray(angles)
    return {
        "coverage_ok": True,
        "n_valid": int(angles.size),
        "median_state_angle_demean": float(np.median(angles)),   # ← PRIMARY
        "mean_state_angle_demean": float(angles.mean()),
        "p90_state_angle_demean": float(np.percentile(angles, 90)),
    }
```

### V4. Layer-update turning angle — 代表量 `median layer-update turning angle`（local）

```python
def v4_layer_update_turning(layers: np.ndarray) -> dict:
    """plan §7 V4：v_{k,l} = d^v_{k,l} / (||d^v_{k,l}|| + eps)，
    theta^{v,update}_{k,l} = arccos(clip(v_{k,l} . v_{k,l-1}, -1, 1))，l = 2..L。

    回答「第 l 层的写入方向是否延续第 l-1 层的写入方向」，与 V2/V3 的 state angle 不同。
    coverage（§5.3）：至少 2 个连续、norm 超阈值的 layer updates，即至少 3 个有效 layer states。
    """
    d_v = np.diff(layers, axis=0)
    if d_v.shape[0] < 2:
        return {"coverage_ok": False, "n_valid": 0, "excluded_zero_norm": 0}
    v, n = unit_directions(d_v)
    ok = n >= NORM_FLOOR
    angles, excluded = [], 0
    for l in range(1, v.shape[0]):
        if ok[l] and ok[l - 1]:
            angles.append(np.arccos(np.clip(float(v[l] @ v[l - 1]), -1.0, 1.0)))
        else:
            excluded += 1
    if not angles:
        return {"coverage_ok": False, "n_valid": 0, "excluded_zero_norm": excluded}
    angles = np.asarray(angles)
    return {
        "coverage_ok": True,
        "n_valid": int(angles.size),
        "excluded_zero_norm": excluded,
        "median_layer_update_turning_angle": float(np.median(angles)),   # ← PRIMARY
        "mean_layer_update_turning_angle": float(angles.mean()),
        "std_layer_update_turning_angle": float(angles.std()),
        "p90_layer_update_turning_angle": float(np.percentile(angles, 90)),
    }
```

### V5. Vertical path metrics — 代表量 `median vertical straightness`（local）

```python
def v5_vertical_path(layers: np.ndarray) -> dict:
    """plan §7 V5：L^v = sum_l ||d^v_{k,l}||，N^v = ||sum_l d^v_{k,l}||，
    S^v = N^v/(L^v+eps)，D^v = log(L^v+eps) - log(N^v+eps)。

    注：sum_l d^v_{k,l} 望远镜化为 y_{k,L} - y_{k,0}，即 N^v 就是首末层的距离。
    coverage（§5.3）：至少 3 个有效 layer states 且层间位移 norm 超阈值。
    """
    d_v = np.diff(layers, axis=0)
    if d_v.shape[0] < 2:
        return {"coverage_ok": False, "n_valid": 0}
    norms = np.linalg.norm(d_v, axis=1)
    if (norms >= NORM_FLOOR).sum() < 2:
        return {"coverage_ok": False, "n_valid": int((norms >= NORM_FLOOR).sum())}
    path_v = float(norms.sum())
    net_v = float(np.linalg.norm(d_v.sum(axis=0)))          # = ||y_{k,L} - y_{k,0}||
    return {
        "coverage_ok": True,
        "n_valid": int(d_v.shape[0]),
        "vertical_straightness": net_v / (path_v + EPS),     # ← PRIMARY
        "vertical_path_length": path_v,                      # L^v, 必须同时报告
        "vertical_net_displacement": net_v,                  # N^v, 必须同时报告
        "vertical_log_detour": float(np.log(path_v + EPS) - np.log(net_v + EPS)),  # D^v
    }
```

### V6. Raw activation entropy — 代表量 `median raw activation entropy`（local）

```python
def v6_raw_activation_entropy(layers: np.ndarray) -> dict:
    """plan §7 V6：e_j = y_{k,l,j}^2（不减 coordinate mean、不用 base 校准），
    H_raw = -sum p log p / log D。

    刻意保留持续高激活坐标的影响 —— 它是 V7 centered 版本的 raw 对照。
    coverage（§5.3）：至少 1 个有效 layer state。
    """
    vals = [normalized_energy_entropy(layers[l], center=False) for l in range(layers.shape[0])]
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if vals.size < 1:
        return {"coverage_ok": False, "n_valid": 0}
    return {
        "coverage_ok": True,
        "n_valid": int(vals.size),
        "median_raw_activation_entropy": float(np.median(vals)),   # ← PRIMARY
        "mean_raw_activation_entropy": float(vals.mean()),
        "p90_raw_activation_entropy": float(np.percentile(vals, 90)),
    }
```

### V7. Centered state / layer-difference / robust-z entropy — 代表量 `median layer-difference entropy`（local）

```python
def base_coordinate_stats(all_base_layers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """plan §7 V7：在 base checkpoint 上按 (representation, layer, coordinate) 算
    label-blind 的 mu^base_{r,l,j} 与 sigma^base_{r,l,j}，冻结复用。

    all_base_layers: (n_chunks_all_rollouts, L+1, D) -> mu, sigma 各 (L+1, D)
    """
    mu = all_base_layers.mean(axis=0)
    sigma = all_base_layers.std(axis=0)
    return mu, sigma


def v7_entropies(layers: np.ndarray,
                 mu_base: np.ndarray | None = None,
                 sigma_base: np.ndarray | None = None) -> dict:
    """plan §7 V7 三个量：

    - H_centered_state(k,l) = H_c(y_{k,l})            先减 token 内 coordinate mean
    - H_delta_v(k,l)        = H_c(d^v_{k,l})          ← PRIMARY 的来源
    - H_robust_z(k,l)       用 base 的 per-coordinate (mu, sigma) 做 z-score + clip[-8,8]，
                            然后直接以 z'^2 构造概率，**不再**做 coordinate mean centering
    """
    out: dict = {"coverage_ok": False, "n_valid": 0}

    # (a) centered state entropy —— V6 raw 的中心化对照
    cs = np.asarray([normalized_energy_entropy(layers[l], center=True)
                     for l in range(layers.shape[0])])
    cs = cs[np.isfinite(cs)]

    # (b) layer-difference entropy —— 本家族 PRIMARY
    d_v = np.diff(layers, axis=0)
    ld = np.asarray([normalized_energy_entropy(d_v[i], center=True)
                     for i in range(d_v.shape[0])]) if d_v.shape[0] >= 1 else np.asarray([])
    ld = ld[np.isfinite(ld)]

    if ld.size < 1:
        return out
    out = {
        "coverage_ok": True,
        "n_valid": int(ld.size),
        "median_layer_difference_entropy": float(np.median(ld)),   # ← PRIMARY
        "mean_layer_difference_entropy": float(ld.mean()),
    }
    if cs.size >= 1:
        out["median_centered_state_entropy"] = float(np.median(cs))

    # (c) robust-z entropy —— rogue-dimension 敏感性控制
    if mu_base is not None and sigma_base is not None:
        denom = np.maximum(sigma_base, SIGMA_FLOOR)
        z = np.clip((layers - mu_base) / denom, -Z_CLIP, Z_CLIP)
        rz = []
        for l in range(z.shape[0]):
            e = z[l] ** 2
            total = e.sum()
            if total <= EPS:
                continue
            p = e / (total + EPS)                                  # 不做 coordinate centering
            rz.append(float(-np.sum(p * np.log(p + EPS)) / np.log(z.shape[1])))
        if rz:
            out["median_robust_z_entropy"] = float(np.median(rz))
        out["low_variance_coordinate_fraction"] = float(
            np.mean(sigma_base < SIGMA_FLOOR))                      # plan 要求必须报告
    return out
```

### V8. Vertical effective rank — 代表量 `median layer-update ER`（local）

```python
def v8_vertical_er(layers: np.ndarray) -> dict:
    """V8 centered layer-state ER plus raw/centered layer-update ER.

    coverage（§5.3）：至少 3 个有效 layer states 且层间位移 norm 超阈值。
    """
    if layers.shape[0] < 3:
        return {"coverage_ok": False, "n_valid": 0}
    d_v = np.diff(layers, axis=0)
    if (np.linalg.norm(d_v, axis=1) >= NORM_FLOOR).sum() < 2:
        return {"coverage_ok": False, "n_valid": 0}
    return {
        "coverage_ok": True,
        "n_valid": int(layers.shape[0]),
        # Differences already remove common state offset; keep raw update ER primary.
        "layer_update_ER": spectral_effective_rank(d_v, center_rows=False),  # ← PRIMARY
        "layer_update_ER_centered": spectral_effective_rank(d_v, center_rows=True),
        # State ER follows the centered trajectory convention.
        "layer_state_ER": spectral_effective_rank(layers, center_rows=True),
        "layer_state_ER_uncentered_sensitivity": spectral_effective_rank(
            layers, center_rows=False
        ),
    }
```

---

## 3. 16 个代表量的调度表

```python
# plan §8 的 16 行。key 是写进 metrics parquet 的 metric 名。
PRIMARY_REGISTRY = {
    # ---- H1-H7: horizontal inputs pts = traj[:, anchor_layer, :] ----
    # anchor_layers is frozen to L3/L12/L24/Lfinal for every entry below.
    "H1": dict(axis="horizontal", family="movement",
               fn=h1_movement,          key="median_relative_movement",        mode="local",
               anchor_layers=HORIZONTAL_ANCHOR_LAYERS),
    "H2": dict(axis="horizontal", family="path",
               fn=h2_path_local,        key="straightness",                    mode="local+cumulative",
               cumulative_fn=h2_path_cumulative, anchor_layers=HORIZONTAL_ANCHOR_LAYERS),
    "H3": dict(axis="horizontal", family="turning",
               fn=h3_turning,           key="median_turn_angle",               mode="local",
               anchor_layers=HORIZONTAL_ANCHOR_LAYERS),
    "H4": dict(axis="horizontal", family="angular_velocity",
               fn=h4_angular_velocity,  key="p90_abs_angular_velocity",        mode="local",
               anchor_layers=HORIZONTAL_ANCHOR_LAYERS),
    "H5": dict(axis="horizontal", family="state_er_dynamics",
               fn=h5_er_dynamics,       key="centered_ERV",                    mode="cumulative",
               anchor_layers=HORIZONTAL_ANCHOR_LAYERS),
    "H6": dict(axis="horizontal", family="directional_spectrum",
               fn=h6_directional_er,    key="directional_ER",                  mode="cumulative",
               anchor_layers=HORIZONTAL_ANCHOR_LAYERS),
    "H7": dict(axis="horizontal", family="length_angle_coupling",
               fn=h7_weighted_turning,  key="weighted_turning",                mode="local",
               anchor_layers=HORIZONTAL_ANCHOR_LAYERS),
    # ---- H8: response-token scalars, no mean/last representation duplication ----
    "H8": dict(axis="horizontal", family="token_entropy_dynamics",
               fn=h8_token_entropy_stage,
               key="median_token_time_diff_coordinate_entropy", mode="local",
               representation="token", primary_layer="final", all_layer_profile=True),
    # ---- vertical，输入 traj[k] = (L+1, D)，stage 内跨 chunk 聚合 ----
    "V1": dict(axis="vertical", family="layer_update_norm",
               fn=v1_layer_update_norm, key="p90_relative_layer_update_norm",  mode="local"),
    "V2": dict(axis="vertical", family="raw_state_angle",
               fn=v2_raw_state_angle,   key="median_raw_state_angle",          mode="local"),
    "V3": dict(axis="vertical", family="demean_state_angle",
               fn=v3_state_angle_demean, key="median_state_angle_demean",      mode="local",
               needs="mu_base"),
    "V4": dict(axis="vertical", family="layer_update_turning",
               fn=v4_layer_update_turning, key="median_layer_update_turning_angle", mode="local"),
    "V5": dict(axis="vertical", family="path",
               fn=v5_vertical_path,     key="vertical_straightness",           mode="local"),
    "V6": dict(axis="vertical", family="raw_activation_entropy",
               fn=v6_raw_activation_entropy, key="median_raw_activation_entropy", mode="local"),
    "V7": dict(axis="vertical", family="centered_difference_robust_entropy",
               fn=v7_entropies,         key="median_layer_difference_entropy", mode="local",
               needs="mu_base+sigma_base"),
    "V8": dict(axis="vertical", family="vertical_spectrum",
               fn=v8_vertical_er,       key="layer_update_ER",                 mode="local"),
}
```

对照表：

| ID | 轴 | 代表量 key | depth scope | stage mode | 需要 base 冻结量 |
|---|---|---|---|---|---|
| H1 | h | `median_relative_movement` | L3/L12/L24/Lfinal | local | — |
| H2 | h | `straightness` | L3/L12/L24/Lfinal | local + cumulative | — |
| H3 | h | `median_turn_angle` | L3/L12/L24/Lfinal | local | — |
| H4 | h | `p90_abs_angular_velocity` | L3/L12/L24/Lfinal | local | — |
| H5 | h | `centered_ERV` | L3/L12/L24/Lfinal | cumulative | — |
| H6 | h | `directional_ER` | L3/L12/L24/Lfinal | cumulative | — |
| H7 | h | `weighted_turning` | L3/L12/L24/Lfinal | local | — |
| H8 | h | `median_token_time_diff_coordinate_entropy` | Lfinal primary | local | — |
| V1 | v | `p90_relative_layer_update_norm` | all layers | local | — |
| V2 | v | `median_raw_state_angle` | all layers | local | — |
| V3 | v | `median_state_angle_demean` | all layers | local | `mu_base` (L+1, D) |
| V4 | v | `median_layer_update_turning_angle` | all layers | local | — |
| V5 | v | `vertical_straightness` | all layers | local | — |
| V6 | v | `median_raw_activation_entropy` | all layers | local | — |
| V7 | v | `median_layer_difference_entropy` | all layers | local | `mu_base`, `sigma_base` |
| V8 | v | `layer_update_ER` | all layers | local | — |

---

## 4. plan §14 的 toy 单元测试，代码形式

```python
def toy_cases():
    """plan §14 的 12 条数值断言，只表达期望值，不在此运行。"""
    D = 8

    # 1. 完全直线：L = N，straightness = 1，log-detour = 0，turn angle = 0
    line = np.cumsum(np.tile(np.eye(1, D, 0), (5, 1)), axis=0)
    # h2_path -> straightness ~ 1.0, log_detour ~ 0.0
    # h3_turning -> median_turn_angle ~ 0.0

    # 2. 走出再返回：L > 0，N ~ 0，straightness ~ 0，log-detour 有限大且非 Inf
    out_back = np.vstack([np.zeros((1, D)), np.eye(1, D, 0), np.zeros((1, D))])
    # h2_path -> net_displacement ~ 0, straightness ~ 0, log_detour 有限（靠 EPS 兜底）

    # 3. 两个等长正交方向：directional ER = 2（容差内）
    orth = np.vstack([np.zeros((1, D)), np.eye(1, D, 0), np.eye(1, D, 0) + np.eye(1, D, 1)])
    # h6_directional_er -> directional_ER ~ 2.0

    # 4. 所有 movement 同方向：directional ER = 1
    same = np.cumsum(np.tile(np.eye(1, D, 0), (4, 1)), axis=0)
    # h6_directional_er -> directional_ER ~ 1.0

    # 5. 单坐标能量：coordinate entropy = 0；等能量：normalized entropy = 1
    # normalized_energy_entropy(np.eye(1, D, 0)[0], center=False) ~ 0.0
    # normalized_energy_entropy(np.ones(D), center=False) ~ 1.0

    # 6. 全体 hidden state 加同一平移：horizontal displacement/path/turn 不变
    # h2_path(d of pts) == h2_path(d of pts + c)，因为 d 只依赖差分

    # 7. 全体乘正比例常数 a > 0：
    #    angle / straightness / directional_ER / normalized_entropy 不变
    #    raw movement / path_length 按 a 比例变化

    # 8. 中间零位移：角度数组保留原长度和 NaN 占位，不得压缩或错配 stage
    # theta = turn_angles(u, r)[0]；len(theta) == len(u)-1；无效 turn 为 NaN
    # np.diff(theta) 在缺口两侧仍为 NaN，不得跨过缺口连接两个不相邻 turn

    # 9. centered H5：给同一 prefix 的全部点加 common vector，centered ER/ERV/ERA 不变
    # h5_er_dynamics(pts, ...) primary == h5_er_dynamics(pts + c, ...) primary

    # 10. V8：同一方向的非零 layer updates -> raw layer_update_ER ~ 1；
    #     centered update matrix 全零 -> layer_update_ER_centered is NaN, not 0.

    # 11. H8：constant token states -> time-diff coordinate entropy = 0; adding a
    #     common coordinate offset can change raw-energy entropy but not centered entropy.

    # 12. 每种最小 coverage 边界：point count 恰好达到 / 少于门槛时状态正确
    #    turning 需 >= 3 点；angular velocity 需 >= 4 点；
    #    ER >= 2 非重复状态点；ERA >= 3 prefix 点；
    #    V4 需 >= 3 有效 layer states；V5/V8 需 >= 3 有效 layer states；
    #    H8 diff entropy 每个 window 至少有 1 个 token difference

    # 13. stage boundary：endpoint 恰好 = 0.25 / 0.50 / 0.75 / 1.00 时只进一个 stage
    # stage_mask(np.array([0.25]), 0) -> [False]；stage_mask(np.array([0.25]), 1) -> [True]
    # stage_mask(np.array([1.00]), 3) -> [True]（末段闭区间）

    # 14. anchor dispatch: every H1-H7 call covers L3/L12/L24/Lfinal; no anchor is dropped.
    # 15. question bootstrap：同题的 rollout / stage / representation / anchor 整体重采样
    # 16. label permutation：correct/wrong 效应多次置换后以 0 为中心
```

---

## 5. 四处实现解释

**1. H5 使用 prefix trajectory centering。** 每个 prefix 独立减去该 prefix 的 row mean 后再做 SVD，
因此 H5 primary 与 VERL 附录 F.2 的 centered Gram \(K = Z_cZ_c^\top\) 一致。
未中心化 ER/ERV/ERA 仍保存，但只能诊断公共 residual offset 的影响。

**2. V8 的两种中心化不能混为一谈。** `layer_state_ER` 在层维度减 row mean，去掉各层共有 offset；
`layer_update_ER` 保持 raw primary，因为 update 本身已由差分去掉 state offset。`layer_update_ER_centered`
只衡量偏离平均 update 的方向多样性，完全相同的 updates 会使它退化为零矩阵并按缺失处理。

**3. V5 的 \(N^v\) 会望远镜化。** \(\sum_l d^v_{k,l} = y_{k,L} - y_{k,0}\)，
所以 vertical net displacement 其实就是首末层距离，与中间层路径完全无关。
这不是错误（straightness 的定义本就如此），但解释时要注意：
`vertical_straightness` 实际测的是「首末层距离 / 逐层累计距离」，
而不是任何关于中间层顺序的信息。V4 的 layer-update turning 才携带顺序信息。

**4. V1 用 relative 是必要的。** 残差流范数随深度单调增长是 transformer 的已知性质，
所以 raw `layer_update_norm` 的大部分变化可能只是范数增长。plan 选
`P90(relative_layer_update_norm)` 作 primary（分母 \(\|y_{k,l-1}\|\)）已经处理了这一点；
V2/V3/V4 用角度也天然免疫。代码里 raw 版本同时保存，供诊断对比。

---

## 6. 本文件的边界

- 只表达**指标怎么算**。不含 hidden extraction、checkpoint 循环、parquet 写出、统计建模（plan §10）、绘图（plan §12）。
- H1-H7 按单条 rollout + 单一 representation + 单一 anchor layer + 单个 stage 设计；`mean_w128_s32` 与 `last_s32`
  走同一套代码，只是传入的 `traj` 不同。H8 使用 response-only token hidden states、`representation=token`，不随两种 chunk representation 重复。
- 没有运行过任何一行；数值正确性由 plan §14 的 toy cases 在实现阶段验证。
