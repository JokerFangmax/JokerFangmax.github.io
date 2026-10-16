"""WM_06 toy: 检索修复的是锚点，不是演化——FOV 几何检索 vs 内容亲和检索

设定（HyDRA「出画-入画」事件的最小机制版）:
  世界: 64 格环形走廊。一个主体以速率 v (|v| ∈ [0.2, 0.8], 方向随机, 每 episode
        抽一次)持续绕环行走——不管看不看得见都在走（对应 HyDRA 的 out-of-view 演化）。
  相机: 宽 16 格的视窗沿 [0, 48] 做三角波往返平移（1 格/步）——「先左移后右移」
        主动制造出画-入画事件（HM-World 28 种相机轨迹的设计原则）。
  记忆: 完美写入端——每一步都存（含空镜头; 主体可见时附带带噪位置观测 σ_obs=0.3）。
  查询: 仅在「主体重新进入视窗」的时刻（DSC 时刻）。检索系统取回 K=5 帧,
        由锚点帧 + 学到的主体速率 v̂ 外推主体当前位置。

三种检索/外推（写入端全部理想化, 只比「取回什么 + 怎么演化」两个变量）:
  FOV       按「相机位姿距当前位姿最近」取回 top-K（Context-as-Memory 式几何检索;
            取回的可能是空镜头, 或上一个周期同位姿的陈旧镜头）
  Affinity  按「内容相关性」取回——理想化为「最近的主体可见帧」（HyDRA 动态亲和检索）
  Oracle    同 Affinity 但 v̂ = v（完美学到的主体动力学, 动力学知识上限）

计分:
  hit   取回集合含主体可见帧 → 锚点 τ = 其中最新一帧, 预测 = x_obs(τ) + v̂·(t−τ)
  miss  全是空镜头 → 主体信息缺失, 预测 = 视窗中心（均匀先验下的最优点估计,
        对应论文「主体消失/错位」的失败模式）
  v̂ = v + N(0, σ_v), σ_v = 0.05（Affinity 与 FOV-hit 共用: 学到的动力学误差）
  误差 = 环上距离 min(|Δ|, 64−|Δ|), 按离场时长 D = t_重入 − t_出画 分桶

预期: FOV 锚点陈旧或缺失 → 误差大、与 D 弱相关（短离场时新鲜命中才小）;
  Affinity 锚点准但演化靠外推 → 误差随 D 线性增长, 斜率 ≈ E|N(0,σ_v)| = 0.04;
  Oracle 只剩锚点观测噪声地板 ≈ E|N(0,σ_obs)| = 0.24。
  → 检索修复锚点; 演化仍是外推（斜率 = 对动力学的无知程度, 呼应 WM_01）。
"""

import numpy as np

rng = np.random.default_rng(11)

# ---- 世界与实验参数 ----
L = 64          # 环形走廊长度
W = 16          # 相机视窗宽度
PMAX = L - W    # 视窗起点范围 [0, PMAX]
K = 5           # 检索帧数 (top-K)
T = 4000        # 每 episode 步数
EPISODES = 40
SIGMA_OBS = 0.3   # 主体位置观测噪声(格)
SIGMA_V = 0.05    # 学到的主体速率误差 (Affinity / FOV-hit)
V_LO, V_HI = 0.2, 0.8

D_BINS = [1, 20, 40, 80, 160, 320, 10**9]
N_BIN = len(D_BINS) - 1


def circ_dist(a, b):
    d = np.abs(a - b) % L
    return np.minimum(d, L - d)


def run_episode():
    """跑一个 episode, 在每个重入时刻当场评估三种检索。"""
    v = rng.uniform(V_LO, V_HI) * rng.choice([-1, 1])   # 主体速率(带符号)
    x = rng.uniform(0, L)                               # 主体位置
    p = rng.uniform(0, PMAX)                            # 视窗起点
    d = rng.choice([-1, 1])                             # 平移方向

    poses = np.empty(T)         # 相机位姿
    has_subj = np.empty(T, bool)
    x_obs = np.full(T, np.nan)  # 主体带噪观测(不可见时 nan)
    x_true = np.empty(T)        # 主体真值位置(oracle 专用, 记忆系统不可见)

    # 累计器: 逐事件 (D 桶) —— 事件数 / D 和 / 命中数 / 锚点年龄和 / 三方法误差和
    out = {
        "n": np.zeros(N_BIN),
        "dsum": np.zeros(N_BIN),
        "fov_hit": np.zeros(N_BIN),
        "fov_age": np.zeros(N_BIN),          # 仅命中事件
        "err": np.zeros((N_BIN, 3)),         # fov / affinity / oracle
    }

    was_in = False
    last_exit = None

    for t in range(T):
        x = (x + v) % L
        p = p + d
        if p < 0:
            p, d = -p, 1
        elif p > PMAX:
            p, d = 2 * PMAX - p, -1

        in_view = ((x - p) % L) < W
        poses[t], has_subj[t], x_true[t] = p, in_view, x
        if in_view:
            x_obs[t] = x + rng.normal(0, SIGMA_OBS)
            if not was_in and last_exit is not None:
                D = t - last_exit
                bi = np.digitize(D, D_BINS) - 1
                if 0 <= bi < N_BIN:
                    out["n"][bi] += 1
                    out["dsum"][bi] += D
                    # --- FOV: top-K by (位姿距离, 越新越优先); 记忆只含 t' < t ---
                    score = np.abs(poses[:t] - p) - 1e-6 * np.arange(t)
                    top = np.argpartition(score, min(K, t - 1))[:K]
                    cand = top[has_subj[top]]
                    if len(cand) == 0:
                        pred_fov = p + W / 2            # miss: 视窗中心(均匀先验最优)
                    else:
                        tau = cand[np.argmax(cand)]
                        pred_fov = x_obs[tau] + (v + rng.normal(0, SIGMA_V)) * (t - tau)
                        out["fov_hit"][bi] += 1
                        out["fov_age"][bi] += t - tau
                    # --- Affinity: 最近主体可见帧 + 带噪动力学 ---
                    tau_a = np.where(has_subj[:t])[0][-1]
                    pred_aff = x_obs[tau_a] + (v + rng.normal(0, SIGMA_V)) * (t - tau_a)
                    # --- Oracle: 同锚点, 完美动力学 ---
                    pred_ora = x_obs[tau_a] + v * (t - tau_a)

                    err = [
                        circ_dist(pred_fov % L, x),
                        circ_dist(pred_aff % L, x),
                        circ_dist(pred_ora % L, x),
                    ]
                    for k in range(3):
                        out["err"][bi, k] += err[k]
            was_in = True
        else:
            if was_in:
                last_exit = t
            was_in = False

    return out


def main():
    tot = {
        "n": np.zeros(N_BIN),
        "dsum": np.zeros(N_BIN),
        "fov_hit": np.zeros(N_BIN),
        "fov_age": np.zeros(N_BIN),
        "err": np.zeros((N_BIN, 3)),
    }
    for _ in range(EPISODES):
        o = run_episode()
        for k in tot:
            tot[k] += o[k]

    names = ["FOV", "Affinity", "Oracle"]
    print(f"events total: {int(tot['n'].sum())}")
    print(f"\n===== 重入时刻主体位置 MAE vs 离场时长 D =====")
    print(f"{'D bin':>10} {'n':>5} {'D̄':>5} {'FOV命中':>7} {'FOV锚龄':>8} | "
          + " | ".join(f"{n:>9}" for n in names))
    for bi in range(N_BIN):
        n = tot["n"][bi]
        if n < 20:
            continue
        lo, hi = D_BINS[bi], min(D_BINS[bi + 1], 999)
        hit = tot["fov_hit"][bi] / n
        dbar = tot["dsum"][bi] / n
        age = tot["fov_age"][bi] / tot["fov_hit"][bi] if tot["fov_hit"][bi] else np.nan
        mae = " | ".join(f"{tot['err'][bi, k] / n:9.2f}" for k in range(3))
        print(f"{f'{lo}-{hi}':>10} {int(n):5d} {dbar:5.0f} {hit:7.2%} {age:8.1f} | {mae}")

    # 斜率: 对三方法在 D∈[1,320] 上按桶内实际平均 D 做最小二乘
    xs, ys = [], []
    for bi in range(N_BIN):
        if 1 <= D_BINS[bi] <= 320 and tot["n"][bi] >= 20:
            xs.append(tot["dsum"][bi] / tot["n"][bi])
            ys.append([tot["err"][bi, k] / tot["n"][bi] for k in range(3)])
    X = np.column_stack([np.array(xs), np.ones(len(xs))])
    slopes = np.linalg.lstsq(X, np.array(ys), rcond=None)[0][0]
    print("\nslope (MAE per unit D, D∈[1,320]):",
          {n: round(float(s), 4) for n, s in zip(names, slopes)})
    print("theoretical: Affinity slope ≈ E|N(0,σ_v)| = "
          f"{SIGMA_V * np.sqrt(2 / np.pi):.4f};"
          f" Oracle floor ≈ E|N(0,σ_obs)| = {SIGMA_OBS * np.sqrt(2 / np.pi):.3f}")


if __name__ == "__main__":
    main()
