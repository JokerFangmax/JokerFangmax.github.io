"""WM_01 toy: 回放式记忆 vs 演化式记忆——「看见」到「知道」的距离

世界: 64 格环形走廊。
  - 每格一个静态属性(颜色值), t=0 抽一次后不变;
  - 6 个固定位置的「累积型」动态过程(水位/熔化), s_i(t+1) = s_i(t) + r_i,
    r_i ~ U(0.1, 0.5) 每 episode 抽一次, 与观测无关地持续演化。
智能体: 视窗 W=8 沿环随机游走(±1..3), 每步 5% 概率随机传送(制造长短不一的
  离开间隔)。观测带噪 σ_obs = 0.1。
查询: 每 10 步让各记忆系统报告全部 64 格的当前状态估计, 按「该格距上次被
  观测的时长 Δ」分桶, 分别统计静态格 / 动态格的平均绝对误差。

四种记忆(都假设完美的写入端——想存就存得到, 隔离「怎么用」这一个变量):
  Replay        报上次观测值(冻结)——检索式记忆的理想上限
  EvolveAll     所有格按全局平均速率 r̄ 外推——「全都当动态」的均匀外推
  Hybrid        静态格冻结 + 动态格按各自学到的速率 r̂_i = r_i + N(0, σ_r) 外推
  HybridOracle  同 Hybrid 但 r̂_i = r_i(完美动力学)——只剩观测噪声地板

只评估至少被观测过一次的格(Δ 有定义); 每格误差 = |估计 - 真值|。
预期: 动态格误差斜率 = 对动力学的无知程度:
  Replay: E[r]=0.30 / EvolveAll: E|r_i-r̄|=0.10 / Hybrid: E|ε|≈0.04 / Oracle: 0
静态格: Replay/Hybrid ≈ 观测噪声地板, EvolveAll 斜率 = r̄ ≈ 0.30。
"""

import numpy as np

rng = np.random.default_rng(7)

# ---- 世界与实验参数 ----
L = 64          # 环形走廊长度
W = 8           # 视窗宽度
N_DYN = 6       # 动态过程个数
T = 2000        # 每 episode 步数
QUERY_EVERY = 10
EPISODES = 60
SIGMA_OBS = 0.1   # 观测噪声
SIGMA_RATE = 0.05  # 学到的动态速率误差 (Hybrid)
R_LO, R_HI = 0.1, 0.5

GAP_BINS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 10**9]


def run_episode():
    static = rng.uniform(-1, 1, L)                    # 静态属性
    dyn_idx = rng.choice(L, N_DYN, replace=False)     # 动态过程位置
    rates = rng.uniform(R_LO, R_HI, N_DYN)            # 各自速率
    is_dyn = np.zeros(L, bool); is_dyn[dyn_idx] = True
    rate_cell = np.zeros(L); rate_cell[dyn_idx] = rates
    rate_hat = np.zeros(L); rate_hat[dyn_idx] = rates + rng.normal(0, SIGMA_RATE, N_DYN)
    r_bar = rates.mean()

    # 真值轨迹 (动态过程从 0 开始累积)
    dyn_state = np.zeros(N_DYN)
    true_state = static.copy(); true_state[dyn_idx] = 0.0

    last_obs = np.full(L, np.nan)   # 上次观测值(带噪)
    last_seen = np.full(L, -1)      # 上次观测时刻

    # 累积器: [n_bins, 4 方法] x {static, dyn}
    acc = {k: np.zeros((len(GAP_BINS) - 1, 4)) for k in ("static", "dyn")}
    cnt = {k: np.zeros(len(GAP_BINS) - 1) for k in ("static", "dyn")}

    pos = rng.integers(L)
    for t in range(T):
        # 演化(与观测无关)
        dyn_state = dyn_state + rates
        true_state = static.copy(); true_state[dyn_idx] = dyn_state

        # 运动
        if rng.random() < 0.05:
            pos = rng.integers(L)
        else:
            pos = (pos + rng.integers(-3, 4)) % L
        win = (pos + np.arange(W)) % L
        last_obs[win] = true_state[win] + rng.normal(0, SIGMA_OBS, W)
        last_seen[win] = t

        # 查询
        if t % QUERY_EVERY == 0 and t > 0:
            seen = ~np.isnan(last_obs)
            gap = t - last_seen[seen]
            cells = np.where(seen)[0]
            est = np.stack([
                last_obs[cells],                                  # Replay
                last_obs[cells] + r_bar * gap,                    # EvolveAll
                last_obs[cells] + rate_hat[cells] * gap,          # Hybrid
                last_obs[cells] + rate_cell[cells] * gap,         # HybridOracle
            ], axis=1)
            err = np.abs(est - true_state[cells][:, None])        # (n_seen, 4)
            b = np.digitize(gap, GAP_BINS) - 1
            # 逐格归类: 静态/动态分开记账; cnt 只按格计一次(与方法无关)
            for j in range(len(cells)):
                kk = "dyn" if is_dyn[cells[j]] else "static"
                bi = b[j]
                acc[kk][bi] += err[j]
                cnt[kk][bi] += 1
    return acc, cnt


def main():
    acc = {k: np.zeros((len(GAP_BINS) - 1, 4)) for k in ("static", "dyn")}
    cnt = {k: np.zeros(len(GAP_BINS) - 1) for k in ("static", "dyn")}
    for _ in range(EPISODES):
        a, c = run_episode()
        for k in acc:
            acc[k] += a[k]; cnt[k] += c[k]

    names = ["Replay", "EvolveAll", "Hybrid", "Oracle"]
    for kind in ("static", "dyn"):
        print(f"\n===== {kind} cells: MAE vs gap Δ =====")
        print(f"{'Δ bin':>12} | " + " | ".join(f"{n:>10}" for n in names) + " |   n")
        for bi in range(len(GAP_BINS) - 1):
            if cnt[kind][bi] < 50:
                continue
            lo, hi = GAP_BINS[bi], min(GAP_BINS[bi + 1], 999)
            row = " | ".join(f"{acc[kind][bi, k] / cnt[kind][bi]:10.3f}" for k in range(4))
            print(f"{f'{lo}-{hi}':>12} | {row} | {int(cnt[kind][bi]):5d}")

        # 斜率估计: 对每个方法在 Δ∈[8, 256] 上做一次最小二乘
        xs, ys = [], []
        for bi in range(len(GAP_BINS) - 1):
            if 8 <= GAP_BINS[bi] <= 256 and cnt[kind][bi] >= 50:
                mid = (GAP_BINS[bi] + min(GAP_BINS[bi + 1], 300)) / 2
                xs.append(mid)
                ys.append([acc[kind][bi, k] / cnt[kind][bi] for k in range(4)])
        if xs:
            X = np.column_stack([np.array(xs), np.ones(len(xs))])
            slopes = np.linalg.lstsq(X, np.array(ys), rcond=None)[0][0]
            print("  slope (MAE per unit Δ, Δ∈[8,256]):",
                  {n: round(float(s), 4) for n, s in zip(names, slopes)})


if __name__ == "__main__":
    main()
