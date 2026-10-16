"""WM_10 toy: E/R 解耦与 monitor 预算——冻结误差 vs 演化漂移 vs 预算淘汰

LiveWorld 的三个可量化机制, 在 WM_01 同款环形世界上做最小标本:

机制 1 (解耦的收益): 冻结式记忆对离场实体的位置误差随离开时长线性增长,
  斜率 = 实体速度 (完全不知道会动 / 动多快); monitor 式记忆按学到的
  速率外推, 斜率压到速率估计误差 |v̂-v|。

机制 2 (预算的代价): 活跃 monitor 上限 M (论文设 M=3) —— 超限时丢弃
  离观察者最远的 monitor, 被丢弃实体的记忆退回冻结,
  误差斜率从估计误差跳回实体速度。

机制 3 (再观测的同步): 实体重入视野时记忆用带噪观测校准, 误差重置到
  观测噪声地板 —— 显式状态让「重访即校准」成为可能 (Δ=0 桶)。

世界: 周长 1200 的环, K=8 个实体沿环恒速运动 (v_i ~ U(0.5,1.5))。
  观察者视窗宽 40, 沿 LiveBench 式 A→B→A→B→A 轨迹在环上两个锚点间
  往返, 每回合 20 步, 共 12 回合。
  误差在「展开的数轴」上记账 (真值 = 出发位置 + v·t, 不取模),
  冻结误差因此不会因绕环而饱和。
记忆系统:
  Frozen          报上次观测位置 (KV cache / 检索式)
  Monitor         每个被监视实体按学到的速率 v̂_i = v_i + N(0,0.05) 外推;
                  重入视野时同步校准
  MonitorOracle   同 Monitor 但 v̂_i = v_i (完美演化引擎)
  MonitorBudget   同 Monitor 但活跃 monitor 上限 M=3 < K=8
"""

import numpy as np

rng = np.random.default_rng(10)

# ---- 参数 ----
L = 1200.0          # 环周长 (须 > 2*max位移=720, 保证首观测的 lap 无歧义)
K = 8               # 实体数
M_BUDGET = 3        # MonitorBudget 的活跃 monitor 上限
WIN = 40            # 观察者视窗宽
STEPS_PER_ROUND = 20
ROUNDS = 12
EPISODES = 400
SIGMA_OBS = 0.05    # 观测噪声 (位置)
SIGMA_VHAT = 0.05   # monitor 学到的速率误差
DELTA_BINS = [0, 1, 2, 3, 4, 6, 12]   # 距上次观测的回合数分桶


def run_episode(systems):
    v = rng.uniform(0.5, 1.5, K)                 # 实体速率
    x0 = rng.uniform(0, L, K)                    # 出发位置 (环坐标)
    vhat = v + rng.normal(0, SIGMA_VHAT, K)      # monitor 学到的速率

    # 观察者 A→B→A→B→A: 环上两个锚点间往返
    A, B = L * 0.2, L * 0.7
    anchors = [A, B] * (ROUNDS // 2)

    # 展开坐标: 真值轨迹 x_true_i(t) = x0_i + v_i·t (不取模)
    # 记忆估计同样在展开坐标上外推; 观测时把环坐标观测值与展开估计对齐
    # (对齐 = 加上最近一圈的整数倍 L)
    t_total = 0
    mem = {name: {"est": x0.copy(), "last_seen_r": np.zeros(K, int),
                  "monitored": np.ones(K, bool),
                  "ever_seen": np.zeros(K, bool)} for name in systems}
    nbin = len(DELTA_BINS) - 1
    acc = np.zeros((nbin, len(systems)))
    cnt = np.zeros(nbin)

    for rd in range(ROUNDS):
        prev_anchor = x0.mean() if rd == 0 else anchors[rd - 1]
        for step in range(STEPS_PER_ROUND):
            u = (step + 1) / STEPS_PER_ROUND
            observer = (prev_anchor * (1 - u) + anchors[rd] * u) % L
            t_total += 1
            x_true = x0 + v * t_total            # 展开真值

            # 观测: 环坐标距离 < WIN/2
            ring_dist = np.abs((x_true % L) - observer)
            ring_dist = np.minimum(ring_dist, L - ring_dist)
            vis = ring_dist < WIN / 2
            if vis.any():
                for name in systems:
                    m = mem[name]
                    # 观测值 = 环坐标带噪; 与展开估计对齐后写入
                    # (shift 必须用展开坐标 est 而非 est%L: 跨圈后环坐标
                    #  无法判断 lap, 会把 est 对齐到错圈, 误差 ±L 且永久传播)
                    z = ((x_true[vis] % L) + rng.normal(0, SIGMA_OBS, vis.sum()))
                    shift = np.round((m["est"][vis] - z) / L) * L
                    m["est"][vis] = z + shift
                    m["last_seen_r"][vis] = rd
                    m["ever_seen"][vis] = True
                    if name == "MonitorBudget":
                        m["monitored"][vis] = True
                        act = np.where(m["monitored"])[0]
                        if len(act) > M_BUDGET:
                            dd = np.minimum(np.abs((x_true[act] % L) - observer),
                                            L - np.abs((x_true[act] % L) - observer))
                            far = act[np.argsort(-dd)]
                            m["monitored"][far[:len(act) - M_BUDGET]] = False

            # 记忆演化: 被 monitor 覆盖且当前不可见的实体按各自速率外推
            for name in systems:
                m = mem[name]
                if name == "Frozen":
                    continue
                rate = v if name == "MonitorOracle" else vhat
                evo = (~vis) & m["monitored"] & m["ever_seen"]
                m["est"][evo] = m["est"][evo] + rate[evo]

        # 回合末查询: 当前不可见且见过的实体
        observer = anchors[rd]
        ring_dist = np.abs((x_true % L) - observer)
        ring_dist = np.minimum(ring_dist, L - ring_dist)
        hidden = (ring_dist >= WIN / 2) & mem[systems[0]]["ever_seen"]
        for j in np.where(hidden)[0]:
            delta = rd - mem[systems[0]]["last_seen_r"][j]
            b = np.digitize(delta, DELTA_BINS) - 1
            if not (0 <= b < nbin):
                continue
            cnt[b] += 1
            for si, name in enumerate(systems):
                acc[b, si] += abs(mem[name]["est"][j] - x_true[j])
    return acc, cnt


def main():
    systems = ["Frozen", "Monitor", "MonitorOracle", "MonitorBudget"]
    nbin = len(DELTA_BINS) - 1
    acc = np.zeros((nbin, len(systems)))
    cnt = np.zeros(nbin)
    for _ in range(EPISODES):
        a, c = run_episode(systems)
        acc += a
        cnt += c

    print(f"环长 {L}, K={K} 实体 (v~U(0.5,1.5)), 视窗宽 {WIN}, "
          f"{ROUNDS} 回合 x {STEPS_PER_ROUND} 步, {EPISODES} episodes")
    print(f"monitor 速率误差 σ_v̂={SIGMA_VHAT}, 观测噪声 σ={SIGMA_OBS}, "
          f"Budget M={M_BUDGET}\n")
    print(f"{'Δ(回合)':>8} | " + " | ".join(f"{n:>14}" for n in systems) + " |     n")
    print("-" * 80)
    for b in range(nbin):
        if cnt[b] < 30:
            continue
        lo, hi = DELTA_BINS[b], DELTA_BINS[b + 1]
        row = " | ".join(f"{acc[b, s] / cnt[b]:>14.2f}"
                         for s in range(len(systems)))
        print(f"{lo:>4}-{hi:<4} | {row} | {int(cnt[b]):5d}")

    # 斜率 (Δ>=1 的桶, 单位: 误差/回合)
    xs, ys = [], []
    for b in range(nbin):
        if DELTA_BINS[b] >= 1 and cnt[b] >= 30:
            xs.append((DELTA_BINS[b] + DELTA_BINS[b + 1]) / 2)
            ys.append([acc[b, s] / cnt[b] for s in range(len(systems))])
    X = np.column_stack([np.array(xs), np.ones(len(xs))])
    slopes = np.linalg.lstsq(X, np.array(ys), rcond=None)[0][0]
    print("\n斜率 (误差/回合, Δ>=1 拟合):",
          {n: round(float(s), 2) for n, s in zip(systems, slopes)})
    print(f"理论: Frozen 斜率 ≈ E[v]·{STEPS_PER_ROUND} = "
          f"{1.0 * STEPS_PER_ROUND:.1f}/回合; Monitor 斜率 ≈ "
          f"E|N(0,{SIGMA_VHAT})|·{STEPS_PER_ROUND} = "
          f"{0.7979 * SIGMA_VHAT * STEPS_PER_ROUND:.2f}/回合")


if __name__ == "__main__":
    main()
