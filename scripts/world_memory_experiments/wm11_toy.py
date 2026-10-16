"""WM_11 toy: 预测 vs 承诺——遮挡时长如何决定重入场误差

世界: 30x30 环面, 3 个实体。每个实体的"计划轨迹"是恒速 + 航向角缓慢
  OU 漂移的平滑路径 (WorldDirector 里由 LLM 一次性规划); 真实轨迹 =
  计划 + 有界执行噪声 (生成器按计划渲染, 但有小的执行偏差, 用快弛豫
  OU 建模, 误差有界不累积)。
相机: 圆形视场 (半径 R), 自身沿 OU 路径游走, 实体自然进出视场。
观测: 实体在视场内时, 记录带噪位置 (σ_obs); 速度只能由相邻两次观测差分。

在实体**重新进入视场的那一刻**, 比较三种"看不见期间的演化机制":
  Frozen        冻结在最后观测位置——现有世界模型的默认行为
  Monitor       恒速外推 (最后两次观测差分出的速度)——LiveWorld 式 monitor 快进
  Orchestrated  直接报告计划位置 (计划就是世界的控制信号)——WorldDirector

按离开时长 Δ 分桶统计: 重入位置误差 + 身份绑定失败率
  (误差 > 当时该实体与其他实体最小间距的一半 → 渲染器会把外观绑到错误的
  实体上, 即 Appearance Binding 失效)。
预期: Frozen 误差随 Δ 线性增长 / Monitor 亚线性增长 (未建模加速度,
  误差 ~ σ_a·Δ^1.5 量级) / Orchestrated 与 Δ 无关——计划不是预测, 是承诺。
"""

import numpy as np

rng = np.random.default_rng(11)

# ---- 世界与实验参数 ----
L = 30.0        # 平面边长
R = 6.0         # 相机视场半径
N_ENT = 3       # 实体数
T = 4000        # 每 episode 步数
EPISODES = 100
DT = 1.0

TAU_TH = 30.0   # 航向角 OU 弛豫时间 (方向漂移的平滑度)
SIGMA_TH = 0.01   # 航向角噪声
TAU_E = 2.0     # 执行噪声弛豫时间
SIGMA_EXEC = 0.05  # 执行噪声幅度 (计划 vs 真实的偏差)
SIGMA_OBS = 0.02   # 位置观测噪声

GAP_BINS = [0, 4, 8, 16, 32, 64, 128, 256, 10**9]


def torus_delta(a, b):
    """环面上的位移向量 (每维取最短绕行方向)。"""
    d = a - b
    return (d + L / 2) % L - L / 2


def ou_step(x, tau, sigma, dt=DT):
    """一步 OU 过程: dx = -x/tau*dt + sigma*sqrt(dt)*N(0,1)"""
    return x * np.exp(-dt / tau) + sigma * np.sqrt(dt) * rng.normal(size=np.shape(x))


def run_episode():
    # 计划轨迹: 恒速 + 航向角缓慢 OU 漂移 (行人/车辆式的持续运动);
    # 真实轨迹 = 计划 + 执行噪声 (有界)
    plan = np.zeros((T, N_ENT, 2))
    speed = rng.uniform(0.15, 0.45, N_ENT)          # 各实体恒定速率
    theta = rng.uniform(0, 2 * np.pi, N_ENT)        # 航向角, 缓慢漂移
    pos = rng.uniform(5, 25, (N_ENT, 2))
    exec_err = np.zeros((N_ENT, 2))
    true = np.zeros((T, N_ENT, 2))
    for t in range(T):
        theta = (theta + ou_step(np.zeros(N_ENT), TAU_TH, SIGMA_TH)) % (2 * np.pi)
        vel = np.stack([speed * np.cos(theta), speed * np.sin(theta)], axis=1)
        pos = (pos + vel * DT) % L
        plan[t] = pos
        exec_err = ou_step(exec_err, TAU_E, SIGMA_EXEC)
        true[t] = (pos + exec_err) % L

    # 相机: 同款运动模型, 速率稍慢
    cam = np.zeros((T, 2))
    c_speed = rng.uniform(0.15, 0.35)
    c_theta = rng.uniform(0, 2 * np.pi)
    cpos = rng.uniform(5, 25, 2)
    for t in range(T):
        c_theta = (c_theta + ou_step(0.0, TAU_TH, SIGMA_TH * 1.5)) % (2 * np.pi)
        cpos = (cpos + c_speed * DT * np.array([np.cos(c_theta), np.sin(c_theta)])) % L
        cam[t] = cpos

    # 观测 + 重入事件
    last_pos = np.full((N_ENT, 2), np.nan)   # 最后观测位置 (带噪)
    last_pos2 = np.full((N_ENT, 2), np.nan)  # 倒数第二次 (差分速度用)
    last_seen = np.full(N_ENT, -1)           # 最后观测时刻
    last_seen2 = np.full(N_ENT, -1)          # 倒数第二次观测时刻
    in_fov = np.zeros(N_ENT, bool)

    names = ["Frozen", "Monitor", "Orchestrated"]
    errs = {n: [] for n in names}
    swaps = {n: [] for n in names}
    gaps = []
    for t in range(T):
        d = np.linalg.norm(torus_delta(true[t], cam[t]), axis=1)
        now = d < R
        for i in range(N_ENT):
            if now[i] and not in_fov[i] and last_seen[i] >= 0:
                # 重入时刻: 三种机制各报一个位置
                delta = t - last_seen[i]
                # 差分速度: 除以两次观测的真实间隔; 单帧观测过 → 视为 0 (退化为 Frozen)
                dt_obs = last_seen[i] - last_seen2[i]
                v_hat = ((last_pos[i] - last_pos2[i]) / dt_obs
                         if dt_obs > 0 and last_seen2[i] >= 0 else np.zeros(2))
                est = {
                    "Frozen": last_pos[i],
                    "Monitor": last_pos[i] + v_hat * delta,
                    "Orchestrated": plan[t][i],   # 计划即控制信号
                }
                other = np.delete(true[t], i, axis=0)
                min_gap = np.linalg.norm(torus_delta(other, true[t][i]), axis=1).min()
                for name, e in est.items():
                    err_i = np.linalg.norm(torus_delta(e % L, true[t][i]))
                    errs[name].append(err_i)
                    swaps[name].append(err_i > 0.5 * min_gap)
                gaps.append(delta)
            if now[i]:
                last_pos2[i] = last_pos[i]
                last_seen2[i] = last_seen[i]
                last_pos[i] = true[t][i] + rng.normal(0, SIGMA_OBS, 2)
                last_seen[i] = t
        in_fov = now
    return errs, gaps, swaps


def main():
    names = ["Frozen", "Monitor", "Orchestrated"]
    errs = {n: [] for n in names}
    swaps = {n: [] for n in names}
    gaps = []
    for _ in range(EPISODES):
        e, g, s = run_episode()
        gaps += g
        for n in names:
            errs[n] += e[n]
            swaps[n] += s[n]

    gaps = np.array(gaps)
    b = np.digitize(gaps, GAP_BINS) - 1
    # 环面上均匀随机位置的期望距离 (完全遗忘锚点): E|U1-U2| in 2D ≈ 0.3826·L
    anchor = 0.3826 * L
    print(f"re-entries total: {len(gaps)};  uniform-random anchor E[d] ≈ {anchor:.2f}")
    print(f"\n===== re-entry position error (MAE) vs out-of-view Δ =====")
    print(f"{'Δ bin':>10} | " + " | ".join(f"{n:>13}" for n in names) + " |    n")
    for bi in range(len(GAP_BINS) - 1):
        m = b == bi
        if m.sum() < 30:
            continue
        lo, hi = GAP_BINS[bi], min(GAP_BINS[bi + 1], 999)
        row = " | ".join(f"{np.mean(np.array(errs[n])[m]):13.3f}" for n in names)
        print(f"{f'{lo}-{hi}':>10} | {row} | {m.sum():4d}")

    print(f"\n===== identity-binding failure rate (err > half min entity gap) =====")
    print(f"{'Δ bin':>10} | " + " | ".join(f"{n:>13}" for n in names))
    for bi in range(len(GAP_BINS) - 1):
        m = b == bi
        if m.sum() < 30:
            continue
        lo, hi = GAP_BINS[bi], min(GAP_BINS[bi + 1], 999)
        row = " | ".join(f"{np.mean(np.array(swaps[n])[m]):13.3f}" for n in names)
        print(f"{f'{lo}-{hi}':>10} | {row}")

    # 汇总: 短/中/长遮挡三档 (环面世界误差会饱和, 斜率不适用, 看分档 MAE)
    print("\n===== summary (MAE @ short/mid/long occlusion) =====")
    for n in names:
        e = np.array(errs[n])
        short = e[(gaps >= 16) & (gaps < 32)].mean() if ((gaps >= 16) & (gaps < 32)).sum() else float("nan")
        mid = e[(gaps >= 64) & (gaps < 128)].mean()
        long_ = e[gaps >= 256].mean()
        print(f"{n:>13}: Δ16-32 {short:6.3f} | Δ64-128 {mid:6.3f} | Δ256+ {long_:6.3f}"
              f"   (uniform anchor {anchor:.2f})")


if __name__ == "__main__":
    main()
