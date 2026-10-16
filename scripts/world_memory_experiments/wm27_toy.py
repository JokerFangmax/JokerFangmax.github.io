"""WM_27 toy: 状态演化记忆——出视野后「怎么演化」决定误差斜率(操纵版重演)

设定: 桌面上 N=12 个「过程量」对象(液位/计数/传送带位置类状态), 每个以恒定
速率 r ~ U(0.15, 0.45)/步持续变化。部分可观测: 每个对象交替「可见块」
(长 4~12 步)与「遮挡块」(长 2~60 步, 对数均匀——覆盖短遮挡到长遮挡,
模仿操纵中物体反复进出腕相机视野)。可见时每步得到观测 o = s + N(0, σ)。

四种记忆(同一观测流, 只比「出视野后怎么演化」):
  Frozen    冻结(静态快照记忆; WM_01 Replay 主线的操纵版):
            估计 = 最后一次观测 → 误差斜率 = r(对动力学的无知)
  SlideWin  滑动窗口(最近 K=8 次观测平均): 对单调过程量没有演化能力,
            只换噪声-滞后权衡 → 斜率 ≈ r, 带滞后偏移
  NaiveDiff 朴素帧差(相邻两帧差当速率——pixel difference / optical flow
            的精神, 即论文 "naive feature difference" 消融):
            速率噪声 σ√2 放大 → 斜率 ≈ 0.8σ√2 > r, 比不演化更糟
            (对应论文 Table 3: naive 47.1 < w/o delta 52.3)
  EvoMem    条件化 delta(多尺度差分: 用窗口内最长基线 ≤45 步的两点差分
            估计速率, 模仿 Delta Tokenizer 的 source-conditioned 压缩):
            速率噪声 σ√2/Δ → 斜率 ≈ 0.8σ√2/Δ, 被基线长度压低
查询: 每 5 步报告全部已见对象的状态; 分桶按遮挡时长(距上次观测步数)。

面板 2: 速率估计 RMSE vs 基线长度 L ∈ {1,2,5,15,45}(两端点观测差/L)——
量化论文把 source-target 间隔定为 15 步(Table 4)而非相邻帧的动机。
"""

import numpy as np

rng = np.random.default_rng(27)

N_OBJ = 12
T = 200
EPISODES = 30
SIGMA = 0.45        # 观测噪声
K_WIN = 8           # SlideWin 窗口
MAX_BASE = 45       # EvoMem 速率估计的最长基线(仿论文 recent chain 跨 45 步)
QUERY_EVERY = 5
GAP_BUCKETS = [(1, 5), (6, 15), (16, 35), (36, 60)]
BASELINES = [1, 2, 5, 15, 45]

SCHEMES = ["Frozen", "SlideWin", "NaiveDiff", "EvoMem"]


def visibility_schedule():
    """每个对象的可见/遮挡块交替时刻表(可见块 4~12 步, 遮挡块 2~60 步对数均匀)。"""
    visible = np.zeros(T, dtype=bool)
    t, vis = 0, True
    while t < T:
        if vis:
            length = rng.integers(4, 13)
        else:
            length = int(np.exp(rng.uniform(np.log(2), np.log(60))))
        visible[t:min(t + length, T)] = vis
        t += length
        vis = not vis
    return visible


class Memory:
    """四种记忆共享同一观测流, 各自维护 (值估计, 速率估计)。"""

    def __init__(self):
        self.last_o = None          # (t, o)
        self.window = []            # SlideWin 最近 K 次观测
        self.naive_rate = 0.0       # 最近一对相邻观测的差分
        self.obs = []               # 全部 (t, o), 供 EvoMem 多尺度差分

    def update(self, t, o):
        if self.last_o is not None and t - self.last_o[0] == 1:
            self.naive_rate = o - self.last_o[1]        # lag-1 帧差
        self.last_o = (t, o)
        self.window.append((t, o))
        if len(self.window) > K_WIN:
            self.window.pop(0)
        self.obs.append((t, o))

    def evo_rate(self):
        """最长基线 ≤ MAX_BASE 的两点差分(跨遮挡块允许)。"""
        if len(self.obs) < 2:
            return 0.0
        t1, o1 = self.obs[-1]
        # 此前观测里最早的一条, 距今不超过 MAX_BASE(不含自身——否则 0/0)
        for (t0, o0) in self.obs[:-1]:
            if t1 - t0 <= MAX_BASE:
                return (o1 - o0) / (t1 - t0)
        # 全部更早: 退回最长可用基线(噪声仍被基线长度压低)
        t0, o0 = self.obs[-2]
        return (o1 - o0) / (t1 - t0)

    def query(self, t):
        if self.last_o is None:
            return None
        t1, o1 = self.last_o
        gap = t - t1
        return {
            "Frozen": o1,
            "SlideWin": float(np.mean([o for _, o in self.window])),
            "NaiveDiff": o1 + self.naive_rate * gap,
            "EvoMem": o1 + self.evo_rate() * gap,
        }


def run_episode():
    """返回 [(scheme, gap, err)] 与面板2的 (L, rate_err) 样本。"""
    rates = rng.uniform(0.15, 0.45, N_OBJ)
    s0 = rng.uniform(0, 100, N_OBJ)
    scheds = [visibility_schedule() for _ in range(N_OBJ)]
    mems = [Memory() for _ in range(N_OBJ)]
    results = []
    base_samples = {L: [] for L in BASELINES}
    noise_log = {}                     # (i, t) -> 该步观测噪声(用于面板2)
    for t in range(T):
        for i in range(N_OBJ):
            if scheds[i][t]:
                eps = rng.normal(0, SIGMA)
                noise_log[(i, t)] = eps
                o = s0[i] + rates[i] * t + eps
                mems[i].update(t, o)
                # 面板2: 基线 L 的两点差分(两端都有观测即可, 跨块允许)
                for L in BASELINES:
                    if t - L >= 0 and (i, t - L) in noise_log:
                        o_prev = s0[i] + rates[i] * (t - L) \
                            + noise_log[(i, t - L)]
                        base_samples[L].append(
                            (o - o_prev) / L - rates[i])
        if t % QUERY_EVERY == 0:
            for i in range(N_OBJ):
                est = mems[i].query(t)
                if est is None:
                    continue
                gap = t - mems[i].last_o[0]
                s_true = s0[i] + rates[i] * t
                for sch, e in est.items():
                    results.append((sch, gap, abs(e - s_true)))
    return results, base_samples


def main():
    allres = []
    base_acc = {L: [] for L in BASELINES}
    for _ in range(EPISODES):
        res, bs = run_episode()
        allres.extend(res)
        for L in BASELINES:
            base_acc[L].extend(bs[L])

    def sel(sch, bucket=None):
        return [e for (s, g, e) in allres
                if s == sch and (bucket is None or bucket[0] <= g <= bucket[1])]

    print(f"参数: σ={SIGMA}, r~U(0.15,0.45)(均值 0.30), K={K_WIN}, "
          f"EvoMem 最长基线 {MAX_BASE} 步; 查询 {EPISODES} 回合 × {T} 步\n")

    print("===== (1) 状态误差: 按遮挡时长(距上次观测步数)分桶 =====")
    print(f"{'遮挡时长':>10} | " + " | ".join(f"{s:>9}" for s in SCHEMES))
    for b in GAP_BUCKETS:
        row = " | ".join(f"{np.mean(sel(s, b)):9.3f}" for s in SCHEMES)
        print(f"{str(b[0])+'-'+str(b[1]):>10} | {row}")

    print("\n===== (2) 误差-遮挡时长 OLS 斜率 (gap<=60 的逐点拟合) =====")
    for s in SCHEMES:
        pts = [(g, e) for (sch, g, e) in allres if sch == s and g <= 60]
        g = np.array([p[0] for p in pts], dtype=float)
        e = np.array([p[1] for p in pts])
        slope = np.polyfit(g, e, 1)[0]
        print(f"  {s:>9}: 斜率 = {slope:.3f} /步   "
              f"(理论: Frozen/SlideWin ≈ r̄=0.30; NaiveDiff ≈ 0.8σ√2=0.51;"
              f" EvoMem ≈ 0.8σ√2/Δ)")

    print("\n===== (3) 速率估计 RMSE vs 基线长度 L (两点差分/L) =====")
    print(f"{'L(步)':>6} | {'RMSE':>8} | 理论 σ√2/L")
    for L in BASELINES:
        rmse = float(np.sqrt(np.mean(np.square(base_acc[L])))) \
            if base_acc[L] else float("nan")
        print(f"{L:>6} | {rmse:8.4f} | {SIGMA * np.sqrt(2) / L:8.4f}")

    print("\n(读法: 冻结的斜率 = r̄ = 0.30 = 对动力学的无知(WM_01 的 Replay 斜率);")
    print(" 朴素帧差斜率 0.51 > 0.30——噪声被当成了变化, 比不演化更糟")
    print("  (对应论文 Table 3: naive feature difference 47.1 < w/o delta 52.3);")
    print(" EvoMem 靠拉长差分基线把速率噪声压低 σ√2/L——论文把 source-target")
    print(" 间隔定为 15 步(Table 4)而非相邻帧, 面板(3)给出量化理由)")


if __name__ == "__main__":
    main()
