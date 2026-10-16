"""WM_23 toy: 感知-认知双记忆 vs 滑动窗口 vs 无记忆——MemoryVLA 的最小机制版

设定（把 PCMB 双流记忆缩成最小可算模型）:
  世界: 桌面 [0,100] 上 M=6 个按钮, 任务 = 按随机指定顺序逐个推按。
  两条记忆需求（对应 MemoryVLA 的感知/认知双流）:
    感知通道(物体级)  按钮被周期性遮挡(可见/遮挡时长 ~ Exp(20/25)); 遮挡期间位置做
      随机游走(σ_drift·√时长, 线性插值到段端点)。当前帧看不见的按钮只能靠记忆里的
      「最后可见快照」。
    认知通道(任务级)  推按前后按钮外观几乎不变(MemoryVLA Fig.1a 的 Push Buttons),
      任务进度只能靠记住「哪些已推过」——每次推按只产生一个一步长的瞬时事件,
      单帧策略在决策时刻永远看不到。
  推按成功 = 选对下一个按钮(进度判断正确) 且 目标按钮位置估计误差 ≤ tol。

五种策略（世界演化全部外生且共享同一实现, 只比「记多久、记多细」）:
  None      单帧策略(OpenVLA/π0 式): 进度永远当作 0; 遮挡按钮只能猜先验中心 50
  FIFO-16   滑动窗口(CronusVLA 式): 只保留最近 W=16 步内的事件与最后可见快照,
            更老的全部逐出
  PCMB      MemoryVLA: 感知流(最后可见快照持久保留, 检索噪声+随 √age 增长的合并
            噪声) + 认知流(每个已推事件以 q_cog 概率在合并中幸存)双流持久记忆
  PCMB-cog  消融: 只留认知流(论文 Table 6 "Cognitive Mem." 63.5)
  PCMB-per  消融: 只留感知流(论文 Table 6 "Perceptual Mem." 64.6)

评估分两趟（脚本教训: 首版让五种策略各自闭环, 失败提前终止后各策略的后续时间轴
不同, 分桶样本不可比; 改为「同轴评估 + 各自闭环」两趟）:
  同轴评估  oracle 时间轴(总是选对下一个按钮)上评估五种策略的逐次决策——同一批
            世界/时间轴, 每个桶里五种策略的样本完全相同, 严格可比;
  闭环评估  各策略独立闭环执行, 报告每次尝试成功率与整任务成功率。
"""

import numpy as np

rng = np.random.default_rng(23)

# ---- 世界与策略参数 ----
M = 6               # 按钮(子任务)数
TABLE = 100.0       # 桌面一维坐标范围
TOL = 6.0           # 推按成功的位置容差
SIGMA_OBS = 0.5     # 观测/记忆写入噪声
SIGMA_RET = 0.8     # 检索噪声(记忆命中时)
SIGMA_MERGE = 0.03  # PCMB 合并噪声(每 √age)
SIGMA_DRIFT = 0.35  # 遮挡期间位置随机游走强度(每 √步)
VIS_MEAN, OCC_MEAN = 20.0, 40.0   # 可见/遮挡段平均时长
MOVE_LO, MOVE_HI = 5.0, 20.0      # 两次推按之间的移动时间
W = 16.0            # FIFO 窗口(MemoryVLA 默认记忆长度 16)
Q_COG = 0.98        # PCMB 认知流: 单个已推事件在合并中幸存的概率
EPISODES = 4000
T_MAX = 800.0       # 可见性时间线生成长度(远超 episode 最长时间)

AGE_BINS = [1, 16, 32, 64, 128, 1e9]
N_BIN = len(AGE_BINS) - 1
VIS_BIN = N_BIN     # 「当前可见」单独一档(累积数组共 N_BIN+1 行)

NAMES = ["None", "FIFO-16", "PCMB", "PCMB-cog", "PCMB-per"]
NPOL = len(NAMES)
IDX = {n: i for i, n in enumerate(NAMES)}


def gen_schedule():
    """一个按钮的可见性时间线。

    返回 (边界时刻数组, 各段可见标志, 各段起点真实位置, 各段起点的记忆快照,
    遮挡段查询期漂移缓存)。遮挡段对「下一段的起点」贡献一次整段位移; 段内查询
    时刻的真实位置 = 段起点 + N(0, σ_drift·√age)——同一(段, 查询时刻)只被查询一次
    (每个按钮在序贯任务里恰为目标一次, 且所有策略共享同一决策时钟), 缓存保证
    各策略看到同一个真实位置。
    """
    times = [0.0]
    vis = [rng.random() < 0.6]
    pos = [rng.uniform(0, TABLE)]
    snap = [pos[0] + rng.normal(0, SIGMA_OBS)]
    while times[-1] < T_MAX:
        dur = rng.exponential(VIS_MEAN if vis[-1] else OCC_MEAN)
        disp = rng.normal(0, SIGMA_DRIFT * np.sqrt(dur)) if not vis[-1] else 0.0
        times.append(times[-1] + dur)
        vis.append(not vis[-1])
        pos.append(float(np.clip(pos[-1] + disp, 0.0, TABLE)))
        snap.append(pos[-1] + rng.normal(0, SIGMA_OBS))
    return np.array(times), vis, np.array(pos), np.array(snap), {}


def query(sched, t):
    """t 时刻按钮的状态: (是否可见, 遮挡年龄, 真实位置, 记忆里的最后可见快照)。"""
    times, vis, pos, snap, drift_cache = sched
    i = int(np.searchsorted(times, t, side="right")) - 1
    i = min(max(i, 0), len(vis) - 2)
    if vis[i]:
        return True, 0.0, float(pos[i]), float(snap[i])
    age = t - times[i]
    if i not in drift_cache:
        drift_cache[i] = rng.normal(0, SIGMA_DRIFT * np.sqrt(age))
    return False, age, float(pos[i] + drift_cache[i]), float(snap[i])


def pos_estimate(pi, visible, age, true, snap):
    """策略 pi 对目标按钮的位置估计。返回 (估计值, 是否丢失)。"""
    if visible:
        return true + rng.normal(0, SIGMA_OBS), False
    prior = TABLE / 2.0
    if pi == IDX["None"] or pi == IDX["PCMB-cog"]:
        return prior, True
    if pi == IDX["FIFO-16"]:
        if age <= W:
            return snap + rng.normal(0, SIGMA_RET), False
        return prior, True
    # PCMB / PCMB-per: 感知流持久保留, 检索噪声 + 随 √age 增长的合并噪声
    noise = rng.normal(0, SIGMA_RET) + rng.normal(0, SIGMA_MERGE * np.sqrt(age))
    return snap + noise, False


def believed_progress(pi, push_times, remembered, t):
    """策略 pi 在 t 时刻的进度信念(已完成推按数)。"""
    if pi == IDX["None"] or pi == IDX["PCMB-per"]:
        return 0
    if pi == IDX["FIFO-16"]:
        if len(push_times) == 0:
            return 0
        return int(np.sum(t - np.asarray(push_times) < W))
    return int(np.sum(remembered))


def run_episode(acc):
    """一个世界: 先做同轴评估(oracle 时间轴), 再让五种策略各自闭环。"""
    order = rng.permutation(M)
    moves = rng.uniform(MOVE_LO, MOVE_HI, M)
    scheds = [gen_schedule() for _ in range(M)]
    t0 = rng.uniform(0.0, 10.0)
    ts = t0 + np.concatenate([[0.0], np.cumsum(moves)[:-1]])  # oracle 决策时刻

    # --- 同轴评估: oracle 时间轴上, 前 k 次推按都已发生 ---
    for k in range(M):
        t, obj = ts[k], order[k]
        visible, age, true, snap = query(scheds[obj], t)
        if visible:
            bi = VIS_BIN
        else:
            bi = min(max(int(np.digitize(age, AGE_BINS)) - 1, 0), N_BIN - 1)
            acc["ora_age"][bi] += age
        for pi in range(NPOL):
            est, lost = pos_estimate(pi, visible, age, true, snap)
            acc["ora_n"][bi, pi] += 1
            acc["ora_err"][bi, pi] += abs(est - true)
            acc["ora_lost"][bi, pi] += lost
        # 进度: 前 k 次推按发生在 ts[:k]
        push_t = list(ts[:k])
        for pi in range(NPOL):
            rem = [bool(rng.random() < Q_COG) for _ in push_t]
            bp = believed_progress(pi, push_t, rem, t)
            acc["prog_n"][k, pi] += 1
            acc["prog_ok"][k, pi] += int(bp == k)

    # --- 闭环评估: 各策略独立执行, 失败即终止 ---
    for pi in range(NPOL):
        progress, t = 0, t0
        push_t, remembered = [], []
        while progress < M:
            obj = order[progress]
            visible, age, true, snap = query(scheds[obj], t)
            est, _ = pos_estimate(pi, visible, age, true, snap)
            bp = believed_progress(pi, push_t, remembered, t)
            ok = (bp == progress) and abs(est - true) <= TOL
            acc["cl_n"][progress, pi] += 1
            acc["cl_ok"][progress, pi] += int(ok)
            if not ok:
                break
            push_t.append(t)
            remembered.append(bool(rng.random() < Q_COG))
            progress += 1
            t += moves[progress - 1]
        acc["task_ok"][pi] += int(progress == M)


def main():
    acc = {
        "ora_n": np.zeros((N_BIN + 1, NPOL)),
        "ora_err": np.zeros((N_BIN + 1, NPOL)),
        "ora_lost": np.zeros((N_BIN + 1, NPOL)),
        "ora_age": np.zeros(N_BIN + 1),
        "prog_n": np.zeros((M, NPOL)),
        "prog_ok": np.zeros((M, NPOL)),
        "cl_n": np.zeros((M, NPOL)),
        "cl_ok": np.zeros((M, NPOL)),
        "task_ok": np.zeros(NPOL),
    }
    for _ in range(EPISODES):
        run_episode(acc)

    print(f"episodes: {EPISODES}, buttons/task: {M}, tol: {TOL}, "
          f"drift sigma: {SIGMA_DRIFT}/sqrt(step), FIFO W: {W:.0f}, q_cog: {Q_COG}")

    print("\n===== 同轴评估: 目标按钮位置 MAE vs 遮挡时长 A (visible 行 = 直接观测) =====")
    print(f"{'A bin':>10} {'Ā':>5} {'n':>6} | " + " | ".join(f"{n:>8}" for n in NAMES))
    xs, ys = [], []
    for bi in range(N_BIN + 1):
        n = acc["ora_n"][bi, 0]
        if n < 30:
            continue
        if bi == VIS_BIN:
            label, dbar = "visible", 0.0
        else:
            label = f"{AGE_BINS[bi]:.0f}-{min(AGE_BINS[bi + 1], 999):.0f}"
            dbar = acc["ora_age"][bi] / n
            xs.append(np.sqrt(dbar))
            ys.append(acc["ora_err"][bi, IDX['PCMB']] / n)
        mae = [acc["ora_err"][bi, pi] / n for pi in range(NPOL)]
        print(f"{label:>10} {dbar:5.0f} {int(n):6d} | " + " | ".join(f"{m:8.2f}" for m in mae))

    if len(xs) >= 3:
        A = np.column_stack([xs, np.ones(len(xs))])
        slope = np.linalg.lstsq(A, np.array(ys), rcond=None)[0][0]
        print(f"PCMB MAE ~ slope * sqrt(A): slope = {slope:.3f} "
              f"(理论主导项 sigma_drift*sqrt(2/pi) = {SIGMA_DRIFT * np.sqrt(2 / np.pi):.3f})")

    print("\n===== 同轴评估: 进度判断准确率 vs 已完成推按数 k =====")
    print(f"{'k':>3} {'n':>6} | " + " | ".join(f"{n:>8}" for n in NAMES))
    for k in range(M):
        n = acc["prog_n"][k, 0]
        acc_rate = [acc["prog_ok"][k, pi] / n for pi in range(NPOL)]
        print(f"{k:>3} {int(n):6d} | " + " | ".join(f"{a:8.1%}" for a in acc_rate))

    print("\n===== 闭环评估: 可达尝试数 n_k (失败即终止) =====")
    print(f"{'k':>3} | " + " | ".join(f"{n:>8}" for n in NAMES))
    for k in range(M):
        print(f"{k:>3} | " + " | ".join(f"{int(acc['cl_n'][k, pi]):8d}" for pi in range(NPOL)))

    print("\n===== 闭环评估: 每次尝试成功率 vs 子任务序号 k (分母 = 各策略自己的 n_k) =====")
    print(f"{'k':>3} | " + " | ".join(f"{n:>8}" for n in NAMES))
    for k in range(M):
        line = []
        for pi in range(NPOL):
            nk = acc["cl_n"][k, pi]
            line.append(f"{acc['cl_ok'][k, pi] / nk:8.1%}" if nk else f"{'—':>8}")
        print(f"{k:>3} | " + " | ".join(line))

    pooled_ok = acc["cl_ok"].sum(axis=0)
    pooled_n = acc["cl_n"].sum(axis=0)
    print("\n===== 闭环评估: 汇总 =====")
    print("每次尝试成功率(合并所有 k): "
          + " | ".join(f"{n}: {pooled_ok[i] / pooled_n[i]:.1%}" for i, n in enumerate(NAMES)))
    print("整任务成功率(全部 6 个按钮按对): "
          + " | ".join(f"{n}: {acc['task_ok'][i] / EPISODES:.1%}" for i, n in enumerate(NAMES)))


if __name__ == "__main__":
    main()
