"""WM_26 toy: 世界-自我分离——全局体素世界状态 vs 依视角状态(腕相机设定)

设定(仿 AtlasVLA 的腕部相机长时程操纵, 40×40 有界桌面, 12 个对象分四类):
  early  静态·帧0可见后离场(初始俯瞰看到, 之后基本不再见)
  freq   静态·常重访(任务路径反复经过)
  mvis   可操纵·帧0可见(被移动一次, 之后第二遍重访)
  mhid   可操纵·帧0不可见(「抽屉里的罐头」: 初始俯瞰看不到, 首次访问才见)
腕相机装在末端执行器上: FoV 半宽 7, 随自我位姿大幅移动; 帧初始为「俯瞰」
(FoV 半宽 15——论文: 首帧通常提供最优视野, 且被永久锚定)。
任务脚本(模仿 change-cubes 式长时程): 依次访问 6 个子目标
[mvis0 → mhid0 → mvis1 → mhid1 → mvis0(重访) → mvis1(重访)],
行进 4.5/步 + 抖动, 到达后驻留; 首次访问可操纵对象时在第 3 步驻留将其移位
(位移 7~10, 模仿 pick-and-place)。
观测: 对象在 FoV 内 → 位置 + N(0,σ), 置信权重 w~U(0.4,1)(仿深度置信, Eq.5)。

四种记忆(同一观测流, 只比「怎么放 / 怎么演化」):
  Ego    依视角画布: 只存最近观测的自我系偏移, 查询 = 当前位姿 + 偏移
         (reactive VLA 无全局锚的极端版; = wm16 的 NoSME)
  Snap   静态快照(SOMA 式): 只用帧0俯瞰观测, 全局坐标, 从不更新
  Accum  全局体素 + 无限累积融合(= 论文 Table 5 row 4 "naive accumulation"):
         每格条目 (M, W) 按 Eq.4 加权融合, W 不衰减不驱逐
  Atlas  全局体素 + Eq.4 融合 + 滑窗衰减 λ + 低权驱逐 + 帧0条目永久锚定
         (完整机制; NoAnchor 变体 = 去锚, 用于锚的收益/代价分析)
查询: 每 5 步报告全部对象世界坐标。地图方案读出 = 存活条目按 W 加权平均
(模仿 cross-attention 按置信度取用), 无条目 → 桌面中心兜底猜测。
误差 = 欧氏距离。输出: 对象类别 × 方案; 遮挡时长(距上次观测步数)分桶;
锚分析(Atlas vs NoAnchor); Ego 误差-遮挡时长斜率(预期 ≈ 自运动速度)。
"""

import numpy as np

rng = np.random.default_rng(26)

L = 40.0            # 桌面边长(有界 [0,L]^2)
CELL = 2.0          # 体素边长(论文 0.025m 的 toy 放大版)
SIGMA = 0.08        # 观测噪声
LAM = 0.95          # 滑窗衰减 λ(Eq.5: W_t = λW_{t-1} + w_t; 论文未给值, toy 取 0.95)
W_EVICT = 0.02      # 条目驱逐阈值(滑窗的「遗忘」)
FOV = 7.0           # 腕相机 FoV 半宽(操纵中)
FOV0 = 15.0         # 帧初始俯瞰 FoV 半宽
SPEED = 4.5         # 行进速度
T = 110             # 总步数
QUERY_EVERY = 5
EPISODES = 20

CLASSES = ["early", "freq", "mvis", "mhid"]
SCHEMES = ["Ego", "Snap", "Accum", "Atlas"]
GAP_BUCKETS = [(1, 5), (6, 15), (16, 40), (41, 10**9)]


START = np.array([34.0, 34.0])     # 初始位姿(开局在桌面一角俯瞰)


def make_objects():
    objs = []
    for _ in range(4):                              # early: 初始俯瞰圈内, 之后不再接近
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(1.5, 3.0)
        objs.append({"pos": START + rad * np.array([np.cos(ang), np.sin(ang)]),
                     "cls": "early"})
    for base in [np.array([14.0, 24.0]), np.array([24.0, 14.0]),
                 np.array([8.0, 14.0]), np.array([14.0, 8.0])]:    # freq: 路径沿途
        objs.append({"pos": base + rng.normal(0, 1.0, 2), "cls": "freq"})
    for base in [np.array([22.0, 28.0]), np.array([28.0, 22.0])]:  # mvis: 俯瞰内
        objs.append({"pos": base + rng.normal(0, 0.5, 2), "cls": "mvis"})
    for base in [np.array([5.0, 20.0]), np.array([20.0, 5.0])]:    # mhid: 俯瞰外
        objs.append({"pos": base + rng.normal(0, 1.0, 2), "cls": "mhid"})
    for o in objs:
        o["pos"] = np.clip(o["pos"], 2, L - 2)
    return objs


def run_episode():
    objs = make_objects()
    n = len(objs)
    idx = {c: [i for i, o in enumerate(objs) if o["cls"] == c] for c in CLASSES}
    sched = [(idx["mvis"][0], 8, True), (idx["mhid"][0], 8, True),
             (idx["mvis"][1], 8, True), (idx["mhid"][1], 8, True),
             (idx["mvis"][0], 5, False), (idx["mvis"][1], 5, False)]

    pose = START.copy()
    ego_mem = [None] * n                       # 最近观测的自我系偏移
    snap_mem = [[] for _ in range(n)]          # 帧0观测
    maps = {s: [dict() for _ in range(n)]      # cell -> [M, W, anchored]
            for s in ["Accum", "Atlas", "NoAnchor"]}
    last_obs_t = [-1] * n
    obs_count = [0] * n
    results = []                               # (scheme, cls, gap, err, t)

    def observe(t, half):
        for i, o in enumerate(objs):
            if np.linalg.norm(o["pos"] - pose) <= half:
                obs = o["pos"] + rng.normal(0, SIGMA, 2)
                w = rng.uniform(0.4, 1.0)
                last_obs_t[i] = t
                obs_count[i] += 1
                ego_mem[i] = obs - pose
                if t == 0:
                    snap_mem[i].append(obs)
                for s, mp in maps.items():
                    cell = tuple(np.floor(obs / CELL).astype(int))
                    e = mp[i].get(cell)
                    if e is None:
                        mp[i][cell] = [obs.copy(), w, t == 0]
                    else:                       # Eq.4: 加权融合
                        e[0] = (e[1] * e[0] + w * obs) / (e[1] + w)
                        e[1] += w

    def decay():
        for s in ["Atlas", "NoAnchor"]:
            for mp in maps[s]:
                for cell in list(mp.keys()):
                    e = mp[cell]
                    if s == "Atlas" and e[2]:   # 帧0条目锚定: 不衰减不驱逐
                        continue
                    e[1] *= LAM
                    if e[1] < W_EVICT:
                        del mp[cell]

    def query_pos(s, i):
        if s == "Ego":
            if ego_mem[i] is None:
                return np.array([L / 2, L / 2])
            return pose + ego_mem[i]
        if s == "Snap":
            if not snap_mem[i]:
                return np.array([L / 2, L / 2])
            return np.mean(snap_mem[i], axis=0)
        mp = maps[s][i]
        if not mp:
            return np.array([L / 2, L / 2])
        M = np.array([e[0] for e in mp.values()])
        W = np.array([e[1] for e in mp.values()])
        return M.T @ W / W.sum()

    vis0 = [bool(np.linalg.norm(o["pos"] - pose) <= FOV0) for o in objs]
    observe(0, FOV0)                            # 帧0俯瞰
    si, dwell_left, move_done = 0, 0, False
    pose_trace = [pose.copy()]
    for t in range(1, T + 1):
        decay()
        tgt, dwell, will_move = sched[si]
        if dwell_left > 0:
            pose = pose + rng.normal(0, 1.2, 2)
            dwell_left -= 1
            if will_move and not move_done and dwell_left <= dwell - 3:
                # 移位(模仿 pick-and-place): 方向随机但落点离初始俯瞰圈
                # (early 对象聚集处) ≥12, 避免任务路径折回开场区域
                for _ in range(50):
                    d = rng.normal(0, 1, 2)
                    d = d / np.linalg.norm(d) * rng.uniform(7, 10)
                    newp = np.clip(objs[tgt]["pos"] + d, 2, L - 2)
                    if np.linalg.norm(newp - START) >= 12:
                        objs[tgt]["pos"] = newp
                        break
                move_done = True
            if dwell_left == 0:
                si = min(si + 1, len(sched) - 1)
                move_done = False
        else:
            v = objs[tgt]["pos"] - pose
            dist = np.linalg.norm(v)
            if dist < 1.5:
                dwell_left = dwell
            else:
                pose = pose + v / dist * min(SPEED, dist) + rng.normal(0, 0.8, 2)
        pose = np.clip(pose, 2, L - 2)
        pose_trace.append(pose.copy())
        observe(t, FOV)
        if t >= 15 and t % QUERY_EVERY == 0:
            for i, o in enumerate(objs):
                gap = (t - last_obs_t[i]) if last_obs_t[i] >= 0 else 10**9
                for s in SCHEMES + ["NoAnchor"]:
                    err = np.linalg.norm(query_pos(s, i) - o["pos"])
                    results.append((s, o["cls"], gap, err, t))
    return results, obs_count, objs, vis0, pose_trace


def main():
    allres = []
    obs_counts = {c: [] for c in CLASSES}
    frame0 = {c: [] for c in CLASSES}
    step_sizes = []
    for _ in range(EPISODES):
        res, oc, objs, v0, ptrace = run_episode()
        allres.extend(res)
        step_sizes.extend(np.linalg.norm(np.diff(np.array(ptrace), axis=0), axis=1))
        for o, cnt, f0 in zip(objs, oc, v0):
            obs_counts[o["cls"]].append(cnt)
            frame0[o["cls"]].append(f0)

    def sel(scheme, cls=None, bucket=None, tmin=0):
        return [err for (s, c, g, err, t) in allres
                if s == scheme and (cls is None or c == cls)
                and (bucket is None or (bucket[0] <= g <= bucket[1]))
                and t >= tmin and g < 10**8]

    print("===== (0) 各类对象实测观测统计 (几何核验) =====")
    for c in CLASSES:
        print(f"  {c:>6}: 平均观测次数 {np.mean(obs_counts[c]):5.1f}, "
              f"帧0可见比例 {np.mean(frame0[c]):.2f}")

    print("\n===== (1) 位置误差: 对象类别 × 方案 (全程平均, 单位: 格) =====")
    print(f"{'类别':>8} | " + " | ".join(f"{s:>8}" for s in SCHEMES))
    for c in CLASSES:
        row = " | ".join(f"{np.mean(sel(s, c)):8.3f}" for s in SCHEMES)
        print(f"{c:>8} | {row}")

    print("\n===== (1b) 同上, 但只用后段查询 (t>=70, 全部对象已至少见过一次) =====")
    print(f"{'类别':>8} | " + " | ".join(f"{s:>8}" for s in SCHEMES))
    for c in CLASSES:
        row = " | ".join(f"{np.mean(sel(s, c, tmin=70)):8.3f}" for s in SCHEMES)
        print(f"{c:>8} | {row}")

    print("\n===== (2) 位置误差: 按遮挡时长(距上次观测步数)分桶, 仅静态对象"
          "(early+freq, 排除移位干扰) =====")
    print(f"{'gap':>10} | " + " | ".join(f"{s:>8}" for s in SCHEMES))
    for b in GAP_BUCKETS:
        row = " | ".join(
            f"{np.mean([e for e in sel(s, 'early', bucket=b)] + sel(s, 'freq', bucket=b)):8.3f}"
            for s in SCHEMES)
        print(f"{str(b[0])+'-'+(str(b[1]) if b[1] < 10**8 else '∞'):>10} | {row}")

    print("\n===== (3) 锚分析: Atlas vs NoAnchor(同 λ 滑窗, 无帧0锚) =====")
    print(f"{'类别':>8} | {'Atlas':>8} | {'NoAnchor':>8} | 兜底误差参考(到中心距离)")
    for c in ["early", "mvis", "mhid"]:
        a = np.mean(sel("Atlas", c))
        w = np.mean(sel("NoAnchor", c))
        print(f"{c:>8} | {a:8.3f} | {w:8.3f}")

    # Ego 误差-遮挡时长斜率(OLS; 只用 gap<=15 的短遮挡段——长遮挡误差饱和于
    # 有界桌面尺寸, 会拉低全局拟合斜率)
    pts = [(g, e) for (s, c, g, e, t) in allres if s == "Ego" and g <= 15]
    g = np.array([p[0] for p in pts], dtype=float)
    e = np.array([p[1] for p in pts])
    slope = np.polyfit(g, e, 1)[0]
    print(f"\nEgo 误差-遮挡时长 OLS 斜率(gap<=15) = {slope:.3f} 格/步; "
          f"实测平均自运动速度 = {np.mean(step_sizes):.3f} 格/步; "
          f"长遮挡(gap>40)平均误差 {np.mean(sel('Ego', bucket=(41, 10**9))):.2f}"
          " (饱和于有界桌面)")
    print("(读法: Ego 连静态早离场对象都跟丢——误差 ∝ 累计自运动;")
    print(" Snap 对移位对象误差 ≈ 位移量, 对帧0不可见对象只能兜底;")
    print(" Accum 对重访的移位对象留有旧体素幽灵(权重不衰减);")
    print(" Atlas 靠 λ 衰减让新证据压过幽灵, 靠帧0锚保住早离场静态对象)")


if __name__ == "__main__":
    main()
