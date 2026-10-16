"""WM_20 toy: Context as Memory —— 检索条件选择: 相关性 vs 时间近邻 vs 随机 vs 全历史

对应关系(与 Context-as-Memory 机制对齐):
  2D 网格世界 + 位姿(位置, 朝向) + 52.67° FOV 楔形视野
      <-> 论文的设定: 相机约束在 XY 平面、只绕 z 轴旋转(附录 B: FOV 52.67°)
  历史帧的可见格集合 V_i <-> 由相机位姿决定的 FOV 共可见关系
  协可见度 o_i = |V_i ∩ V_q| / |V_q|
      <-> 论文的 FOV overlap 检测(Fig. 4: 左右两条射线求交的离散化版本)
  生成帧漂移 drift(t) = 0.03 + 0.09·t/T
      <-> HC 评测的误差累积: 越晚生成的帧累积误差越大, 早期帧更干净
  未覆盖格 -> 幻觉误差 1.0 <-> 检索不到相关上下文时只能靠生成先验编
策略(k=20 上下文; 其中 1 帧固定为最近生成帧——对应论文「预测序列首帧也
      作为上下文保证连续性」):
  recent-1      仅最近帧(≈ Table 1 的「1st Frame as Context」)
  temporal-20   最近 20 帧(≈ DFoT 的固定窗口)
  random-20     历史随机 20 帧(≈「1st Frame + Random Context」)
  fov-20        协可见度 top-19(≈ FOV 过滤后按相关性取, 介于 FOV+Random
                 与 FOV+Non-adj 之间的排序版)
  fov-nonadj-20 协可见度过滤 + 时间去冗余(选中帧间隔 >= 4 步, ≈ Non-adj:
                 连续帧冗余, 每组只取一帧)
  all-history   全部历史(「全历史上下文」上界, 超预算地板)
查询: 每 10 步一次, 用当前位姿的可见格做「重建」; 每格误差取
      drift(最早覆盖该格的选中帧)——早期帧漂移小; 未覆盖则 1.0。
      按「该格距上次被观测的时长 Δ」分桶(记忆只含 t 之前的帧)。
Part 2: fov-nonadj 的 k 扫描(1/5/10/20/30)——对照论文 Table 2 的
        context size 消融(收益递减、20 附近是好的权衡点)。
写入端理想化: 覆盖即恢复、无视角/遮挡误差——只隔离「选哪些帧当条件」
这一个变量(与 WM_01/WM_02 toy 同一精神)。
"""

import numpy as np

rng = np.random.default_rng(2020)

G = 48              # 网格边长
R = 9.0             # 视野半径(格)
FOV = np.deg2rad(52.67)   # 论文相机的 FOV
SPEED = 0.55        # 每步位移(格)
T = 1200            # 每 episode 步数
EPISODES = 25
QUERY_EVERY = 10
K = 20              # 上下文帧数(论文 Table 2 的权衡点)
SIGMA0, KAPPA = 0.03, 0.09   # drift(t) = SIGMA0 + KAPPA * t / T
SIGMA_H = 1.0       # 幻觉误差(未覆盖)
GAP = 4             # non-adj: 选中帧的最小时间间隔
COV_TH = 0.05       # FOV 过滤阈值

ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
CENTERS = np.stack([ii.ravel(), jj.ravel()], 1) + 0.5
NC = G * G

BINS = [0, 20, 80, 240]   # Δ 边界; bin 0 = first(从未见过), 共 5 桶
NB = len(BINS) + 1
NAMES = ["recent-1", "temporal-20", "random-20", "fov-random-20", "fov-20",
         "fov-nonadj-20", "all-history"]
KS = [1, 5, 10, 20, 30]


def vis_mask(pos, heading):
    rel = CENTERS - pos
    d = np.hypot(rel[:, 0], rel[:, 1])
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    diff = np.angle(np.exp(1j * (ang - heading)))
    return (d <= R) & (np.abs(diff) <= FOV / 2)


def drift(tau):
    return SIGMA0 + KAPPA * np.asarray(tau, dtype=float) / T


def make_anchors():
    c = G / 2.0
    return np.array([[c, c], [c - 14, c - 10], [c + 13, c - 12],
                     [c - 10, c + 14], [c + 12, c + 13]])


def sel_temporal(t, k):
    return list(range(max(0, t - k), t))


def sel_random(t, k):
    n = min(k - 1, t)
    idx = rng.choice(t, size=n, replace=False).tolist() if n > 0 else []
    return sorted(set(idx) | {t - 1})


def sel_fov_random(t, o, k):
    """FOV 过滤后在候选中随机取(≈ 论文 Table 3 的 FOV+Random)"""
    cand = np.where(o >= COV_TH)[0]
    n = min(k - 1, len(cand))
    idx = rng.choice(cand, size=n, replace=False).tolist() if n > 0 else []
    return sorted(set(idx) | {t - 1})


def sel_fov(t, o, k):
    n = min(k - 1, t)
    idx = np.argsort(-o)[:n].tolist() if n > 0 else []
    return sorted(set(idx) | {t - 1})


def sel_fov_nonadj(t, o, k):
    """协可见度过滤 + 时间去冗余(每 GAP 步至多 1 帧), 不足则放宽补齐"""
    need = k - 1
    picked = []
    if need > 0:
        cand = np.where(o >= COV_TH)[0]
        order = cand[np.argsort(-o[cand])]
        for f in order:
            if all(abs(int(f) - g) >= GAP for g in picked):
                picked.append(int(f))
                if len(picked) >= need:
                    break
        if len(picked) < need:
            for f in order:
                if int(f) not in picked:
                    picked.append(int(f))
                    if len(picked) >= need:
                        break
    return sorted(set(picked) | {t - 1})


def earliest_cover(sel, masks):
    earliest = np.full(NC, np.inf)
    for f in sel:
        m = masks[f]
        earliest[m] = np.minimum(earliest[m], f)
    return earliest


def run_episode():
    anchors = make_anchors()
    pos = anchors[0].copy()
    heading = rng.uniform(0, 2 * np.pi)
    target = anchors[int(rng.integers(1, len(anchors)))]
    H = np.zeros((T, NC), dtype=np.float32)
    masks = []
    last_obs = np.full(NC, -1)
    first_obs = np.full(NC, np.inf)

    acc = np.zeros((NB, len(NAMES)))
    cov = np.zeros((NB, len(NAMES)))
    cnt = np.zeros(NB)
    k_err = np.zeros(len(KS))
    k_long = np.zeros(len(KS))
    k_n = 0
    k_nlong = 0

    for t in range(T):
        if rng.random() < 0.012:      # 场景切换(跳到别的锚点)
            pos = anchors[int(rng.integers(0, len(anchors)))].copy()
            heading = rng.uniform(0, 2 * np.pi)
            target = anchors[int(rng.integers(0, len(anchors)))]
        des = np.arctan2(target[1] - pos[1], target[0] - pos[0])
        dh = np.angle(np.exp(1j * (des - heading)))
        heading += float(np.clip(dh, -0.10, 0.10)) + rng.normal(0, 0.04)
        pos = pos + SPEED * np.array([np.cos(heading), np.sin(heading)])
        pos = np.clip(pos, 2.0, G - 2.0)
        if np.hypot(pos[0] - target[0], pos[1] - target[1]) < 2.0:
            target = anchors[int(rng.integers(0, len(anchors)))]
        m = vis_mask(pos, heading)
        masks.append(m)
        H[t] = m

        if t >= 60 and t % QUERY_EVERY == 0:
            q = m                      # 待生成帧应保持一致的内容 = 当前位姿可见格
            qf = q.astype(np.float32)
            nq = max(int(q.sum()), 1)
            qidx = np.where(q)[0]
            overlaps = H[:t] @ qf
            o = overlaps / nq

            sels = {
                "recent-1": [t - 1],
                "temporal-20": sel_temporal(t, K),
                "random-20": sel_random(t, K),
                "fov-random-20": sel_fov_random(t, o, K),
                "fov-20": sel_fov(t, o, K),
                "fov-nonadj-20": sel_fov_nonadj(t, o, K),
            }

            # 分桶: 该格距上次被观测的时长(记忆只含 t 之前)
            bins = np.empty(len(qidx), dtype=int)
            for i, c in enumerate(qidx):
                if last_obs[c] < 0:
                    bins[i] = 0
                else:
                    bins[i] = int(np.digitize(t - last_obs[c], BINS))
            np.add.at(cnt, bins, 1)

            for ni, name in enumerate(NAMES):
                if name == "all-history":
                    earliest = first_obs     # 全历史: 最早(最干净)的覆盖帧
                else:
                    earliest = earliest_cover(sels[name], masks)
                e = np.where(earliest[qidx] < np.inf,
                             drift(np.minimum(earliest[qidx], T - 1)), SIGMA_H)
                cvd = (earliest[qidx] < np.inf).astype(float)
                np.add.at(acc[:, ni], bins, e)
                np.add.at(cov[:, ni], bins, cvd)

            # Part 2: fov-nonadj 的 k 扫描
            longm = bins >= 3            # Δ >= 80 的重访
            for ki, k in enumerate(KS):
                sel = sel_fov_nonadj(t, o, k)
                earliest = earliest_cover(sel, masks)
                e = np.where(earliest[qidx] < np.inf,
                             drift(np.minimum(earliest[qidx], T - 1)), SIGMA_H)
                k_err[ki] += e.mean()
                k_long[ki] += e[longm].sum()
            k_n += 1
            k_nlong += int(longm.sum())

        last_obs[m] = t
        first_obs[m] = np.minimum(first_obs[m], t)

    return acc, cov, cnt, k_err, k_long, k_n, k_nlong


def main():
    print("=" * 72)
    print("Part 1: k=20 上下文下六种检索策略, MAE 按重访间隔 Δ 分桶")
    print("  (Δ = 查询格距上次被观测的步数; drift(t)=0.03+0.09t/T; 幻觉=1.0)")
    print("=" * 72)
    acc = np.zeros((NB, len(NAMES)))
    cov = np.zeros((NB, len(NAMES)))
    cnt = np.zeros(NB)
    k_err = np.zeros(len(KS))
    k_long = np.zeros(len(KS))
    k_n, k_nlong = 0, 0
    for _ in range(EPISODES):
        a, cv, c, ke, kl, kn, knl = run_episode()
        acc += a
        cov += cv
        cnt += c
        k_err += ke
        k_long += kl
        k_n += kn
        k_nlong += knl

    labels = ["first"] + [f"{BINS[i]}-{BINS[i + 1] if i + 1 < len(BINS) else '∞'}"
                          for i in range(len(BINS))]
    hdr = " | ".join(f"{n:>13}" for n in NAMES)
    print(f"{'Δ bin':>10} | {hdr} |    n")
    for b in range(NB):
        if cnt[b] < 30:
            continue
        row = " | ".join(f"{acc[b, k] / cnt[b]:13.3f}" for k in range(len(NAMES)))
        print(f"{labels[b]:>10} | {row} | {int(cnt[b]):5d}")
    print()
    print("覆盖率(可见格中被选中上下文覆盖的比例):")
    for b in range(NB):
        if cnt[b] < 30:
            continue
        row = " | ".join(f"{cov[b, k] / cnt[b]:13.3f}" for k in range(len(NAMES)))
        print(f"{labels[b]:>10} | {row} | {int(cnt[b]):5d}")
    tot = cnt.sum()
    print()
    print("总体 MAE: " + ", ".join(f"{n}={acc[:, k].sum() / tot:.3f}"
                                    for k, n in enumerate(NAMES)))

    print()
    print("=" * 72)
    print("Part 2: fov-nonadj 的 context size k 扫描(对照论文 Table 2)")
    print("=" * 72)
    print(f"{'k':>4} | {'总体 MAE':>10} | {'长程段(Δ>=80) MAE':>16}")
    for ki, k in enumerate(KS):
        print(f"{k:>4} | {k_err[ki] / k_n:10.3f} | {k_long[ki] / max(k_nlong, 1):16.3f}")


if __name__ == "__main__":
    main()
