"""WM_21 toy: WorldPack —— 等预算下的动态帧压缩: 几何选择 x 轨迹打包

对应关系(与 WorldPack 机制对齐):
  2D 网格世界 + 位姿(位置, 朝向) + 52.67° FOV 楔形视野 <-> LoopNav 的环游导航
  协可见度 o_i = |V_i ∩ V_q| / |V_q|                 <-> 论文 Eq. 6 的 FoV overlap
                                                      (论文用 10,000 次 Monte Carlo
                                                       采样估计, toy 用离散格精确算)
  分数 s_i = 1.0·o_i - 0.2·(t-i)/t                   <-> 论文 Eq. 7 (w_o=1.0, w_t=0.2)
  压缩率 r ∈ {1,4,16,64,256}, 预算以「未压缩帧」为单位, 1/r/帧 <-> Eq. 4/5 的 ℓ_i 与 L_pack
  压缩损失 σ(r): 1x=0.05, 4x=0.15, 16x=0.35, 64x=0.60, 256x=0.85
  未覆盖格 -> 幻觉误差 1.0
  误差模型: 每格取「覆盖它的选中帧中保真度最高者」的 σ(r)(内容寻址的
  cross-attention 倾向于最匹配/最清晰的 token——写入端理想化, 只隔离
  「哪些帧进上下文、各占多少预算」这一个变量)

Part 1: 等预算(4 个未压缩帧单位)下五种策略, 对照论文 Table 3 的四路消融:
  baseline-4        最近 4 帧, 不压缩(Bar et al. 的标准做法)
  retrieval-4       FoV 分数 top-4, 不压缩(= WorldMem / Context-as-Memory 的
                    检索式记忆: 固定窗口, 非选中帧全部丢弃——GS only)
  temporal-pack-22  最近 22 帧打包: 2@1x + 4@4x + 16@16x(= FramePack 的时间
                    近邻分级——TP only)
  worldpack-22      22 帧打包, 分级按 Eq. 7 的几何分数排(TP + GS)
  floor-all-raw     窗口内全部帧不压缩(超预算地板)
Part 2: 压缩深度扫描(全部按几何分数分级, 预算恒为 4 单位), 对照论文
  Table 12「压得太狠会崩」:
  L1: 4@1x(4 帧) / L2: 2@1x+4@4x+16@16x(22 帧) /
  L3: 1@1x+4@4x+16@16x+64@64x(85 帧) / L4: 1@1x+3@4x+12@16x+48@64x+192@256x(256 帧)
轨迹: A 为锚点的环游(A->X->A, X 一半取近锚点一半取远锚点), 制造不同
  重访间隔 Δ; 按 Δ 分桶报告 MAE。
"""

import numpy as np

rng = np.random.default_rng(2121)

G = 56              # 网格边长
R = 9.0             # 视野半径(格)
FOV = np.deg2rad(52.67)
SPEED = 0.6
T = 1500            # 每 episode 步数
EPISODES = 25
QUERY_EVERY = 10
WINDOW = 400        # 检索候选窗口(帧数)
WO, WT = 1.0, 0.2   # Eq. 7 的权重(论文设置)
SIG = {1: 0.05, 4: 0.15, 16: 0.35, 64: 0.60, 256: 0.85}
SIGMA_H = 1.0

ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
CENTERS = np.stack([ii.ravel(), jj.ravel()], 1) + 0.5
NC = G * G

BINS = [0, 30, 80, 160]   # Δ 边界; bin 0 = first(从未见过), 共 5 桶
NB = len(BINS) + 1

NAMES = ["baseline-4", "retrieval-4", "temporal-pack-22", "worldpack-22", "floor-all-raw"]
LADDERS = [
    ("L1: 4@1x                    (4 帧)", [(1, 4)]),
    ("L2: 2@1x+4@4x+16@16x       (22 帧)", [(1, 2), (4, 4), (16, 16)]),
    ("L3: 1@1x+4@4x+16@16x+64@64x (85 帧)", [(1, 1), (4, 4), (16, 16), (64, 64)]),
    ("L4: 1@1x+3@4x+12@16x+48@64x+192@256x (256 帧)",
     [(1, 1), (4, 3), (16, 12), (64, 48), (256, 192)]),
]


def vis_mask(pos, heading):
    rel = CENTERS - pos
    d = np.hypot(rel[:, 0], rel[:, 1])
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    diff = np.angle(np.exp(1j * (ang - heading)))
    return (d <= R) & (np.abs(diff) <= FOV / 2)


def make_anchors():
    c = G / 2.0
    A = np.array([c, c])
    near = [A + np.array(d) for d in [(5, 0), (-5, 0), (0, 5), (0, -5)]]
    far = [np.array([9, 9]), np.array([G - 9, 9]),
           np.array([9, G - 9]), np.array([G - 9, G - 9])]
    return A, near, far


def budget_of(ladder):
    return sum(cnt / r for r, cnt in ladder)


def eval_assignment(assign, qidx):
    """assign: [(帧索引, 压缩率)]; 返回每个查询格的误差(min-σ 模型)"""
    best = np.full(NC, np.inf)
    for f, r in assign:
        m = MASKS[f]
        best[m] = np.minimum(best[m], SIG[r])
    return np.where(best[qidx] < np.inf, best[qidx], SIGMA_H), best[qidx] < np.inf


def pack_by_order(order, ladder):
    """按优先级顺序 order 分配 ladder 的 (率, 数量) 槽位"""
    assign = []
    p = 0
    for r, cnt in ladder:
        assign.extend([(int(f), r) for f in order[p:p + cnt]])
        p += cnt
    return assign


MASKS = None   # 当前 episode 的历史帧可见掩码(供 eval_assignment 引用)


def run_episode():
    global MASKS
    A, near, far = make_anchors()
    pos = A.copy()
    heading = rng.uniform(0, 2 * np.pi)

    def new_target():
        pool = near if rng.random() < 0.5 else far
        return pool[int(rng.integers(0, len(pool)))].copy()

    target = new_target()
    going_home = False
    leg = 0

    masks = []
    H = np.zeros((T, NC), dtype=np.float32)
    last_obs = np.full(NC, -1)

    acc = np.zeros((NB, len(NAMES)))
    cov = np.zeros((NB, len(NAMES)))
    cnt = np.zeros(NB)
    l_acc = np.zeros((NB, len(LADDERS)))
    l_n = np.zeros(NB)

    for t in range(T):
        des = np.arctan2(target[1] - pos[1], target[0] - pos[0])
        dh = np.angle(np.exp(1j * (des - heading)))
        # 转向速率较高(0.35 rad/步): 快速「转走-转回」制造 Δ∈(5,25] 的中段
        # 重访——对应 LoopNav 小导航范围(ABA-5)里 FramePack 类方法获益的区间
        heading += float(np.clip(dh, -0.35, 0.35)) + rng.normal(0, 0.05)
        pos = pos + SPEED * np.array([np.cos(heading), np.sin(heading)])
        pos = np.clip(pos, 2.0, G - 2.0)
        leg += 1
        if np.hypot(pos[0] - target[0], pos[1] - target[1]) < 2.0 or leg > 300:
            if going_home:
                target = new_target()
                going_home = False
            else:
                target = A.copy()
                going_home = True
            leg = 0
        m = vis_mask(pos, heading)
        masks.append(m)
        H[t] = m

        if t >= 80 and t % QUERY_EVERY == 0:
            MASKS = masks
            q = m
            qf = q.astype(np.float32)
            nq = max(int(q.sum()), 1)
            qidx = np.where(q)[0]
            lo = max(0, t - WINDOW)
            Hw = H[lo:t]
            o = (Hw @ qf) / nq                       # Eq. 6 的协可见度
            age = (t - (lo + np.arange(len(o)))) / t
            s = WO * o - WT * age                    # Eq. 7
            rank = np.argsort(-s) + lo               # 转回全局帧索引
            by_recency = list(range(t - 1, max(0, t - 22) - 1, -1))  # 最新在前

            assigns = {
                "baseline-4": [(f, 1) for f in range(t - 4, t)],
                "retrieval-4": [(int(f), 1) for f in rank[:4]],
                "temporal-pack-22": [(int(f), 1) for f in by_recency[:2]]
                                   + [(int(f), 4) for f in by_recency[2:6]]
                                   + [(int(f), 16) for f in by_recency[6:22]],
                "worldpack-22": pack_by_order(rank, [(1, 2), (4, 4), (16, 16)]),
                "floor-all-raw": [(f, 1) for f in range(lo, t)],
            }

            bins = np.empty(len(qidx), dtype=int)
            for i, c in enumerate(qidx):
                if last_obs[c] < 0:
                    bins[i] = 0
                else:
                    bins[i] = int(np.digitize(t - last_obs[c], BINS))
            np.add.at(cnt, bins, 1)

            for ni, name in enumerate(NAMES):
                e, cvd = eval_assignment(assigns[name], qidx)
                np.add.at(acc[:, ni], bins, e)
                np.add.at(cov[:, ni], bins, cvd.astype(float))

            np.add.at(l_n, bins, 1)
            for li, (_, ladder) in enumerate(LADDERS):
                e, _ = eval_assignment(pack_by_order(rank, ladder), qidx)
                np.add.at(l_acc[:, li], bins, e)

        last_obs[m] = t

    return acc, cov, cnt, l_acc, l_n


def main():
    print("=" * 72)
    print("Part 1: 等预算(4 个未压缩帧单位)下的记忆策略, MAE 按 Δ 分桶")
    print("  (σ: 1x=0.05, 4x=0.15, 16x=0.35; 未覆盖=1.0; 检索窗口 %d 帧)" % WINDOW)
    print("=" * 72)
    acc = np.zeros((NB, len(NAMES)))
    cov = np.zeros((NB, len(NAMES)))
    cnt = np.zeros(NB)
    l_acc = np.zeros((NB, len(LADDERS)))
    l_n = np.zeros(NB)
    for _ in range(EPISODES):
        a, cv, c, la, ln = run_episode()
        acc += a
        cov += cv
        cnt += c
        l_acc += la
        l_n += ln

    labels = ["first"] + [f"{BINS[i]}-{BINS[i + 1] if i + 1 < len(BINS) else '∞'}"
                          for i in range(len(BINS))]
    hdr = " | ".join(f"{n:>16}" for n in NAMES)
    print(f"{'Δ bin':>10} | {hdr} |    n")
    for b in range(NB):
        if cnt[b] < 30:
            continue
        row = " | ".join(f"{acc[b, k] / cnt[b]:16.3f}" for k in range(len(NAMES)))
        print(f"{labels[b]:>10} | {row} | {int(cnt[b]):5d}")
    print()
    print("覆盖率(可见格中被上下文覆盖的比例):")
    for b in range(NB):
        if cnt[b] < 30:
            continue
        row = " | ".join(f"{cov[b, k] / cnt[b]:16.3f}" for k in range(len(NAMES)))
        print(f"{labels[b]:>10} | {row} | {int(cnt[b]):5d}")
    tot = cnt.sum()
    print()
    print("总体 MAE: " + ", ".join(f"{n}={acc[:, k].sum() / tot:.3f}"
                                    for k, n in enumerate(NAMES)))
    print("长程段(Δ>=80) MAE: " + ", ".join(
        f"{n}={acc[3:, k].sum() / cnt[3:].sum():.3f}" for k, n in enumerate(NAMES)))

    print()
    print("=" * 72)
    print("Part 2: 压缩深度扫描(按几何分数分级, 预算恒 4 单位; 对照 Table 12)")
    print("=" * 72)
    for li, (name, ladder) in enumerate(LADDERS):
        b_tot = l_acc[:, li].sum() / max(l_n.sum(), 1)
        b_long = l_acc[3:, li].sum() / max(l_n[3:].sum(), 1)
        frames = sum(c for _, c in ladder)
        print(f"{name}  预算={budget_of(ladder):.2f} | 总体 MAE {b_tot:.3f} "
              f"| 长程段(Δ>=80) {b_long:.3f}")


if __name__ == "__main__":
    main()
