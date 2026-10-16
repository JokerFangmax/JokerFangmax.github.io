"""WM_24 toy: 遮挡持久性——surfel 打分检索 vs 滑窗 / 固定步长 (Mem-World, arXiv:2606.18960)

对应 W-VMem 的核心机制: 在「腕部相机窄 FOV + 末端频繁遮挡 + 快速腕动」的抓取流程里,
检索哪些历史帧作为生成条件。打分 (论文 Eq. 2) = 几何可见性 × 任务相关性 × 时间衰减,
并对候选帧做 NMS 避免冗余。

世界: 100×100 工作区, 14 个静态物体 + 4 个被操纵物体 (抓取瞬间状态改变 ±4~10)。
末端执行器带腕部相机: FOV 半角 35°、距离 55; 距夹爪 14 以内的物体被遮挡。
轨迹为脚本化抓取流程 (2 轮 × 4 物体): 接近 (22 步, 相机朝向目标) → 驻留 (10 步,
目标被夹爪遮挡, 第 5 步状态改变) → 后撤 (16 步, 前 8 步相机仍回看目标 → 产生
改变后的干净观测) → 游走 (20 步)。观测噪声 σ(d) = 0.4 + 0.06d, 越近越清晰。

查询: 每步给定未来动作 chunk 的平均腕部位姿 (动作已知, 对应论文的 FK+固定变换),
只评估未来视野内的物体 (世界模型必须预测的部分)。四种检索策略从全部历史中取 K=4 帧:
  ShortTerm  最近 4 帧 (Interactive World Simulator 式短时上下文)
  Stride     等距铺满全历史的 4 帧 (Ctrl-World / EVAC-AC 式固定步长)
  WVMem      论文机制: 逐帧打分 (可见性×任务 flag×时间衰减) + NMS
  WVMem-noD  去掉时间衰减项的消融 (动态物体应检索到「看得最清楚」的改变前旧帧)

证据读取理想化 (隔离「检索质量」这一个变量): 物体在检索帧中被干净观测到即得证据,
误差 = min over 候选帧 (0.8σ(d) + 改变前证据的过期罚 |δ|), 无任何候选 → 先验误差
E|N(0,10)| ≈ 7.98。按 Δ = 距该物体上次被干净观测的时长分桶。
"""

import numpy as np

rng = np.random.default_rng(24)

# ---- 参数 ----
L = 100.0
N_STAT, N_DYN = 14, 4
M_OBJ = N_STAT + N_DYN
K = 4
FOV_HALF = np.deg2rad(22)
FOV_RANGE = 45.0
OCCL_R = 14.0
SIG0, SIGD = 0.4, 0.06          # 观测噪声 σ(d) = SIG0 + SIGD·d
PRIOR_SIG = 10.0
PRIOR_ERR = PRIOR_SIG * np.sqrt(2 / np.pi)   # ≈ 7.98
LAM_MIN, H_FRAC = 0.1, 0.3      # 论文 Eq. 2 的衰减参数 (λmin=0.1, H=0.3·T)
FLAG_DYN = 3.0                  # 任务相关性 flag: 被操纵物体 ×3 (ln(e+m) 的简化)
EPISODES = 12
GAP_BINS = [0, 4, 8, 16, 32, 64, 128, 10 ** 9]


def build_traj(dyn_pos):
    """脚本化抓取轨迹 → (E: T×2 末端位置, H: T 相机朝向, change: N_DYN 改变时刻, delta)"""
    pos, head = [], []
    change = np.full(N_DYN, -1)
    delta = np.zeros(N_DYN)
    cur = rng.uniform(15, 85, 2)
    for rnd in range(2):
        for j in rng.permutation(N_DYN):
            o = dyn_pos[j]
            side = rng.normal(0, 1, 2)
            side /= np.linalg.norm(side)
            dwell = np.clip(o + 10.0 * side, 8, 92)     # 距目标 10 → 驻留时被遮挡
            # 接近 22 步, 相机朝向目标
            for s in range(22):
                p = cur + (dwell - cur) * s / 22
                pos.append(p)
                head.append(np.arctan2(o[1] - p[1], o[0] - p[0]))
            cur = dwell
            # 驻留 10 步 (目标被夹爪遮挡), 第 5 步状态改变
            base = len(pos)
            for s in range(10):
                pos.append(cur)
                head.append(np.arctan2(o[1] - cur[1], o[0] - cur[0]))
                if s == 5:
                    change[j] = base + s if change[j] < 0 else change[j]
                    delta[j] = rng.uniform(4, 10) * (1 if rng.random() < 0.5 else -1)
            # 后撤 16 步: 第 4~10 步相机回看目标 (产生改变后的较远观测), 其余朝运动方向
            rdir = rng.normal(0, 1, 2)
            rdir /= np.linalg.norm(rdir)
            while rdir @ (o - cur) > 0:                 # 不朝目标方向撤
                rdir = rng.normal(0, 1, 2)
                rdir /= np.linalg.norm(rdir)
            for s in range(16):
                cur = np.clip(cur + 3.0 * rdir, 6, 94)
                pos.append(cur)
                if 4 <= s < 10:
                    head.append(np.arctan2(o[1] - cur[1], o[0] - cur[0]))
                else:
                    head.append(np.arctan2(rdir[1], rdir[0]))
            # 游走 20 步, 朝向 = 运动方向
            wp = rng.uniform(10, 90, 2)
            for s in range(20):
                p = cur + (wp - cur) * s / 20
                pos.append(p)
                d = wp - p
                head.append(np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 1e-9
                            else head[-1])
            cur = wp
    return np.array(pos), np.array(head), change, delta


def visibility(E, H, O):
    """V[t,i]: 物体 i 在 t 步被腕部相机干净观测 (在 FOV 内且未被夹爪遮挡); d/σ 同步返回"""
    diff = O[None, :, :] - E[:, None, :]                # T×M×2
    d = np.linalg.norm(diff, axis=2)                    # T×M
    ang = np.abs(np.arctan2(diff[:, :, 1], diff[:, :, 0]) - H[:, None])
    ang = np.minimum(ang, 2 * np.pi - ang)
    V = (d <= FOV_RANGE) & (ang <= FOV_HALF) & (d > OCCL_R)
    sig = SIG0 + SIGD * d
    return V, d, sig


def run_episode():
    # ---- 场景 ----
    obj = rng.uniform(12, 88, (M_OBJ, 2))
    for _ in range(200):                                # 保持物体间最小间距
        dd = np.linalg.norm(obj[:, None] - obj[None, :], axis=2)
        np.fill_diagonal(dd, 99)
        i, j = np.unravel_index(np.argmin(dd), dd.shape)
        if dd[i, j] > 18:
            break
        obj[j] = rng.uniform(12, 88, 2)
    dyn_idx = np.arange(N_STAT, M_OBJ)

    E, H, change, delta = build_traj(obj[dyn_idx])
    T = len(E)
    V, dmat, sig = visibility(E, H, obj)
    H_HALF = H_FRAC * T

    flag = np.ones(M_OBJ)
    flag[dyn_idx] = FLAG_DYN
    Vq = V / (1.0 + dmat / 10.0)                        # 几何质量项 1/(1+d/10)
    base = Vq * flag[None, :]                           # T×M (不含衰减)
    cover = V.copy()                                     # 帧的覆盖集 (可见即覆盖)

    is_dyn = np.zeros(M_OBJ, bool)
    is_dyn[dyn_idx] = True

    acc = np.zeros((len(GAP_BINS) - 1, 4))
    cnt = np.zeros(len(GAP_BINS) - 1)
    stale_hit = np.zeros(4)
    stale_tot = 0
    miss = np.zeros(4)
    miss_tot = 0

    last_clean = np.full(M_OBJ, -1)
    for t in range(15, T - 5):
        last_clean = np.where(V[t], t, last_clean)
        # 未来平均腕部位姿 (动作已知) 与朝向
        ef = E[t + 1:t + 5].mean(0)
        de = ef - E[t]
        hf = np.arctan2(de[1], de[0]) if np.linalg.norm(de) > 1e-9 else H[t]
        dif = obj - ef[None, :]
        df = np.linalg.norm(dif, axis=1)
        angf = np.abs(np.arctan2(dif[:, 1], dif[:, 0]) - hf)
        angf = np.minimum(angf, 2 * np.pi - angf)
        fview = (df <= FOV_RANGE) & (angf <= FOV_HALF) & (df > OCCL_R)
        idxs = np.where(fview & (last_clean >= 0))[0]
        if len(idxs) == 0:
            continue

        # ---- 四种检索策略的候选帧 ----
        cands = [
            np.arange(t - 3, t + 1),                                      # ShortTerm
            np.unique(np.linspace(0, t, K).astype(int)),                  # Stride
            None,                                                         # WVMem
            None,                                                         # WVMem-noDecay
        ]
        # WVMem 打分 (含/不含衰减)
        sc = base[:t + 1] @ (fview.astype(float))
        dt = t - np.arange(t + 1)
        decay = LAM_MIN + (1 - LAM_MIN) * 2 ** (-dt / H_HALF)
        sc_dec = sc * decay
        cov = cover[:t + 1][:, fview]
        cands[2] = nms_select(sc_dec, cov, K)
        cands[3] = nms_select(sc, cov, K)

        # ---- 证据评估: 每个策略独立取其候选帧中的最优证据 ----
        best_k, arg_k = [], []
        for k in range(4):
            best = np.full(M_OBJ, PRIOR_ERR)
            arg = np.full(M_OBJ, -1)
            for u in cands[k]:
                for i in idxs:
                    if not V[u, i]:
                        continue
                    st = 0.0
                    if is_dyn[i]:
                        dj = np.where(dyn_idx == i)[0][0]
                        if u < change[dj]:
                            st = abs(delta[dj])
                    e = 0.8 * sig[u, i] + st
                    if e < best[i]:
                        best[i] = e
                        arg[i] = u
            best_k.append(best)
            arg_k.append(arg)

        # 分桶记账: 计数与策略无关, 只记一次 (勿放回 k 循环——见 wm01 脚本教训)
        gap = t - last_clean[idxs]
        b = np.digitize(gap, GAP_BINS) - 1
        for bi, i in enumerate(idxs):
            for k in range(4):
                acc[b[bi], k] += best_k[k][i]
            cnt[b[bi]] += 1
            if is_dyn[i]:
                dj = np.where(dyn_idx == i)[0][0]
                stale_tot += 1
                miss_tot += 1
                for k in range(4):
                    if arg_k[k][i] < 0:
                        miss[k] += 1
                    elif arg_k[k][i] < change[dj]:
                        stale_hit[k] += 1
    return acc, cnt, stale_hit, stale_tot, miss, miss_tot


def nms_select(score, cov, k):
    """按分数贪心选帧, 覆盖集 Jaccard > 0.5 视为冗余跳过"""
    order = np.argsort(-score)
    sel = []
    for u in order:
        if score[u] <= 0:
            break
        if any(_jac(cov[u], cov[v]) > 0.5 for v in sel):
            continue
        sel.append(u)
        if len(sel) == k:
            break
    if not sel:
        sel = [int(order[0])]
    return np.array(sel)


def _jac(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def main():
    acc = np.zeros((len(GAP_BINS) - 1, 4))
    cnt = np.zeros(len(GAP_BINS) - 1)
    sh = np.zeros(4); st_tot = 0
    ms = np.zeros(4); ms_tot = 0
    for _ in range(EPISODES):
        a, c, s, st, m, mt = run_episode()
        acc += a; cnt += c
        sh += s; st_tot += st
        ms += m; ms_tot += mt

    names = ["ShortTerm", "Stride", "WVMem", "WVMem-noDecay"]
    print("===== 物体状态估计误差 MAE vs Δ (距上次干净观测的时长, 先验误差 7.98) =====")
    print(f"{'Δ bin':>12} | " + " | ".join(f"{n:>13}" for n in names) + " |     n")
    for bi in range(len(GAP_BINS) - 1):
        if cnt[bi] < 50:
            continue
        lo, hi = GAP_BINS[bi], min(GAP_BINS[bi + 1], 999)
        row = " | ".join(f"{acc[bi, k] / cnt[bi]:13.3f}" for k in range(4))
        print(f"{f'{lo}-{hi}':>12} | {row} | {int(cnt[bi]):6d}")

    tot = acc.sum(0) / max(cnt.sum(), 1)
    print("\n===== 总体 MAE =====")
    print({n: round(float(x), 3) for n, x in zip(names, tot)})

    print("\n===== 被操纵物体: 检索到「改变前旧帧」的比例 (过期证据率) / 无任何证据率 =====")
    for k, n in enumerate(names):
        print(f"  {n:>13}: stale {sh[k]/max(st_tot,1):.3f}  miss {ms[k]/max(ms_tot,1):.3f}")
    print(f"  (被操纵物体查询共 {st_tot} 次)")


if __name__ == "__main__":
    main()
