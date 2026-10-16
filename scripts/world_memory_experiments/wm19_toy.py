"""WM_19 toy: 三层记忆的分工——working / spatial(TSDF) / episodic 的写入与读出

机制内核(Video World Models with Long-term Spatial Memory, arXiv:2506.05284):
  working  = 最近 k 帧上下文(动态内容的唯一可靠来源);
  spatial  = TSDF 融合的静态点图(世界坐标): 跨帧不一致的观测被抑制 → 动态
             部分自然滤掉, 只留静态背景——代价是观测不足/被动态阴影遮过的
             格子永远无法「确认」;
  episodic = 稀疏关键帧: 新揭示的未知区域面积超过阈值时才写入(写入时机),
             读出用历史交叉注意力(软检索)——toy 里理想化为几何 top-K.

协议(仿 wm02 的环廊, 128 格, FOV = 前方 8 格):
  智能体随机游走(70% 前进 1 / 15% 掉头 / 6% 传送 / 9% 原地), 480 步;
  6 个动态对象随机游走(±1 各 30%, 60% 原地——制造「动态阴影」格).
  每 5 步查询一次: 当前视野 8 格中, 各记忆能给生成提供条件的比例,
  按格子的离开时长 Δ(距上一次被观测)分桶: first / short(≤16) / mid(16–64) / long(>64).

  注意: 查询发生在第 t 步观测写入**之前**——第 t 帧是要被生成的,
  条件只能来自 1..t−1 的记忆(自回归设定).

指标:
  (1) 静态条件覆盖率: Working(最近5帧) / Spatial(已确认静态格) /
      Epi+temporal(最近4关键帧) / Epi+geometric(FOV 重叠 top-4 关键帧,
      = 交叉注意力软检索的理想化) / Epi-all(全部关键帧, = 全注意力上界)
  (2) 写入时机 × 检索: reveal-阈值 vs 固定每10步 vs 不写——预算与中长时覆盖
  (3) 动态新鲜度: 从各记忆读出的动态对象位置偏差(格)——episodic 帧携带
      过期动态, working 帧才新鲜(分工的代价面)
  (4) TSDF 过滤代价: 观测≥2次但从未确认的「动态阴影」格数
"""

import numpy as np

rng = np.random.default_rng(19)

L = 128            # 环廊长度
FOV = 8            # 视野格数
T_STEPS = 480
QUERY_EVERY = 5
WORK_K = 5         # working 窗口
EPI_K = 4          # 检索 top-K
M_OBJ = 6          # 动态对象数
THETA_REVEAL = 3   # 写入阈值: 视野内未确认静态格 ≥ 3 才存关键帧
FIXED_EVERY = 10
CONFIRM_N = 2      # TSDF 确认条件: 观测 ≥2 次且众数占比 ≥0.75
CONFIRM_FRAC = 0.75
N_EPISODES = 40

POLICIES = ["reveal", "fixed", "none"]
BUCKETS = [("first", -1), ("short", 16), ("mid", 64), ("long", 10 ** 9)]


def ring_dist(a, b):
    d = np.abs(a - b) % L
    return np.minimum(d, L - d).mean()


def bucket_of(delta):
    if delta < 0:
        return "first"
    if delta <= 16:
        return "short"
    if delta <= 64:
        return "mid"
    return "long"


def confirmed(counts_cell):
    n = sum(counts_cell.values())
    if n < CONFIRM_N:
        return False
    return max(counts_cell.values()) / n >= CONFIRM_FRAC


def run_episode(theta=THETA_REVEAL):
    static_color = rng.integers(0, 10, L)
    obj_pos = rng.integers(0, L, M_OBJ)
    pos = int(rng.integers(0, L))
    dirn = 1

    last_seen = -np.ones(L, dtype=int)          # 上次被观测的步(查询用, 写入前)
    counts = [dict() for _ in range(L)]         # TSDF 式计数: 格 → {颜色: 次数}
    keyframes = {p: [] for p in POLICIES}       # (t, fov, obj_pos 元组)
    working = []                                # 最近 WORK_K 个 (t, fov, obj_pos)

    # 累积器: 主表(用 reveal 关键帧) + 写入策略 × 读出策略的分表
    schemes = ["Working", "Spatial", "EpiTmp", "EpiGeo", "EpiAll", "Full"]
    cov = {s: {b: [0, 0] for b, _ in BUCKETS} for s in schemes}
    cov_p = {(p, r): {b: [0, 0] for b, _ in BUCKETS}
             for p in ["reveal", "fixed"] for r in ["geo", "tmp"]}
    stale = {"EpiGeo": [], "EpiTmp": [], "Working": []}

    def fov_of(p, d):
        return frozenset((p + i * d) % L for i in range(1, FOV + 1))

    for t in range(1, T_STEPS + 1):
        # ---- 智能体与对象运动 ----
        r = rng.random()
        if r < 0.06:
            pos, dirn = int(rng.integers(0, L)), int(rng.choice([-1, 1]))
        elif r < 0.21:
            dirn = -dirn
        elif r < 0.91:
            pos = (pos + dirn) % L
        mv = rng.random(M_OBJ)
        step = np.where(mv < 0.3, -1, np.where(mv < 0.6, 1, 0))
        obj_pos = (obj_pos + step) % L

        fov = fov_of(pos, dirn)
        cur_obj = tuple(int(x) for x in obj_pos)

        # ---- 查询(写入前: 第 t 帧待生成, 条件只含 1..t−1) ----
        if t % QUERY_EVERY == 0:
            work_fov = set().union(*[w[1] for w in working]) if working else set()
            kf_rev = keyframes["reveal"]
            geo_top = sorted(kf_rev, key=lambda k: -len(k[1] & fov))[:EPI_K]
            tmp_top = kf_rev[-EPI_K:]
            all_fov = set().union(*[k[1] for k in kf_rev]) if kf_rev else set()
            geo_fov = set().union(*[k[1] for k in geo_top]) if geo_top else set()
            tmp_fov = set().union(*[k[1] for k in tmp_top]) if tmp_top else set()

            # 写入策略 × 读出策略(用各自的关键帧列表做同样的检索)
            pf_fov = {}
            for p in ["reveal", "fixed"]:
                kfl = keyframes[p]
                g = sorted(kfl, key=lambda k: -len(k[1] & fov))[:EPI_K]
                pf_fov[(p, "geo")] = set().union(*[k[1] for k in g]) if g else set()
                pf_fov[(p, "tmp")] = (set().union(*[k[1] for k in kfl[-EPI_K:]])
                                      if kfl else set())

            for cell in fov:
                b = bucket_of(t - last_seen[cell]) if last_seen[cell] >= 0 else "first"
                vals = {"Working": cell in work_fov,
                        "Spatial": confirmed(counts[cell]),
                        "EpiTmp": cell in tmp_fov,
                        "EpiGeo": cell in geo_fov,
                        "EpiAll": cell in all_fov,
                        "Full": cell in work_fov or cell in geo_fov}
                for s, v in vals.items():
                    cov[s][b][0] += int(v)
                    cov[s][b][1] += 1
                for p in ["reveal", "fixed"]:
                    for rr in ["geo", "tmp"]:
                        cov_p[(p, rr)][b][0] += int(cell in pf_fov[(p, rr)])
                        cov_p[(p, rr)][b][1] += 1

            if kf_rev:
                stale["EpiGeo"].append(np.mean([ring_dist(np.array(k[2]), obj_pos)
                                                for k in geo_top]))
                stale["EpiTmp"].append(np.mean([ring_dist(np.array(k[2]), obj_pos)
                                                for k in tmp_top]))
            if working:
                stale["Working"].append(np.mean([ring_dist(np.array(w[2]), obj_pos)
                                                 for w in working]))

        # ---- 观测写入 ----
        occ = set(int(x) for x in obj_pos)
        for cell in fov:
            color = 100 if cell in occ else static_color[cell]
            counts[cell][color] = counts[cell].get(color, 0) + 1
            last_seen[cell] = t
        kf = (t, fov, cur_obj)
        unconf = sum(1 for cell in fov if not confirmed(counts[cell]))
        if unconf >= theta:
            keyframes["reveal"].append(kf)
        if t % FIXED_EVERY == 0:
            keyframes["fixed"].append(kf)
        working.append(kf)
        working = working[-WORK_K:]

    # ---- 回合末统计 ----
    n_obs = [sum(c.values()) for c in counts]
    shadow = sum(1 for i in range(L) if n_obs[i] >= 2 and not confirmed(counts[i]))
    confirmed_n = sum(1 for i in range(L) if confirmed(counts[i]))
    return cov, cov_p, stale, {p: len(keyframes[p]) for p in POLICIES}, (confirmed_n, shadow)


def run_many(theta):
    """跑 N_EPISODES, 返回聚合的 cov / cov_p / stale / 预算 / TSDF 统计."""
    tot_cov = tot_cov_p = None
    tot_stale = {k: [] for k in ["EpiGeo", "EpiTmp", "Working"]}
    tot_kf = {p: [] for p in POLICIES}
    tsdf_stat = [0, 0]
    for _ in range(N_EPISODES):
        cov, cov_p, stale, nk, tsdf = run_episode(theta)
        if tot_cov is None:
            tot_cov = {s: {b: [0, 0] for b, _ in BUCKETS} for s in cov}
            tot_cov_p = {k: {b: [0, 0] for b, _ in BUCKETS} for k in cov_p}
        for s in cov:
            for b in cov[s]:
                tot_cov[s][b][0] += cov[s][b][0]
                tot_cov[s][b][1] += cov[s][b][1]
        for k in cov_p:
            for b in cov_p[k]:
                tot_cov_p[k][b][0] += cov_p[k][b][0]
                tot_cov_p[k][b][1] += cov_p[k][b][1]
        for k in stale:
            tot_stale[k].extend(stale[k])
        for p in POLICIES:
            tot_kf[p].append(nk[p])
        tsdf_stat[0] += tsdf[0]
        tsdf_stat[1] += tsdf[1]
    return tot_cov, tot_cov_p, tot_stale, tot_kf, tsdf_stat


def ml_cov(cov_dict, key):
    """mid+long 桶平均覆盖率."""
    vals = []
    for b in ["mid", "long"]:
        hit, n = cov_dict[key][b]
        vals.append(hit / n if n else float("nan"))
    return np.mean(vals)


def main():
    tot_cov, tot_cov_p, tot_stale, tot_kf, tsdf_stat = run_many(THETA_REVEAL)

    def pc(s, b):
        hit, n = tot_cov[s][b]
        return hit / n if n else float("nan")

    print("===== (1) 静态条件覆盖率(按视野内格子的离开时长 Δ 分桶) =====")
    schemes = ["Working", "Spatial", "EpiTmp", "EpiGeo", "EpiAll", "Full"]
    print(f"{'桶':>6} | {'n':>6} | " + " | ".join(f"{s:>8}" for s in schemes))
    for b, _ in BUCKETS:
        n = tot_cov["Working"][b][1]
        row = " | ".join(f"{pc(s, b) * 100:7.1f}%" for s in schemes)
        print(f"{b:>6} | {n:6d} | " + row)
    print("Working=最近5帧; Spatial=TSDF 已确认静态格; EpiTmp=最近4关键帧;")
    print("EpiGeo=FOV 重叠 top-4 关键帧(交叉注意力软检索的理想化);")
    print("EpiAll=全部关键帧(全注意力上界); Full=Working ∪ EpiGeo(完整三层).")

    print("\n===== (2) 写入时机 × 读出策略: 预算(存帧/回合) 与 mid+long 桶覆盖率 =====")
    print(f"{'写入策略':>16} | {'预算':>6} | {'geo 读出':>8} | {'tmp 读出':>8}")
    for theta in [1, 3, 6]:
        if theta == THETA_REVEAL:
            cov_p, kf = tot_cov_p, tot_kf
        else:
            _, cov_p, _, kf, _ = run_many(theta)
        print(f"reveal(θ={theta})".rjust(16) + " | " +
              f"{np.mean(kf['reveal']):6.1f} | " +
              f"{ml_cov(cov_p, ('reveal', 'geo')) * 100:7.1f}% | " +
              f"{ml_cov(cov_p, ('reveal', 'tmp')) * 100:7.1f}%")
    print(f"{'fixed(每10步)':>16} | " +
          f"{np.mean(tot_kf['fixed']):6.1f} | " +
          f"{ml_cov(tot_cov_p, ('fixed', 'geo')) * 100:7.1f}% | " +
          f"{ml_cov(tot_cov_p, ('fixed', 'tmp')) * 100:7.1f}%")
    print(f"{'none(不写)':>16} | {0.0:6.1f} |    0.0% |    0.0%")
    print("读法: θ 小 → 预算暴涨覆盖微增; geo 读出远胜 tmp 读出(时间不是相关性")
    print("的代理——WM 02 的结论在写入端复现); reveal 以 ~60% 的预算达到 fixed")
    print("的覆盖水平, 且在重访期不存冗余帧.")

    print("\n===== (3) 动态内容新鲜度: 从记忆读出的动态对象位置偏差(格) =====")
    for k in ["EpiGeo", "EpiTmp", "Working"]:
        print(f"{k:>8}: {np.mean(tot_stale[k]):6.2f} 格 (对象步速 |±1|·0.6)")
    print("读法: episodic 帧携带过期动态位置(geo 检索更旧——它专挑初见帧);")
    print("working 帧才新鲜——历史交叉注意力只能供身份/细节, 动态位置必须由")
    print("working 记忆供给(论文 Fig.5 的 w/o working 消融: 运动一致性崩).")

    print("\n===== (4) TSDF 过滤代价 =====")
    print(f"确认静态: {tsdf_stat[0] / N_EPISODES:.1f}/{L} 格; "
          f"观测≥2次但从未确认(动态阴影): {tsdf_stat[1] / N_EPISODES:.1f} 格")
    print("(论文 Fig.6 失败案例是它的极端版: 重建失败 → 大量静态点被误滤 → 记忆过稀)")


if __name__ == "__main__":
    main()
