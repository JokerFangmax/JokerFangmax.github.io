"""WM_08 toy: 归因链与事件因子——WRBench 两个核心结构的机制版

WRBench (arXiv:2606.20545) 的两个核心经验结构:
  (1) Figure 4 相关结构 + Finding 1: 可见质量 / 重访通道 (support) / 重访后正确性
      是三个松耦合的块——support 与可见分负相关 (r=-0.42),
      "画质最干净的系统最不被考验" (Hailuo 可见 0.891 但 support 仅 6.3%);
  (2) Table 8 事件因子 2×2 + Finding 2: 空间位移 (relocation) 给模型一个新坐标
      + 静态场景可锚定, 原地形变 (in-place change) 无锚点 → 漂移/涂抹,
      状态因子把可见位置分拉低最多 (-0.114)。

Part A (归因链): 23 个模拟模型分属 4 种控制范式 (source-video / geometry-cache /
  model-inferred / prompt-only), 各有 (相机胆量 b, 潜在质量 Q):
    b 决定目标是否离场又回场 (support = P(离场∧回场∧可判));
    Q 共同驱动渲染质量 g 与重访正确性 κ (跨块正相关);
    范式偏移: prompt-only 画质高但相机胆小 → visible 与 support 负相关。
  重入会带来可见连贯性惩罚 (re-entry artifact): 出过场的片段可见分 -0.10。
  输出: 模型级相关矩阵 (对照论文 Figure 4) + 天真平均榜 vs 门控档案的排名反转。

Part B (事件因子): 事件 = (是否位移, 是否形变) 的 2×2。生成器两条通道:
    位置通道 (where-memory, 几何/运动先验, 训练充分):
      位移 → 运动先验残差小 (新坐标 + 静态场景锚定);
      原地形变 → 无锚点, 位置漂移 (drift/smear);
    状态通道 (what-memory, 遮挡下形变的监督稀缺):
      形变的写回率 q̂ ~ Beta(2,4) (每 episode 抽, 有先验 fired/没 fired 之分),
      位移事件的运动完成先验更好 (跳→落在数据里多, q̂ 上移);
      无形变 → 原样写回 (但纯位移事件重入后重渲染更干净, 轻微 bonus)。
  另以 7% 概率发生「目标丢失」(遮挡期间目标消解, HyDRA 的 sit 案例)。
  输出: 2×2 四格的 (可见位置, 可见状态, 重访位置, 重访状态) 四列得分,
  对照论文 Table 8 的因子对照方向; 以及三种失败模式 (擦除/错态/丢失) 的占比。
"""

import numpy as np

rng = np.random.default_rng(23)


# ---------- Part A: 归因链 ----------

def part_a():
    print("=" * 76)
    print("Part A: 归因链——support / 可见质量 / 重访正确性的相关结构与排名反转")
    print("=" * 76)

    N_EV = 400
    # 范式 → (模型数, 相机胆量 b 的范围, 画质范式偏移)
    paradigms = [
        ("source-video",   3, (0.40, 0.62), -0.02),
        ("geometry-cache", 3, (0.30, 0.72), -0.04),
        ("model-inferred", 9, (0.06, 0.40), +0.00),
        ("prompt-only",    8, (0.01, 0.13), +0.05),
    ]

    models = []
    for pname, n, (blo, bhi), off in paradigms:
        for _ in range(n):
            Q = rng.uniform(0, 1)
            b = rng.uniform(blo, bhi)
            g = np.clip(0.78 + 0.10 * Q + off, 0.6, 0.95)   # 渲染质量
            kappa_s = 0.55 + 0.20 * Q                        # 重访位置正确性
            kappa_t = 0.50 + 0.22 * Q                        # 重访状态正确性
            models.append(dict(paradigm=pname, b=b, g=g, ks=kappa_s, kt=kappa_t))

    rows = []
    for m in models:
        exits = rng.random(N_EV) < m["b"]
        # 可见分: 出场→重入伪影惩罚
        vis_sp = np.clip(m["g"] - 0.10 * exits + rng.normal(0, 0.04, N_EV), 0, 1)
        vis_st = np.clip(m["g"] - 0.08 * exits + rng.normal(0, 0.05, N_EV), 0, 1)
        re_sp = np.clip(m["ks"] + rng.normal(0, 0.06, N_EV), 0, 1)
        re_st = np.clip(m["kt"] + rng.normal(0, 0.06, N_EV), 0, 1)
        rows.append(dict(
            paradigm=m["paradigm"],
            vis_sp=vis_sp.mean(), vis_st=vis_st.mean(),
            support=exits.mean(),
            re_sp=re_sp[exits].mean() if exits.any() else np.nan,
            re_st=re_st[exits].mean() if exits.any() else np.nan,
            n_re=int(exits.sum()),
        ))

    # --- 相关结构 (对照论文 Figure 4) ---
    def corr(x, y):
        return np.corrcoef(x, y)[0, 1]

    vis_sp = np.array([r["vis_sp"] for r in rows])
    vis_st = np.array([r["vis_st"] for r in rows])
    sup = np.array([r["support"] for r in rows])
    ok = ~np.isnan([r["re_st"] for r in rows])
    re_sp = np.array([r["re_sp"] for r in rows])[ok]
    re_st = np.array([r["re_st"] for r in rows])[ok]

    print("\n模型级相关 (toy vs 论文 Figure 4):")
    pairs = [
        ("可见位置 ↔ 可见状态", vis_sp, vis_st, "0.97"),
        ("重访位置 ↔ 重访状态", re_sp, re_st, "0.94"),
        ("可见位置 ↔ 重访状态", vis_sp[ok], re_st, "0.60–0.79"),
        ("可见位置 ↔ support", vis_sp, sup, "-0.42"),
        ("support ↔ 重访状态", sup[ok], re_st, "-0.15"),
    ]
    for name, x, y, ref in pairs:
        print(f"  {name:<22} toy r = {corr(x, y):+.2f}   (论文 {ref})")

    # --- 天真平均榜 vs 门控档案 ---
    print("\n天真平均榜 (四分直接平均, 缺失列跳过——天真基准的自然做法):")
    ranked = sorted(rows, key=lambda r: -np.mean([r["vis_sp"], r["vis_st"],
                                                  r["re_sp"], r["re_st"]]))
    print(f"{'#':>2} {'范式':<15} {'可见位':>6} {'可见态':>6} {'support':>8} "
          f"{'重访态':>6} {'n重访':>5}")
    for i, r in enumerate(ranked[:3]):
        print(f"{i+1:>2} {r['paradigm']:<15} {r['vis_sp']:>6.3f} {r['vis_st']:>6.3f} "
              f"{r['support']:>8.1%} {r['re_st']:>6.3f} {r['n_re']:>5}")
    top = ranked[0]
    se = 0.06 / np.sqrt(max(top["n_re"], 1))   # 重访分的标准误 (近似)
    print(f"\n读法: 榜首 support 仅 {top['support']:.1%}, 重访分只由 {top['n_re']} 个样本"
          f"平均 (标准误 ≈ ±{se:.2f})——")
    print("  天真榜单把「没被考到」当成了「考得不错」。")
    print("  WRBench 的做法是把 support 单独报告, 重访分只在可判子集上条件化。")


# ---------- Part B: 事件因子 2×2 ----------

def part_b():
    print("\n" + "=" * 76)
    print("Part B: 事件因子 2×2——锚点机制与 what-memory 写回失败")
    print("=" * 76)

    N_EV = 400
    # ---- 通道参数 (一次标定, 不逐格拟合) ----
    P_LOSS = 0.07        # 目标丢失概率 (遮挡期间消解, HyDRA 的 sit 案例)
    POS_SIG_STATIC = 0.22  # 无形变: 静态锚 + 运动先验 → 位置残差
    POS_SIG_CHANGE = 0.38  # 原地形变: 无锚点 → 漂移/涂抹
    POS_MOVE_BONUS = 0.02  # 位移: 新坐标 + 静态场景可锚定 (对可见位影响很小)
    REENTRY_SIG = 0.15     # 重入重渲染噪声 (位置)
    ST_VIS_SIG = 0.55      # 可见期状态基线误差 (VLM 判定严格, 状态列整体偏低)
    ST_VIS_LEGIB = 0.10    # 位移事件结果更可判 (跳→落 vs 折→?)
    ST_VIS_CHANGE = 0.02   # 形变轻微拉低可见态
    ST_RE_SIG = 0.45       # 重访态基线误差
    ST_WRITE = 0.50        # 写回残差系数: (1-q̂)·c·ST_WRITE
    ST_MOVE_PRIOR = 0.08   # 位移事件的运动完成先验 (跳→落 在数据里多)
    QHAT_MOVE = 0.20       # q̂ 加成 (同上, 作用于写回率本身)

    def run_grid(move, change):
        c = rng.uniform(0.3, 0.6, N_EV)     # 形变幅度
        # ---- 位置通道 (where-memory: 几何/运动先验, 训练充分) ----
        sigma_pos = POS_SIG_CHANGE if change else POS_SIG_STATIC
        pos_err = np.abs(rng.normal(0, sigma_pos, N_EV)) \
            - (POS_MOVE_BONUS if move else 0.0)
        pos_err = np.clip(pos_err, 0, None)
        lost = rng.random(N_EV) < P_LOSS
        pos_err_reobs = np.where(lost, 1.0,
                                 pos_err + np.abs(rng.normal(0, REENTRY_SIG, N_EV)))
        # ---- 状态通道 (what-memory: 遮挡下形变的监督稀缺) ----
        # 写回率 q̂ ~ Beta(2,4) 每 episode 异质 (先验 fired / 没 fired 之分)
        qhat = np.clip(rng.beta(2, 4, N_EV) + (QHAT_MOVE if move else 0.0), 0, 1)
        pred_frac = qhat + rng.normal(0, 0.05, N_EV) / c   # 预测形变 / 真实形变
        # 可见期: 变化看得见 (可见演化不是瓶颈), 位移事件结果更可判
        state_err_vis = np.abs(rng.normal(0, ST_VIS_SIG, N_EV)) \
            + (ST_VIS_CHANGE if change else 0.0) - (ST_VIS_LEGIB if move else 0.0)
        state_err_vis = np.clip(state_err_vis, 0, None)
        # 重访期: 写回残差 + (位移时) 运动完成先验——位移事件 (跳→落) 的结果
        # 状态在数据里覆盖好, 无论有没有形变都带这个先验
        state_err_reobs = np.abs(rng.normal(0, ST_RE_SIG, N_EV)) \
            + ((1.0 - qhat) * c * ST_WRITE if change else 0.0) \
            - (ST_MOVE_PRIOR if move else 0.0)
        state_err_reobs = np.clip(state_err_reobs, 0, None)
        state_err_reobs = np.where(lost, 1.0, state_err_reobs)
        # 失败模式分类 (只对形变事件): 擦除 ≈ 原样写回 / 错态 = 部分写回 / 丢失
        modes = ""
        if change:
            erasure = ((pred_frac < 0.25) & ~lost).mean()
            wrong = ((0.25 <= pred_frac) & (pred_frac < 0.75) & ~lost).mean()
            modes = (f"   失败模式: 擦除 {erasure:.0%} / 错态 {wrong:.0%} "
                     f"/ 丢失 {lost.mean():.0%}")
        return (np.clip(1 - pos_err, 0, 1).mean(),
                np.clip(1 - state_err_vis, 0, 1).mean(),
                np.clip(1 - pos_err_reobs, 0, 1).mean(),
                np.clip(1 - state_err_reobs, 0, 1).mean(),
                modes)

    print(f"\n{'事件类型':<10} {'可见位':>6} {'可见态':>6} {'重访位':>6} {'重访态':>6}"
          "   (论文 Table 8)")
    paper = {
        "none":       "0.820/0.580/0.678/0.618",
        "spatial":    "0.825/0.644/0.705/0.668",
        "state-only": "0.697/0.541/0.601/0.547",
        "full":       "0.711/0.598/0.641/0.603",
    }
    grid = {}
    for name, move, change in [("none", False, False), ("spatial", True, False),
                               ("state-only", False, True), ("full", True, True)]:
        vp, vs, rp, rs, modes = run_grid(move, change)
        grid[name] = (vp, vs, rp, rs)
        print(f"{name:<10} {vp:>6.3f} {vs:>6.3f} {rp:>6.3f} {rs:>6.3f}"
              f"   论文: {paper[name]}{modes}")

    print("\n因子对照 (该格 − none; 论文: 空间因子 可见位 +0.008 n.s. / 可见态 +0.070 /"
          " 重访态 +0.038;")
    print("  状态因子 可见位 −0.114*** / 可见态 −0.031 n.s. / 重访位 −0.075** /"
          " 重访态 −0.068***):")
    for a, label in [("spatial", "空间因子 (spatial − none)"),
                     ("state-only", "状态因子 (state-only − none)"),
                     ("full", "双因子 (full − none)")]:
        d = np.array(grid[a]) - np.array(grid["none"])
        print(f"  {label:<26} 可见位 {d[0]:+.3f}  可见态 {d[1]:+.3f}  "
              f"重访位 {d[2]:+.3f}  重访态 {d[3]:+.3f}")
    print("\n机制读法: 原地形变无锚点 → 位置通道也跟着掉 (被改变的物体在原地漂移/涂抹);")
    print("  状态写回靠稀缺先验 → 重访态掉得最深; 位移给位置通道一个新坐标 + 静态锚,")
    print("  运动完成先验还顺带拉高状态列——「最难的一格」= 原地发生的形变 (state-only)。")


if __name__ == "__main__":
    part_a()
    part_b()
