"""WM_13 toy: 权威共享状态 vs 逐视图历史 vs 稠密网格——载体决定身份存亡

对应 MASS (arXiv:2608.06257) 的两个受控对照:
  Panel A 载体对照 (论文 B-UN 消融): 同一个服务器、同一份转移噪声,
    typed records (槽位↔实体永久绑定) vs dense occupancy (只有格子集合,
    身份只能靠下游最近邻重关联) —— 预测器质量相同, 载体不同。
  Panel B 协议对照 (论文 B-PV 基线): 权威服务器 (全局状态推进一次,
    所有视图读同一份) vs 逐视图历史 (每个视图自己外推看不见的实体)。

世界: G×G 环面网格, N 个实体不重叠随机游走(撞格则停留)。
预测器(所有方案共享同一质量): 每实体每步 w.p. 1-P_ERR 猜对位移, 否则随机错向。
视图: K 个相机随机游走, 半径 R 圆盘内完美观测(对 PerView 已是宽容假设)。

指标:
  Panel A: 位置准确率(|est-true|≤1 格)、身份准确率(槽位仍跟对实体)
  Panel B: 跨视图分歧(两视图对同一实体的估计差 ≥1 格的比例)、各视图位置 MAE
  成本: 每 tick 转移执行次数 —— Typed/Dense 1 次, PerView K 次
"""

import numpy as np

rng = np.random.default_rng(13)

# ---- 参数 ----
G = 24            # 网格边长
N = 40            # 实体数
T = 300           # 每 episode 步数
K = 6             # 相机数 (Panel B)
R = 5             # 视域半径
P_ERR = 0.01      # 转移预测误差(每实体每步猜错概率) —— 所有方案相同
EPISODES = 20
QUERY_EVERY = 10
REPORT_TICKS = (50, 100, 200, 300)
FREEZE_LIMIT = 3  # Dense 槽位连续冻结局数上限, 超过判死

MOVES = np.array([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])


def torus_diff(a, b):
    """环面上的坐标差 a-b, 折到 [-G/2, G/2)"""
    d = (a - b) % G
    return np.where(d > G // 2, d - G, d)


def true_step(pos):
    """真值世界: 随机序随机游走, 目标格被占则停留。返回新位置与实际位移。"""
    occ = {tuple(p) for p in pos}
    new = pos.copy()
    for i in rng.permutation(N):
        m = MOVES[rng.integers(9)]
        tgt = tuple((pos[i] + m) % G)
        if tgt not in occ:
            occ.discard(tuple(new[i]))
            occ.add(tgt)
            new[i] = tgt
    return new, (new - pos) % G


def draw_move_hats(true_disp):
    """共享质量的预测器: w.p. 1-P_ERR 猜对实际位移, 否则随机错向。"""
    hats = np.empty((N, 2), dtype=int)
    for i in range(N):
        if rng.random() < 1 - P_ERR:
            hats[i] = true_disp[i]
        else:
            wrong = MOVES[(MOVES[:, 0] != true_disp[i, 0]) | (MOVES[:, 1] != true_disp[i, 1])]
            hats[i] = wrong[rng.integers(len(wrong))]
    return hats


def main():
    report_idx = {t: k for k, t in enumerate(REPORT_TICKS)}
    # 累计器
    # Panel A: [n_report, 2 方案(Typed, Dense)] x {pos_acc, id_acc}
    pa = {m: np.zeros((len(REPORT_TICKS), 2)) for m in ("pos_acc", "id_acc")}
    # Panel B: [n_report, 2 方案(Auth, PerView)] x {xview, mae}; mae 对 Auth 只算一份
    pb = {m: np.zeros((len(REPORT_TICKS), 2)) for m in ("xview", "mae")}
    cnt = np.zeros(len(REPORT_TICKS))

    for _ in range(EPISODES):
        # 初始化: N 个不重叠格子
        cells = rng.choice(G * G, N, replace=False)
        true_pos = np.stack([cells // G, cells % G], axis=1).astype(int)

        # Typed server (MASS): 槽位 i ↔ 实体 i 永久绑定
        typed_pos = true_pos.copy()
        # Dense server (B-UN): 内部仍按实体推进(预测器同等知情), 但发布的是格子集合;
        # 消费端用 N 个槽位做贪心最近邻重关联 —— 载体里没有身份。
        dense_ent = true_pos.copy()          # 每实体的内部推进位置(=Typed 同款噪声)
        slot_pos = true_pos.copy()           # 消费端槽位估计
        slot_follow = np.arange(N)           # 槽位当前跟随的实体链
        frozen = np.zeros(N, int)            # 连续冻结局数
        alive_slot = np.ones(N, bool)
        # PerView (B-PV): 每视图自己的全局估计
        pv_pos = np.tile(true_pos, (K, 1, 1))
        # 相机
        cam = np.stack([rng.integers(G, size=2) for _ in range(K)])

        for t in range(1, T + 1):
            true_pos, disp = true_step(true_pos)
            hats = draw_move_hats(disp)

            # --- Typed server: 一次转移, 所有视图读同一份 ---
            typed_pos = (typed_pos + hats) % G

            # --- Dense server: 同一份 hats 推进(公平), 发布去重格子集 ---
            dense_ent = (dense_ent + hats) % G
            published_cells, owner_ent = np.unique(dense_ent, axis=0, return_index=True)
            # 消费端贪心最近邻: 槽位×格子 按距离升序匹配
            if len(published_cells):
                dists = np.abs(torus_diff(slot_pos[:, None, :], published_cells[None, :, :])).sum(-1)
                pairs = sorted(
                    [(dists[si, ci], si, ci) for si in range(N) for ci in range(len(published_cells))]
                )
                used_s, used_c = set(), set()
                new_slot_pos = slot_pos.copy()
                for d, si, ci in pairs:
                    if si in used_s or ci in used_c or d > 4:
                        continue
                    used_s.add(si); used_c.add(ci)
                    new_slot_pos[si] = published_cells[ci]
                    slot_follow[si] = owner_ent[ci]   # 换链 = 可能的身份交换
                    frozen[si] = 0
                unmatched = [si for si in range(N) if si not in used_s]
                for si in unmatched:
                    frozen[si] += 1
                    if frozen[si] > FREEZE_LIMIT:
                        alive_slot[si] = False
                slot_pos = new_slot_pos

            # --- PerView: 视野内置为真值, 视野外各自外推 ---
            for k in range(K):
                m = MOVES[rng.integers(9)]
                cam[k] = (cam[k] + m) % G
                d = torus_diff(true_pos, cam[k])
                inview = (np.abs(d) <= R).all(1)
                hats_k = draw_move_hats(disp)      # 每视图独立噪声
                pv_pos[k] = np.where(inview[:, None], true_pos, (pv_pos[k] + hats_k) % G)

            # --- 查询 ---
            if t % QUERY_EVERY == 0 and t in report_idx:
                ri = report_idx[t]
                cnt[ri] += 1

                # Panel A: Typed
                ok_t = (np.abs(torus_diff(typed_pos, true_pos)) <= 1).all(1)
                pa["pos_acc"][ri, 0] += ok_t.mean()
                pa["id_acc"][ri, 0] += 1.0                      # 槽位绑定, 构造保证
                # Panel A: Dense (只统计活槽位; 死槽位记身份错误)
                ok_d = (np.abs(torus_diff(slot_pos, true_pos)) <= 1).all(1)
                pa["pos_acc"][ri, 1] += (ok_d & alive_slot).mean()
                pa["id_acc"][ri, 1] += ((slot_follow == np.arange(N)) & alive_slot).mean()

                # Panel B: Auth = Typed (所有视图读同一份 → 分歧恒 0)
                pb["xview"][ri, 0] += 0.0
                pb["mae"][ri, 0] += np.abs(torus_diff(typed_pos, true_pos)).sum(1).mean()
                # Panel B: PerView
                dis = 0.0
                for a in range(K):
                    for b in range(a + 1, K):
                        bad = (np.abs(torus_diff(pv_pos[a], pv_pos[b])) >= 1).any(1)
                        dis += bad.mean()
                pb["xview"][ri, 1] += dis / (K * (K - 1) / 2)
                mae_k = np.abs(torus_diff(pv_pos, true_pos)).sum(-1).mean()
                pb["mae"][ri, 1] += mae_k

    names_a = ["Typed (MASS)", "Dense (B-UN)"]
    names_b = ["Auth shared (MASS)", "PerView (B-PV)"]
    print(f"参数: G={G}, N={N}, T={T}, K={K}, R={R}, P_ERR={P_ERR}, episodes={EPISODES}")
    print(f"每 tick 转移执行次数: Typed/Dense = 1, PerView = {K} (构造事实, 非测量)\n")

    print("===== Panel A 载体对照: 同一服务器, 同一份转移噪声 =====")
    print(f"{'tick':>6} | {'位置准确率':>22} | {'身份准确率':>22}")
    for k, t in enumerate(REPORT_TICKS):
        row_p = " | ".join(f"{pa['pos_acc'][k, j] / cnt[k]:10.3f}" for j in range(2))
        row_i = " | ".join(f"{pa['id_acc'][k, j] / cnt[k]:10.3f}" for j in range(2))
        print(f"{t:>6} | {row_p} | {row_i}")
    print(f"{'':>6} | {names_a[0]:>10} vs {names_a[1]:>9} | {names_a[0]:>10} vs {names_a[1]:>9}")

    print("\n===== Panel B 协议对照: 权威共享 vs 逐视图历史 =====")
    print(f"{'tick':>6} | {'跨视图分歧':>22} | {'位置 MAE (格)':>22}")
    for k, t in enumerate(REPORT_TICKS):
        row_x = " | ".join(f"{pb['xview'][k, j] / cnt[k]:10.3f}" for j in range(2))
        row_m = " | ".join(f"{pb['mae'][k, j] / cnt[k]:10.3f}" for j in range(2))
        print(f"{t:>6} | {row_x} | {row_m}")
    print(f"{'':>6} | {names_b[0]:>10} vs {names_b[1]:>9} | {names_b[0]:>10} vs {names_b[1]:>9}")


if __name__ == "__main__":
    main()
