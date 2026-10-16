"""WM_02 toy: 「最相关」vs「最近」—— 检索策略的覆盖率实验

VMem 的核心主张: 条件该用「与目标视图看得见同一片区域」的过去视图(内容投票),
而不是「最近的」(temporal) 或「相机位姿最近的」(camera distance)。
本 toy 把三种检索策略放进同一个部分可观测世界, 直接量「条件视图对当前查询
的覆盖率」——当前可见格中, 有多少被 top-K 条件视图的可见集合并集覆盖。

世界: 64 格环形走廊(与 wm01_toy 同一世界, 保持系列连续), 每格静态外观。
Agent: 位置 + 朝向(±1)。视图 = (pos, dir), 可见 = 朝向前方 W=8 格。
  随机游走 + 15% 概率转向 + 5% 概率随机传送 → 自然产生长短重访间隔、
  同位置反向重访、传送后异地重访三种检索难点。

四种检索(从历史视图库选 K=4 个作条件, 与论文高效版一致):
  Temporal        最近 K 个视图 —— 有限上下文窗口的默认做法
  CameraDistance  与当前位置环距最近的 K 个视图(忽略朝向) —— SEVA 检索的做法
  VMem           当前可见格的「历史观测者」按频率投票 top-K
                  —— surfel 索引投票的无遮挡 1D 版: 格 g 的 surfel 属性
                  I_g = 观测过 g 的视图索引集, 投票即对 I_g 计数
  OracleUpper     上帝视角: 从历史里贪心挑「对当前可见格新增覆盖最大」的 K 个,
                  给内容投票一个理论上限参照

分桶: 按当前可见格「距上次被观测的时长」中位数分桶:
  first(中位=从未见过) / short(≤8) / mid(8–64) / long(>64)。
覆盖率 = |当前可见格 ∩ 条件视图可见并集| / W。
查询节流: 每 10 步评估一次(同 wm01_toy), 统计口径不变。
"""
import numpy as np

rng = np.random.default_rng(11)

L, W, K = 128, 8, 4
T, EPISODES = 2000, 40
P_TURN, P_TP = 0.25, 0.06
QUERY_EVERY = 10

BUCKETS = ["first", "short", "mid", "long"]


def ring_visible(pos, d):
    return (pos + d * np.arange(W)) % L


def run_episode():
    pos, d = int(rng.integers(L)), 1
    hist_pos, hist_dir = [], []
    observers = [[] for _ in range(L)]     # 格 g -> 观测过它的视图索引(=surfel 的 I_g)
    last_seen = np.full(L, -1)
    acc = np.zeros((4, 4))                 # [bucket, strategy] 累积覆盖率
    cnt = np.zeros(4)

    for t in range(T):
        if rng.random() < P_TP:
            pos = int(rng.integers(L))
        elif rng.random() < P_TURN:
            d = -d
        else:
            pos = (pos + d * int(rng.integers(1, 4))) % L
        vis = ring_visible(pos, d)

        n_hist = len(hist_pos)
        if t % QUERY_EVERY == 0 and n_hist >= 16:
            vis_set = set(vis.tolist())
            # --- 分桶: 可见格的上次观测时刻 → 中位 Δ ---
            delta = np.where(last_seen[vis] >= 0, t - last_seen[vis], 10**9)
            med = float(np.median(delta))
            b = 0 if med > 10**8 else int(np.digitize(med, [8, 32])) + 1

            hp = np.array(hist_pos)
            hd = np.array(hist_dir)
            hist_vis = (hp[:, None] + hd[:, None] * np.arange(W)) % L   # (n_hist, W)

            # --- Temporal: 最近 K 个 ---
            i_t = np.arange(n_hist - K, n_hist)
            # --- CameraDistance: 环距最近 K 个(忽略朝向) ---
            dd = np.abs(hp - pos)
            dists = np.minimum(dd % L, (L - dd) % L)
            i_c = np.argpartition(dists, K)[:K]
            # --- VMem: 可见格的观测者频率投票 top-K ---
            votes = np.zeros(n_hist, dtype=np.int32)
            for g in vis:
                if observers[g]:
                    votes[np.array(observers[g])] += 1
            i_v = np.argsort(-votes)[:K]
            # --- OracleUpper: 对查询可见格贪心选新增覆盖最大的 K 个 ---
            vis_mask = np.zeros(L, dtype=bool)
            vis_mask[vis] = True
            covered = np.zeros(L, dtype=bool)
            i_o = []
            for _ in range(K):
                gains = (vis_mask & ~covered)[hist_vis].sum(axis=1)
                j = int(np.argmax(gains))
                if gains[j] == 0:
                    break
                i_o.append(j)
                covered[hist_vis[j]] = True

            # 覆盖率: 各策略选中视图的可见并集 ∩ 当前可见格
            for si, idx in enumerate([i_t, i_c, i_v, i_o]):
                u = np.zeros(L, dtype=bool)
                u[hist_vis[idx].ravel()] = True
                acc[b, si] += u[vis].sum() / W
            cnt[b] += 1

        # 写入记忆: 当前视图入历史, 更新 observers / last_seen
        hist_pos.append(pos)
        hist_dir.append(d)
        for g in vis:
            observers[g].append(len(hist_pos) - 1)
        last_seen[vis] = t
    return acc, cnt


def main():
    acc, cnt = np.zeros((4, 4)), np.zeros(4)
    for _ in range(EPISODES):
        a, c = run_episode()
        acc += a
        cnt += c
    names = ["Temporal", "CameraDist", "VMem", "OracleUp"]
    print(f"{'bucket':>8} | " + " | ".join(f"{n:>10}" for n in names) + " |    n")
    for b in range(4):
        if cnt[b] < 15:
            continue
        row = " | ".join(f"{acc[b, s] / cnt[b]:10.3f}" for s in range(4))
        print(f"{BUCKETS[b]:>8} | {row} | {int(cnt[b]):5d}")


if __name__ == "__main__":
    main()
