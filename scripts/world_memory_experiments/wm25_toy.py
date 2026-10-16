"""WM_25 toy: 近期帧 + 事件边界锚帧 + gist token 的混合持久记忆——MemoryWAM 最小机制版

设定（把混合记忆的三层级缩成等预算的存储-检索博弈）:
  一条长为 N 帧的轨迹, 决策时需要回答三类查询（对应 MemoryWAM 的动机）:
    onset  开局信息(壳里哪颗球/桌上写了哪些数字): 出现在头 N_anchor 帧, 精粒度
           (容差 tol_fine)——「初始场景锚定指令关键信息, 之后会被遮挡/滑出窗口」;
    mid    中程语义(进度/计数/已发生事件): 年龄均匀分布在 (8, N-8), 粗粒度
           (容差 tol_coarse)——长期记忆「保留 gist 痕迹而非逐字细节」;
    recent 即时控制(接触状态/手-物构型): 年龄 < 4, 精粒度——短期记忆。
  存储层级决定检索精度: 全 token 帧 σ_full; gist token σ_gist ≫ σ_full;
  被逐出 = 只剩先验猜测(真值 ~ U(0,100), 猜中心, 误差期望 25)。
  每集查询构成: 1 onset(fine) + 8 mid(coarse) + 2 recent(fine)。

策略（① 的四种, 等预算比较: FIFO-eq 的窗口取到与 Hybrid 相同的 token 预算）:
  FIFO-4    有界窗口 WAM(FastWAM/π0.5 式): 只留最近 4 帧全 token, 成本恒定
  Full      全历史 KV(LingBot-VA 式): 所有帧全 token, 成本 O(N·L); 检索附加
            干扰噪声 σ_int·N——论文「稠密冗余上下文妨碍检索」论断的模型化(假设项,
            置 0 则 Full 为精度上限)
  Hybrid    MemoryWAM: 2 锚帧(开局, 全 token) + 4 近期帧(全 token) + 其余每帧
            1 个 gist token, 长期成本 O(N/d), d = L/M = 15
  FIFO-eq   等预算对照: 把 Hybrid 的全部预算换成「更长的 FIFO 窗口」——检验
            「把钱摊到 gist + 锚帧」是否优于「把钱堆到更长的窗口」

输出: ①三类查询精度 vs 轨迹长度 N + 上下文 token 成本 vs N;
     ②N=400 组件消融(对标论文 Table 3); ③N=800 精粒度查询 MAE vs 年龄。
"""

import numpy as np

rng = np.random.default_rng(25)

# ---- 参数 ----
L_FULL, M_GIST = 15.0, 1.0      # 每帧全 token 数 / 每帧 gist token 数 (d = 15)
N_ANCHOR, N_RECENT = 2, 4       # 锚帧数 / 近期窗口(MemoryWAM 实现值)
SIG_FULL, SIG_GIST = 0.8, 6.0   # 全 token / gist 的检索噪声
SIG_INT = 0.001                 # Full 的检索干扰噪声(每帧, 假设项)
TOL_FINE, TOL_COARSE = 3.0, 15.0
TABLE = 100.0                   # 查询真值 ~ U(0, 100); 逐出后猜先验中心 50
EPISODES = 400
NS = [50, 100, 200, 400, 800, 1600]

NAMES1 = ["FIFO-4", "Full", "Hybrid", "FIFO-eq"]
NAMES2 = ["w/o Anchor", "w/o Gist", "w/o Window", "Full", "Ours"]


def hybrid_cost(N):
    """Hybrid 的上下文 token 成本: (锚+近期)全 token + 其余帧各 1 gist。"""
    mid = max(0, N - N_ANCHOR - N_RECENT)
    return (N_ANCHOR + N_RECENT) * L_FULL + mid * M_GIST


def fifo_eq_window(N):
    """等预算 FIFO 的窗口: 与 Hybrid 相同预算能买到的全 token 帧数。"""
    return int(hybrid_cost(N) // L_FULL)


def sample_queries(N):
    """一集的查询: (年龄, 类型)。onset 1 条 / mid 8 条 / recent 2 条。"""
    qs = [(N - 1 - int(rng.integers(0, N_ANCHOR)), "onset")]
    qs += [(int(rng.integers(8, N - 8)), "mid") for _ in range(8)]
    qs += [(int(rng.integers(0, N_RECENT)), "recent") for _ in range(2)]
    return qs


def estimate_error(variant, age, N):
    """策略 variant 对年龄 age 的历史信息的检索误差 |err|。

    存储层级: 近期(age < N_RECENT)与锚帧(age >= N - N_ANCHOR)为全 token,
    其余看策略; 逐出的位置只剩先验。
    """
    is_recent = age < N_RECENT
    is_anchor = age >= N - N_ANCHOR

    def full(sigma_extra=0.0):
        return abs(rng.normal(0, np.hypot(SIG_FULL, sigma_extra)))

    def gist():
        return abs(rng.normal(0, SIG_GIST))

    def prior():
        return abs(rng.uniform(0, TABLE) - TABLE / 2)

    if variant == "FIFO-4":
        return full() if is_recent else prior()
    if variant == "FIFO-eq":
        return full() if age < fifo_eq_window(N) else prior()
    if variant == "Full":
        return full(SIG_INT * N)
    if variant == "Hybrid":
        return full() if (is_recent or is_anchor) else gist()
    # ---- N=400 消融变体(对标论文 Table 3) ----
    if variant == "w/o Anchor":        # 开局只剩 gist
        return full() if is_recent else gist()
    if variant == "w/o Gist":          # 中程全部逐出
        return full() if (is_recent or is_anchor) else prior()
    if variant == "w/o Window":        # 近期只剩 gist
        return full() if is_anchor else gist()
    raise ValueError(variant)


def tol_of(kind):
    return TOL_FINE if kind in ("onset", "recent") else TOL_COARSE


def eval_variant(N, variant):
    """返回 (onset/mid/recent 命中率, 总体命中率)。查询集逐集重抽。"""
    hits = {"onset": [], "mid": [], "recent": []}
    for _ in range(EPISODES):
        for age, kind in sample_queries(N):
            hits[kind].append(estimate_error(variant, age, N) <= tol_of(kind))
    rate = {k: float(np.mean(v)) for k, v in hits.items()}
    overall = (rate["onset"] + 8 * rate["mid"] + 2 * rate["recent"]) / 11
    return rate, overall


def main():
    print(f"L={L_FULL:.0f} tokens/full-frame, M={M_GIST:.0f} gist/frame (d=15), "
          f"anchor={N_ANCHOR}, recent={N_RECENT}; "
          f"sigma_full={SIG_FULL}, sigma_gist={SIG_GIST}, sigma_int={SIG_INT}*N (Full 的假设项)")
    print(f"tol_fine={TOL_FINE}(onset/recent), tol_coarse={TOL_COARSE}(mid); "
          f"查询构成: 1 onset + 8 mid + 2 recent; episodes={EPISODES}")

    # ---- ① 精度 vs N ----
    print("\n===== ① 各策略三类查询精度 vs 轨迹长度 N =====")
    print(f"{'N':>5} | " + " | ".join(f"{n:^32}" for n in NAMES1))
    print(f"{'':>5} | " + " | ".join(f"{'onset':>7} {'mid':>8} {'recent':>8} {'总体':>6}" for _ in NAMES1))
    for N in NS:
        cells = [eval_variant(N, name) for name in NAMES1]
        print(f"{N:>5} | " + " | ".join(
            f"{r['onset']:7.1%} {r['mid']:8.1%} {r['recent']:8.1%} {ov:6.1%}"
            for r, ov in cells))

    print("\n----- 上下文 token 成本 vs N (KV cache 大小) -----")
    print(f"{'N':>5} | " + " | ".join(f"{n:>8}" for n in NAMES1) + f" | {'FIFO-eq窗口':>8}")
    for N in NS:
        costs = [N_RECENT * L_FULL, N * L_FULL, hybrid_cost(N),
                 fifo_eq_window(N) * L_FULL]
        print(f"{N:>5} | " + " | ".join(f"{c:8.0f}" for c in costs)
              + f" | {fifo_eq_window(N):8d}")
    print("复杂度: Full O(N·L) / Hybrid O(N·L/d) / FIFO-4 O(1) / FIFO-eq 预算=Hybrid")

    # ---- ② N=400 组件消融 (对标论文 Table 3) ----
    N = 400
    print(f"\n===== ② N={N} 组件消融 (对标论文 Table 3) =====")
    print(f"{'变体':>12} | {'onset':>7} {'mid':>8} {'recent':>8} {'总体':>6}")
    for name in NAMES2:
        r, ov = eval_variant(N, "Hybrid" if name == "Ours" else name)
        print(f"{name:>12} | {r['onset']:7.1%} {r['mid']:8.1%} {r['recent']:8.1%} {ov:6.1%}")

    # ---- ③ N=800 精粒度查询 MAE vs 年龄 ----
    N = 800
    bins = [(0, 4), (4, 16), (16, 64), (64, 256), (256, N - N_ANCHOR)]
    print(f"\n===== ③ N={N}: 精粒度查询(tol={TOL_FINE})的 MAE 与命中率 vs 查询年龄 "
          f"(FIFO-eq 窗口 = {fifo_eq_window(N)} 帧) =====")
    print(f"{'年龄桶':>10} | " + " | ".join(f"{n + '  MAE/命中':^18}" for n in ["Hybrid", "FIFO-eq", "Full"]))
    for lo, hi in bins:
        row = []
        for name in ["Hybrid", "FIFO-eq", "Full"]:
            errs = [estimate_error(name, int(rng.integers(lo, hi)), N) for _ in range(300)]
            row.append((float(np.mean(errs)), float(np.mean(np.array(errs) <= TOL_FINE))))
        print(f"{f'{lo}-{hi}':>10} | " + " | ".join(f"{m:8.2f} / {h:6.1%}" for m, h in row))
    print("注: 年龄 >= N-2 的 onset 由锚帧覆盖(全 token); Hybrid 的中程只余 gist"
          "(σ=6 > tol), FIFO-eq 的中程大多被逐出。")


if __name__ == "__main__":
    main()
