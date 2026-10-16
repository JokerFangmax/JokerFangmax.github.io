"""WM_04 toy: RELIC 的压缩记忆——「压缩换覆盖」与 KV cache 算术

Part 1: 压缩 schedule 的算术核对（全部数字可由论文参数推出）
  Wan-2.1 14B, 480x832, 16 FPS, 20 s = 317 帧; ST-VAE 8x 空间 / 4x 时间压缩
  n_lat = (317-1)//4 + 1 = 80 个 latent
  每 latent token 数 = (480/8/2) x (832/8/2) = 30 x 52 = 1560 (patchify 2x2)
  全上下文 = 80 x 1560 = 124,800 ≈ 论文口径 "120K"
  schedule S = [1,4,2,4,4,4,2,4,4,2,4,4,4,2,4,4,2,4] (循环取用, s_i = S[i % 18])
  空间 s 倍下采样 → token 数 / s^2:
    mean fraction = (1*1 + 5*(1/4) + 12*(1/16)) / 18 = 3/18 = 1/6  → schedule 本身 6x
  压缩后 cache = 80*1560/6 = 20,800
  论文口径 "~4x, 120K → 30K" → 反解不压缩滚动窗 w = (30000-20800)/1560 ≈ 6 latents ≈ 1.5 s
  (w 是论文未给出的量——这是本脚本的推导, 不是论文陈述)

Part 2: 同一存储预算下「压缩换覆盖」vs「不压缩的近因」
  对应关系(与 RELIC 机制对齐):
    走廊位置 p <-> 一个已生成的过去 latent 帧(它自己的 8x8 token 网格)
    帧内空间冗余 <-> patch 内部平滑(压缩便宜) + 小的高频残差(压缩有代价)
    视角失配 <-> 不同 p 的 patch 内容不同(存储位置离查询位置越远误差越大)
  世界: 64 格环形走廊。F[p] = g(p)*S1 + h(p)*S2 + 高频噪声(0.12), 其中
    S1/S2 是固定的帧内缓变「坡形」图案(帧内空间冗余), g/h 是沿走廊的
    平滑随机游走(相关长度 ~2 格, 远处不相关)。
  Agent: 沿环走(方向持续, 2% 反向, 2% 随机传送), 观测噪声 sigma_obs=0.1。
  写入端理想化: 每个位置的最新带噪观测都可拿到, 再按策略在预算内选子集——
    隔离「保留什么」这一个变量(与 WM_01 toy 同一精神)。
  存储成本以 cell 计: 原始 patch 64 cells, 2x 压缩 16, 4x 压缩 4。预算 B = 512。
  策略:
    full-raw      全部 64 位置, 不压缩 (4096 cells, 超预算, 作地板参照)
    recency-raw   最近 8 步到过的位置, 不压缩 (512)
    recency-comp4 最近 128 步到过的位置, 4x 压缩 (<=256)
    geo-raw       每 8 格一个位置, 不压缩 (512)
    geo-comp2     每 2 格一个位置, 2x 压缩 (512)
    geo-comp4     每格一个位置, 4x 压缩 (256, 静态世界用不满)
  查询: 每 5 步, 当前位置 -> 环形最近存储位置 -> 解压 -> 与真值比 MAE,
        按「该位置距上次被观测的时长 Δ」分桶(记忆只含 t 之前的观测)。
  实测(40 episodes):
    4x 压缩损失 0.34 < 偏一格视角失配 0.42 —— 「压缩比失配便宜」
    等预算 512: geo-comp2 0.31 / geo-comp4 0.34(只用 256) vs geo-raw 0.64
    recency 被压缩把视界从 8 步拉到 128 步; 几何策略无悬崖(全程平坦)
"""

import numpy as np

rng = np.random.default_rng(11)

# ========== Part 1: schedule 算术 ==========

def part1():
    print("=" * 70)
    print("Part 1: RELIC 压缩 schedule 算术")
    print("=" * 70)
    frames, fps = 317, 16
    n_lat = (frames - 1) // 4 + 1
    tokens_per_lat = (480 // 8 // 2) * (832 // 8 // 2)
    full = n_lat * tokens_per_lat
    S = [1, 4, 2, 4, 4, 4, 2, 4, 4, 2, 4, 4, 4, 2, 4, 4, 2, 4]
    frac = np.mean([1.0 / (s * s) for s in S])
    comp = full * frac
    w_implied = (30000 - comp) / tokens_per_lat
    print(f"20 s = {frames} 帧 @ {fps} FPS -> {n_lat} latents (4x 时间压缩)")
    print(f"每 latent token 数 = 30 x 52 = {tokens_per_lat} (8x VAE + 2x2 patchify)")
    print(f"全上下文 = {n_lat} x {tokens_per_lat} = {full:,} (论文口径 ~120K)")
    print(f"schedule S 长度 {len(S)}: 1x{S.count(1)}, 2x{S.count(2)}, 4x{S.count(4)}")
    print(f"平均 token 份额 = (1 + 5/4 + 12/16)/18 = {frac:.4f} -> schedule 本身 {1/frac:.1f}x")
    print(f"压缩 cache = {full:,.0f} x 1/6 = {comp:,.0f}")
    print(f"论文口径 30K -> 反解滚动窗 w ≈ {w_implied:.1f} latents "
          f"≈ {w_implied * 4 / fps:.1f} s (推导值, 论文未给出 w)")


# ========== Part 2: 预算下的记忆策略 ==========
#
# 对应关系(与 RELIC 机制对齐):
#   走廊位置 p <-> 一个已生成的过去 latent 帧(它自己的 8x8 token 网格)
#   帧内空间冗余 <-> patch 内部平滑(压缩便宜) + 小的高频残差(压缩有代价)
#   视角失配 <-> 不同 p 的 patch 内容不同(存储位置离查询位置越远误差越大)
# 生成模型: F[p] = g(p)*S1 + h(p)*S2 + 高频噪声
#   S1/S2: 固定的帧内平滑 8x8 图案(波长 20-28) —— 压缩几乎无损的来源
#   g/h:   沿走廊的快变系数(波长 8-16) —— 位置失配产生误差的来源

L = 64          # 走廊长度
WIN = 8         # 视窗(帧)8x8
T = 3000        # 每 episode 步数
EPISODES = 40
QUERY_EVERY = 5
SIGMA_OBS = 0.1     # 观测(写入)噪声
SIGMA_HF = 0.12     # 帧内高频残差: 压缩的不可恢复代价
BUDGET = 512        # 存储 budget, 以 cell 数计
GAP_BINS = [0, 4, 8, 16, 32, 64, 128, 256, 10**9]

NAMES = ["full-raw", "recency-raw", "recency-comp4", "geo-raw", "geo-comp2", "geo-comp4"]
COSTS = {"full-raw": 64 * 64, "recency-raw": 8 * 64, "recency-comp4": 64 * 4,
         "geo-raw": 8 * 64, "geo-comp2": 32 * 16, "geo-comp4": 64 * 4}


def make_world():
    """返回 F, 其中 F[p] 是位置 p 的 8x8 patch

    F[p] = g(p)*S1 + h(p)*S2 + 高频噪声
      S1/S2: 固定的帧内「坡形」图案(近似线性, 缓变) —— 帧内空间冗余
      g/h:   沿走廊的平滑随机游走(相关长度 ~2 格) —— 位置失配误差来源
    远处位置不相关(而非正弦的反相关), 近处相关。
    """
    ii, jj = np.meshgrid(np.arange(WIN), np.arange(WIN), indexing="ij")

    def ramp_pattern():
        a, b = rng.normal(0, 1, 2)
        return (a * (ii - 3.5) + b * (jj - 3.5)) / 4.0

    def smooth_walk(sigma):
        x = rng.normal(0, 1, L)
        k = np.arange(-8, 9)
        kern = np.exp(-k ** 2 / (2 * sigma ** 2))
        kern /= kern.sum()
        y = np.convolve(x, kern, mode="same")
        return y / y.std() * 0.8

    S1, S2 = ramp_pattern(), ramp_pattern()
    g, h = smooth_walk(1.2), smooth_walk(1.2)
    F = (g[:, None, None] * S1[None] + h[:, None, None] * S2[None]
         + rng.normal(0, SIGMA_HF, (L, WIN, WIN)))
    return F


def compress(patch, s):
    if s == 1:
        return patch.copy()
    k = WIN // s
    return patch.reshape(k, s, k, s).mean(axis=(1, 3))


def decompress(small, s):
    if s == 1:
        return small
    # 最近邻上采样回 8x8 (保守: 不引入插值先验)
    return np.kron(small, np.ones((s, s)))


def run_episode():
    """先跑完整条轨迹; 再逐查询点按策略构造存储集(只用 t 之前的观测)评估"""
    F = make_world()
    traj = []
    pos, direction = rng.integers(L), rng.choice([-1, 1])
    for t in range(T):
        if rng.random() < 0.02:
            pos = rng.integers(L)
        elif rng.random() < 0.02:
            direction = -direction
        pos = (pos + direction) % L
        traj.append(pos)
    traj = np.array(traj)

    obs = {p: F[p] + rng.normal(0, SIGMA_OBS, (WIN, WIN)) for p in range(L)}

    # 静态几何策略的存储集: {位置: 压缩倍率}
    stores = {
        "full-raw":  {p: 1 for p in range(L)},
        "geo-raw":   {p: 1 for p in range(0, L, 8)},
        "geo-comp2": {p: 2 for p in range(0, L, 2)},
        "geo-comp4": {p: 4 for p in range(L)},
    }

    acc = np.zeros((len(GAP_BINS) - 1, len(NAMES)))
    cnt = np.zeros(len(GAP_BINS) - 1)
    last_seen = np.full(L, -1)

    for t in range(T):
        pos = traj[t]
        # 该位置距上次被观测的时长(本次之前)
        gap = t - last_seen[pos] if last_seen[pos] >= 0 else 10**9

        if t % QUERY_EVERY == 0 and t > 0:
            true_patch = F[pos]
            b = min(np.digitize(gap, GAP_BINS) - 1, len(GAP_BINS) - 2)

            # 记忆只能用 t 之前的观测; recency 窗口不含当前步
            recent8 = set(traj[max(0, t - 8):t])
            recent128 = set(traj[max(0, t - 128):t])
            live = dict(stores)
            live["recency-raw"] = {p: 1 for p in recent8}
            live["recency-comp4"] = {p: 4 for p in recent128}

            for ni, name in enumerate(NAMES):
                store = live[name]
                if not store:
                    err = np.abs(true_patch).mean()  # 无记忆 -> 全幅幻觉
                else:
                    # 环形距离最近的存储位置
                    dmin, pbest = min((min(abs(p - pos), L - abs(p - pos)), p) for p in store)
                    est = decompress(compress(obs[pbest], store[pbest]), store[pbest])
                    err = np.abs(est - true_patch).mean()
                acc[b, ni] += err
            cnt[b] += 1

        last_seen[pos] = t
    return acc, cnt


def part2():
    print()
    print("=" * 70)
    print("Part 2: 同一预算(512 cells)下的记忆策略, MAE 按重访间隔 Δ 分桶")
    print("=" * 70)
    acc = np.zeros((len(GAP_BINS) - 1, len(NAMES)))
    cnt = np.zeros(len(GAP_BINS) - 1)
    for _ in range(EPISODES):
        a, c = run_episode()
        acc += a
        cnt += c

    print(f"存储成本(cells): " + ", ".join(f"{n}={COSTS[n]}" for n in NAMES))
    print()
    print(f"{'Δ bin':>10} | " + " | ".join(f"{n:>13}" for n in NAMES) + " |    n")
    for bi in range(len(GAP_BINS) - 1):
        if cnt[bi] < 30:
            continue
        lo, hi = GAP_BINS[bi], min(GAP_BINS[bi + 1], 999)
        row = " | ".join(f"{acc[bi, k] / cnt[bi]:13.3f}" for k in range(len(NAMES)))
        print(f"{f'{lo}-{hi}':>10} | {row} | {int(cnt[bi]):5d}")

    # 长程段 (Δ>=128) 平均
    long_idx = [bi for bi in range(len(GAP_BINS) - 1) if GAP_BINS[bi] >= 128 and cnt[bi] >= 30]
    if long_idx:
        print()
        print("长程段 (Δ>=128) 平均 MAE:")
        for k, n in enumerate(NAMES):
            e = sum(acc[bi, k] for bi in long_idx) / sum(cnt[bi] for bi in long_idx)
            print(f"  {n:>14}: {e:.3f}")


if __name__ == "__main__":
    part1()
    part2()
