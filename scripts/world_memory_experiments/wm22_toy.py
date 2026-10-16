"""WM_22 toy: 测试时训练(TTT)作为持久记忆——RLS 快权重 vs 冻结先验 / 滑窗激活 / 朴素梯度写入

对应 World-T3 (arXiv:2610.17434) 的核心机制: 把「位姿 → 内容」联想记忆写进一个固定大小的
读出矩阵 (快权重 W2), 写入用在线递归最小二乘 (RLS, 论文 Eq. 4), 读取用一次前向 W2·φ(k)
(论文 Eq. 5)。

世界: 环形位姿空间 (P 个 bin), 每个 episode 一个新「场景」:
    f(θ) = f_shared(θ) + f_specific(θ) ∈ R^M
  f_shared 跨场景固定 (基座先验 W(0) 可以精确学到), f_specific 每 episode 重抽
  (只能靠测试时的写入吸收)。相机沿环平滑游走 (动量步进 + 周期性折返制造长短不一的
  重访间隔), 每步把当前 bin 的干净观测 (φ(θ), f(θ)) 写入记忆——对应论文里
  「块被逐出上下文窗时 consolidate 一次」。

查询: 每步对全部已观测过的 bin 用当前记忆读出估计, 按 Δ = 距该 bin 上次写入的
时长分桶统计 MAE。写入内容对所有策略完全相同, 只比较「怎么存」这一个变量:

  Frozen  冻结基座 W(0), 不做任何测试时写入 —— 无记忆下限
  Window  只保留最近 W_WIN 条 (key, value) 的滑窗激活记忆, 环距高斯核回归,
          窗内无证据时退回先验 —— 「激活即记忆」(上下文窗口 / KV cache) 的理想化
  SGD     朴素 TTT: 每次写入做一步梯度下降 W ← W − η(Wφ−v)φᵀ
          —— 论文点名的失效模式: 平滑轨迹 → 相邻键高度相关 → 病态曲率
  RLS     论文机制: 加权岭回归 (Eq. 3) 的精确在线解 (Eq. 4), 白化预条件 + 顺序不变

附加两个小实验:
  (a) 顺序不变性: 同一批写入按轨迹顺序 vs 随机打乱, 比较最终记忆的读出 MAE——
      RLS 每步都是全局最优解, 与顺序无关; SGD 不是。
  (b) η 敏感性: SGD 学习率扫描, 展示「快而不稳 vs 慢而欠适配」的两难。
"""

import numpy as np

rng = np.random.default_rng(22)

# ---- 世界与实验参数 ----
P = 192        # 环上 bin 数 (位姿空间分辨率)
M = 4          # 场景内容维度
D = 48         # 姿态键特征维度 (固定投影 W1 的输出)
T = 1500       # 每 episode 步数
EP = 24        # episode 数
W_WIN = 24     # 滑窗记忆容量 (写入条数)
BW = 1.5       # 滑窗核回归带宽 (bin)
ETA = 0.02     # SGD 写入学习率 (主表)
LAMBDA = 1e-3  # RLS 正则 (论文取值)
GAP_BINS = [0, 2, 4, 8, 16, 32, 64, 128, 10 ** 9]
QUERY_EVERY = 2

th = np.linspace(0, 2 * np.pi, P, endpoint=False)


def make_field(freqs, scale):
    """f(θ) = Σ_k [cos kθ · A_k + sin kθ · B_k], A/B ∈ R^M ~ N(0, scale²)"""
    A = rng.normal(0, scale, (len(freqs), M))
    B = rng.normal(0, scale, (len(freqs), M))

    def f(t):
        out = np.zeros(t.shape + (M,))
        for k, j in enumerate(freqs):
            out = out + np.cos(j * t)[:, None] * A[k] + np.sin(j * t)[:, None] * B[k]
        return out

    return f


# ---- 固定构件: 特征图 φ (论文的 W1, 部署时冻结) 与共享场 (先验可学) ----
_BAS = np.stack([np.ones(P), np.cos(th), np.sin(th),
                 np.cos(2 * th), np.sin(2 * th),
                 np.cos(4 * th), np.sin(4 * th)], 1)            # P×7
_F = rng.normal(0, 1, (7, D)) / np.sqrt(7)
PHI = np.tanh(_BAS @ _F)                                       # P×D, 固定
SHARED = make_field([1, 2], 1.0)(th)                           # P×M, 跨 episode 固定
# 基座先验 W(0): 对共享场的岭回归拟合 (部署时冻结)
_W0 = np.linalg.solve(PHI.T @ PHI + 1e-6 * np.eye(D), PHI.T @ SHARED)   # D×M
EST0 = PHI @ _W0                                               # 先验读出 P×M


def ring_dist(i, j):
    d = np.abs(np.asarray(i)[:, None] - np.asarray(j)[None, :])
    return np.minimum(d, P - d)


class RLS:
    """论文 Eq. 4 (w=1): g = Pφ/(1+φᵀPφ); W ← W + (v−φW)gᵀ; P ← P − gφᵀP"""

    def __init__(self):
        self.W = _W0.copy()
        self.Pm = np.eye(D) / LAMBDA

    def write(self, phi, v):
        Pphi = self.Pm @ phi
        g = Pphi / (1.0 + phi @ Pphi)
        resid = v - phi @ self.W
        self.W += np.outer(g, resid)
        self.Pm -= np.outer(g, Pphi)


class SGD:
    def __init__(self, eta):
        self.eta = eta
        self.W = _W0.copy()

    def write(self, phi, v):
        resid = phi @ self.W - v
        self.W -= self.eta * np.outer(phi, resid)


def window_est(window_bins, window_vals, idx):
    """滑窗激活记忆: 环距高斯核回归, 证据不足退回先验"""
    d = ring_dist(idx, window_bins)                    # n×W
    w = np.exp(-(d / BW) ** 2)
    s = w.sum(1)
    est = (w @ window_vals) / np.maximum(s, 1e-9)[:, None]
    fall = s < 0.35
    est[fall] = EST0[idx][fall]
    return est


def run_episode():
    f_true = SHARED + make_field([3, 4, 5, 6], 0.55)(th)     # P×M 本 episode 真值
    err0 = np.abs(EST0 - f_true).mean(1)                    # Frozen 恒定逐 bin 误差

    rls, sgd = RLS(), SGD(ETA)
    w_bins, w_vals = [], []                                 # 滑窗 (并行维护)
    last_seen = np.full(P, -1)
    writes = []                                             # (bin,) 全量写入记录

    acc = np.zeros((len(GAP_BINS) - 1, 4))
    cnt = np.zeros(len(GAP_BINS) - 1)

    pos = int(rng.integers(P))
    direction = 1 if rng.random() < 0.5 else -1
    speed = int(rng.integers(1, 3))
    rev_until = -1

    for t in range(T):
        # ---- 平滑相机轨迹: 动量步进, 偶发转向, 每 ~250 步折返 (out-and-back) ----
        if rng.random() < 0.03:
            direction *= -1
        if rng.random() < 0.02:
            speed = int(rng.integers(1, 3))
        if t >= rev_until and t > 0 and t % 250 == 0:
            direction *= -1
            rev_until = t + 60
        pos = (pos + direction * speed) % P

        # ---- 写入: 当前 bin 的干净观测 (对所有策略相同) ----
        v = f_true[pos]
        phi = PHI[pos]
        rls.write(phi, v)
        sgd.write(phi, v)
        w_bins.append(pos)
        w_vals.append(v)
        if len(w_bins) > W_WIN:
            w_bins.pop(0)
            w_vals.pop(0)
        last_seen[pos] = t
        writes.append(pos)

        # ---- 查询: 全部已观测 bin, 按 Δ = t − 上次写入 分桶 ----
        if t % QUERY_EVERY == 0 and t > 0:
            idx = np.where(last_seen >= 0)[0]
            gap = t - last_seen[idx]
            b = np.digitize(gap, GAP_BINS) - 1
            est = np.stack([
                EST0[idx],
                window_est(np.array(w_bins), np.array(w_vals), idx),
                PHI[idx] @ sgd.W,
                PHI[idx] @ rls.W,
            ], axis=2)                                        # n×M×4
            e = np.abs(est - f_true[idx][:, :, None]).mean(1)  # n×4
            for j in range(len(idx)):
                acc[b[j]] += e[j]
                cnt[b[j]] += 1

    # ---- 从未观测过的 bin: 参数记忆的泛化 vs 滑窗的空白 ----
    never = np.where(last_seen < 0)[0]
    e_never = np.stack([
        err0[never],
        err0[never],                                          # Window 无证据 → 先验
        np.abs(PHI[never] @ sgd.W - f_true[never]).mean(1),
        np.abs(PHI[never] @ rls.W - f_true[never]).mean(1),
    ], 1) if len(never) else np.zeros((0, 4))

    # ---- 顺序不变性: 同一批写入, 轨迹序 vs 打乱序 (离线复演) ----
    phi_seq = PHI[writes]
    v_seq = f_true[writes]
    perm = rng.permutation(len(writes))

    def replay(order, learner):
        for j in order:
            learner.write(PHI[writes[j]], f_true[writes[j]])
        return np.abs(PHI @ learner.W - f_true).mean()        # 全环读出 MAE

    m_rls_a = replay(list(range(len(writes))), RLS())
    m_rls_b = replay(perm, RLS())
    m_sgd_a = replay(list(range(len(writes))), SGD(ETA))
    m_sgd_b = replay(perm, SGD(ETA))
    # 批量岭回归全局最优解 (RLS 的解析对照)
    X = phi_seq.T @ phi_seq + LAMBDA * np.eye(D)
    W_batch = np.linalg.solve(X, phi_seq.T @ v_seq + LAMBDA * _W0)
    m_batch = np.abs(PHI @ W_batch - f_true).mean()

    return acc, cnt, e_never, (m_rls_a, m_rls_b, m_sgd_a, m_sgd_b, m_batch)


def main():
    acc = np.zeros((len(GAP_BINS) - 1, 4))
    cnt = np.zeros(len(GAP_BINS) - 1)
    e_never_all, ord_stat = [], []
    for _ in range(EP):
        a, c, en, st = run_episode()
        acc += a
        cnt += c
        if len(en):
            e_never_all.append(en)
        ord_stat.append(st)

    names = ["Frozen", "Window", "SGD", "RLS"]
    print("===== 重访误差 MAE vs Δ (距该 bin 上次写入的时长) =====")
    print(f"{'Δ bin':>12} | " + " | ".join(f"{n:>8}" for n in names) + " |     n")
    for bi in range(len(GAP_BINS) - 1):
        if cnt[bi] < 50:
            continue
        lo, hi = GAP_BINS[bi], min(GAP_BINS[bi + 1], 999)
        row = " | ".join(f"{acc[bi, k] / cnt[bi]:8.3f}" for k in range(4))
        print(f"{f'{lo}-{hi}':>12} | {row} | {int(cnt[bi]):6d}")

    if e_never_all:
        en = np.concatenate(e_never_all, 0).mean(0)
        print("\n===== 从未观测过的 bin (参数记忆的泛化) =====")
        print({n: round(float(x), 3) for n, x in zip(names, en)})

    st = np.array(ord_stat)
    print("\n===== 顺序不变性 (最终全环读出 MAE, 24 episode 均值) =====")
    print(f"  RLS   轨迹序 {st[:,0].mean():.4f} | 打乱序 {st[:,1].mean():.4f} "
          f"| 差 {np.abs(st[:,0]-st[:,1]).mean():.2e}")
    print(f"  SGD   轨迹序 {st[:,2].mean():.4f} | 打乱序 {st[:,3].mean():.4f} "
          f"| 差 {np.abs(st[:,2]-st[:,3]).mean():.4f}")
    print(f"  批量岭回归 (解析最优) {st[:,4].mean():.4f} —— RLS 应与之重合")

    # ---- η 敏感性 ----
    print("\n===== SGD 学习率扫描 (episode 1 复演, 最终全环 MAE) =====")
    f_true = SHARED + make_field([3, 4, 5, 6], 0.55)(th)
    pos_seq = []
    pos = int(rng.integers(P)); direction = 1; speed = 1; rev_until = -1
    for t in range(T):
        if rng.random() < 0.03: direction *= -1
        if rng.random() < 0.02: speed = int(rng.integers(1, 3))
        if t >= rev_until and t > 0 and t % 250 == 0:
            direction *= -1; rev_until = t + 60
        pos = (pos + direction * speed) % P
        pos_seq.append(pos)
    for eta in [0.005, 0.01, 0.02, 0.05, 0.1]:
        s = SGD(eta)
        for b in pos_seq:
            s.write(PHI[b], f_true[b])
        print(f"  η={eta:<6} MAE={np.abs(PHI @ s.W - f_true).mean():.4f}")
    r = RLS()
    for b in pos_seq:
        r.write(PHI[b], f_true[b])
    print(f"  RLS         MAE={np.abs(PHI @ r.W - f_true).mean():.4f}")


if __name__ == "__main__":
    main()
