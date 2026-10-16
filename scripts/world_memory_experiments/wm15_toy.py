"""WM_15 toy: 对象级掩码为什么让交互推理变得「必要」——捷径解的训练信号放大器

世界: K=6 个对象在 2D 平面上, 每步
    v_i += F_i(t),  x_i += v_i,
其中交互场 F_i(t) = gain * sum_{j≠i} u(x_j(t)) 只依赖**其他对象的位置**(可观测),
u 是确定的光滑向量场——对应 C-JEPA 里「被掩对象的状态可以从其他实体推断」
(影响邻域, Theorem 1)的最小标本。每个 episode 从随机初速/位置开始。

掩码协议(仿 C-JEPA): 对象 i 在 t0=10 前可观测(锚点 = 最后一次完整观测
(x_i, v_i)), t0 之后其全部轨迹被掩掉。三种估计器预测 x_i(t0+Δ):

  AnchorReplay  报锚点位置(冻结)              —— 纯检索下限
  SelfDyn       x_i(t0) + v_i(t0)·Δ           —— 自动力学捷径(时序插值),
                                                  掩码要杀死的那条捷径
  Interact      SelfDyn + 双重积分观测到的交互场 —— C-JEPA 希望模型学出的解

反事实查询(仿 CLEVRER counterfactual): 给定到 t0 为止的观测历史, 在 t0 时刻移走
对象 j, 问 i 在 t0+Δ 的位置变化。反事实世界从真实世界的 t0 状态**分支**出去
(共享全部历史, 只改未来)——这才是「给定观测做干预」的语义。
  SelfDyn 对反事实的预测: 0(它根本不用别人的状态)
  Interact 的预测: -双重积分 g·u(x_j) 项(一阶效应; 高阶耦合——别人也因 j 消失
    而改变轨迹——是它的残余误差来源)

训练信号放大器(核心论点的定量版):
  无掩码目标(下一步预测, 自有历史可见): 自动力学之后每步残差 = ||F_i|| —— 交互
    特征能解释的损失份额极小, 梯度压力弱;
  掩码目标(Δ 步补全): 自动力学(锚点外推)残差 = ||∫∫F|| ≈ ||F̄||·Δ²/2 —— 全部
    位移只能靠交互特征解释。
  放大系数 ≈ Δ²/2: 掩码不添加任何信息, 只是把捷径堵死, 把交互特征的梯度
    信号放大 Δ²/2 倍。对每个 Δ 实测两种残差并报告比值。
"""

import numpy as np

rng = np.random.default_rng(15)

K = 6
T0 = 10            # 掩码时刻(之前可观测)
DELTAS = [2, 5, 10, 20, 30]
SIGMA_OBS = 0.05   # 对其他对象位置的观测噪声
GAIN = 0.012       # 交互场强度
EPISODES = 400
D = 2


def u_field(x):
    """光滑确定性向量场 u: R^2 -> R^2, 只看位置。"""
    return np.array([np.cos(0.31 * x[0] + 0.17 * x[1]),
                     np.sin(0.23 * x[1] - 0.11 * x[0])])


def field_on(i, X_t, removed=None):
    """对象 i 受到的交互场 = gain * sum_{j≠i} u(x_j)。(不含 i 自身, 可观测)"""
    f = np.zeros(D)
    for j in range(K):
        if j != i and j != removed:
            f += GAIN * u_field(X_t[j])
    return f


def simulate(x0=None, v0=None, removed=None, T=None):
    """跑一个 episode。可传入初始条件(反事实世界与真实世界共享同一初值)。"""
    T = T or (T0 + max(DELTAS) + 1)
    X = np.zeros((T, K, D))
    X[0] = x0 if x0 is not None else rng.uniform(-4, 4, (K, D))
    V = v0.copy() if v0 is not None else rng.normal(0, 0.2, (K, D))
    alive = [j for j in range(K) if j != removed]
    for t in range(T - 1):
        for i in alive:
            V[i] += field_on(i, X[t], removed)
        for i in alive:
            X[t + 1, i] = X[t, i] + V[i]
        if removed is not None:  # 被移走的对象冻结在原位, 且不产生场
            X[t + 1, removed] = X[t, removed]
    return X


def observe_others(X, t, i):
    """其他对象在时刻 t 的带噪位置 → 交互场的带噪估计。"""
    f = np.zeros(D)
    for j in range(K):
        if j != i:
            f += GAIN * u_field(X[t, j] + rng.normal(0, SIGMA_OBS, D))
    return f


def branch(Xa, removed, d):
    """从 Xa 的 T0 状态分支: 之后移走 removed, 演化 d 步(共享到 T0 为止的历史)。"""
    x = Xa[T0].copy()
    v = (Xa[T0] - Xa[T0 - 1]).copy()
    traj = np.zeros((d + 1, K, D))
    traj[0] = x
    for t in range(d):
        for i in range(K):
            if i != removed:
                v[i] += field_on(i, x, removed)
        for i in range(K):
            if i != removed:
                x[i] += v[i]
        traj[t + 1] = x
    return traj


def main():
    # ---- (1) 掩码补全误差 vs Δ ----
    err = {k: {d: [] for d in DELTAS} for k in
           ["AnchorReplay", "SelfDyn", "Interact"]}
    for _ in range(EPISODES):
        X = simulate()
        for i in range(K):  # 每个对象轮流当被掩对象
            x0 = X[T0, i].copy()
            v0 = X[T0, i] - X[T0 - 1, i]  # 锚点速度: 两次观测差分即得
            for d in DELTAS:
                tgt = X[T0 + d, i]
                e_replay = np.linalg.norm(x0 - tgt)
                e_self = np.linalg.norm(x0 + v0 * d - tgt)
                # Interact: 双重积分 t0..t0+d-1 的带噪交互场
                est = x0 + v0 * d
                for tau in range(d):
                    est = est + (d - tau) * observe_others(X, T0 + tau, i)
                e_int = np.linalg.norm(est - tgt)
                err["AnchorReplay"][d].append(e_replay)
                err["SelfDyn"][d].append(e_self)
                err["Interact"][d].append(e_int)

    print("===== (1) 掩码补全: 位置误差 vs 掩码时长 Δ (均值 ± 标准误) =====")
    names = ["AnchorReplay", "SelfDyn", "Interact"]
    print(f"{'Δ':>4} | " + " | ".join(f"{n:>18}" for n in names))
    for d in DELTAS:
        row = []
        for n in names:
            a = np.array(err[n][d])
            row.append(f"{a.mean():8.3f}±{a.std()/np.sqrt(len(a)):5.3f}")
        print(f"{d:>4} | " + " | ".join(f"{r:>18}" for r in row))

    # ---- (2) 反事实查询: 在 t0 移走对象 j, i 的位置变化 ----
    err_cf = {"SelfDyn": [], "Interact": []}
    n_cf = 0
    while n_cf < EPISODES // 2:
        Xa = simulate(T=T0 + 11)                    # 真实世界(观测到 t0)
        j = int(rng.integers(K))
        i = int(rng.integers(K))
        if i == j:
            continue
        d = 10
        Xb = branch(Xa, removed=j, d=d)             # 反事实: 从 t0 分支, 无 j
        effect_true = Xb[d, i] - Xa[T0 + d, i]
        # SelfDyn 的反事实预测: 0(不看别人)
        err_cf["SelfDyn"].append(np.linalg.norm(effect_true))
        # Interact 的反事实预测: -ΣΣ g·u(x_j)(一阶效应)
        est_effect = np.zeros(D)
        for tau in range(d):
            est_effect -= (d - tau) * GAIN * u_field(Xa[T0 + tau, j])
        err_cf["Interact"].append(np.linalg.norm(est_effect - effect_true))
        n_cf += 1

    print("\n===== (2) 反事实效应估计误差 (Δ=10, 移走一个对象) =====")
    for n in err_cf:
        a = np.array(err_cf[n])
        print(f"  {n:>10}: {a.mean():.3f} ± {a.std()/np.sqrt(len(a)):.3f}")

    # ---- (3) 训练信号放大器: 两种目标下「只有交互特征能解释」的残差 ----
    print("\n===== (3) 训练信号放大器 =====")
    res_unmasked, res_masked = [], []
    for _ in range(EPISODES):
        X = simulate(T=T0 + 21)
        i = int(rng.integers(K))
        # 无掩码目标: 下一步预测, 自有历史可见 → 捷径(匀速外推)残差 = ||F_i||
        res_unmasked.append(np.linalg.norm(field_on(i, X[T0])))
        # 掩码目标: Δ 步补全 → 锚点匀速外推残差 = ||∫∫F||
        for d in [10, 20]:
            v0 = X[T0, i] - X[T0 - 1, i]
            est = X[T0, i] + v0 * d
            res_masked.append((d, np.linalg.norm(X[T0 + d, i] - est)))
    ru = np.mean(res_unmasked)
    print(f"  无掩码目标(下一步预测): 捷径残差 = E||F|| = {ru:.4f}")
    for d in [10, 20]:
        rm = np.mean([v for (dd, v) in res_masked if dd == d])
        print(f"  掩码目标(Δ={d:>2} 补全): 捷径残差 = {rm:8.3f}"
              f"   → 放大系数 = {rm/ru:7.1f}×   (理论 Δ²/2 = {d*d/2:.0f}×)")


if __name__ == "__main__":
    main()
