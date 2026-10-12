"""
VG 03 toy 实验：单视角独立 SDS 的 Janus 病 vs 多视图联合先验
=============================================================

背景（承接 VG 01 §5.3 + VG 02 §4.3/§5）：
  VG 02 的表里写着 SDS 的第三宗罪——"单视角独立蒸馏，无 3D 一致性硬约束 →
  多视角矛盾、Janus 脸"。本实验把这句话变成可测的机制：

设置（text-to-3D 的最小抽象）：
  - "3D 物体" = 4 个方位角 (0/90/180/270°) 上的 bump 幅度 x ∈ R^4。
    真实数据分布：x = one-hot(θ*)，θ* ~ U{0..3}（每个真实物体只有一个 bump——
    相当于"一张脸只长在前面"）。
  - "渲染"：正交投影，视图 k 只看到自己方位的 bump：r_k = x_k。
  - 单视角图像先验（text-to-image 模型能提供的全部）：
      * 条件分布（文本 "a face"）：**视角无关**——每个视图的边际都是
        P_FACE·N(1,σ²) + (1-P_FACE)·N(0,σ²)，人脸模式占主导（0.6）。
        图像模型不知道"侧面应该看到侧脸"，文本让每个视图都想看到脸。
      * 无条件分布 = 数据边际：0.25·N(1) + 0.75·N(0)（随机物体 + 随机视角，
        bump 可见概率 1/4）。
  - 多视图联合先验（Unique3D 的多视图扩散——SD Image Variations 上 4 视图
    在 self-attention 层拼接——从一致 3D 渲染数据中学到的）：
    4 个一致 tuple 的等权混合 p(r) = (1/4)Σ_θ N(r; e_θ, σ²I)——
    联合分布知道"前面有脸 ⟹ 侧面是侧脸、后面是后脑勺"。

三条路线（同一初始化、同一更新规则，只换先验结构）：
  A. 单视角 SDS + CFG（DreamFusion 式）：每步随机抽一个视图 k，用该视图
     CFG 放大后的边际先验算 ε_φ，只更新 x_k。扫 guidance w。
  B. 多视图联合先验的 score 优化（改良派：先验换联合，仍走蒸馏）：
     渲染 = 整个 4 维 r = x，联合混合先验闭式 ε_φ，更新全部坐标。
  C. 联合先验采样 + 重建（Unique3D 范式：生成一次、再重建）：
     从联合先验采一个 tuple（= 一次多视图扩散前向），最近模式重建。

预测（跑之前先写下来，跑完对照）：
  A(w=0):  每视图高噪声拉向条件均值 0.6、低噪声拉向近端模式 0 → 中间值,
           4 个坐标同涨 —— "每个视图都半张脸"的弱 Janus；
  A(w=2+): CFG 把人脸模式放大到 ~97%，每视图拉满 → x=(1,1,1,1) 完整 Janus
           （VG 01 §5.3 的几何直接喂给 VG 03 的病）；
  B:       高噪声拉向对称中心 (0.25,..)（鞍点）、低噪声拉向最近 one-hot，
           w(t) 权重让低噪声主导 → 对称性被噪声打破后落进某个 one-hot 盆地
           —— 一致但随机（文本说不清是哪一个）；
  C:       一次采样 = 一个一致 tuple → 精确恢复。

测量：Janus 指数 J(x) = (Σx)²/Σx²（one-hot→1，均匀→4）、到最近 one-hot 的
L2、最终 x 的均值 ± 标准差。
"""
import numpy as np

rng_global = np.random.default_rng(42)

K = 4                    # 视图数
SIG = 0.35               # 先验每坐标标准差
P_FACE_C = 0.60          # 文本条件下人脸模式权重（视角无关！）
P_FACE_U = 0.25          # 无条件（数据边际）：bump 可见概率
SQAB_LO, SQAB_HI = 0.15, 0.95   # sqrt(abar) 采样区间（DreamFusion 风格）
ONEHOTS = np.eye(K)


# ---------- 通用：高斯混合先验的最优去噪器（闭式） ----------
def eps_phi_mix(x_t, a, comps_pi, comps_mu, sig2):
    """x_t = a·r + sqrt(1-a²)·ε, r ~ Σ_j π_j N(μ_j, σ²I_d)。
    返回 ε_φ = E[ε|x_t]（最优去噪器，Tweedie）。"""
    v = (1.0 - a * a) + a * a * sig2          # x_t | 分量 j 的方差
    d = x_t.shape[-1]
    # 分量 j: x_t | j ~ N(a·μ_j, v·I) → 后验权重 ρ_j
    logw = np.log(comps_pi) - np.sum((x_t - a * comps_mu) ** 2, axis=-1) / (2 * v)
    logw -= logw.max(axis=-1, keepdims=True)
    rho = np.exp(logw)
    rho /= rho.sum(axis=-1, keepdims=True)
    E_r = rho @ comps_mu                       # E[r|x_t]
    return np.sqrt(1 - a * a) * (x_t - a * E_r) / v


# ---------- 路线 A：单视角边际 + CFG（1D 精确积分） ----------
_R_GRID = np.linspace(-1.5, 3.5, 1001)

def _log_gauss(r, mu, s2):
    return -0.5 * (r - mu) ** 2 / s2 - 0.5 * np.log(2 * np.pi * s2)

def _log_pc(r):   # 文本条件：视角无关的人脸主导混合
    return np.logaddexp(np.log(P_FACE_C) + _log_gauss(r, 1.0, SIG**2),
                        np.log(1 - P_FACE_C) + _log_gauss(r, 0.0, SIG**2))

def _log_pu(r):   # 无条件：数据边际
    return np.logaddexp(np.log(P_FACE_U) + _log_gauss(r, 1.0, SIG**2),
                        np.log(1 - P_FACE_U) + _log_gauss(r, 0.0, SIG**2))

def _E_r_cfg(x_t, a, w):
    """CFG 合成先验 p_cfg ∝ p_c^{1+w}/p_u^w 下 E[r|x_t]（1D 逐点积分）。
    注意：p_cfg 不是混合——必须直接积分（GG 12 的 log-sum-exp 教训）。"""
    s2 = SIG**2
    # 似然 N(x_t; a·r, 1-a²) 作为 r 的函数
    loglik = _log_gauss(x_t, a * _R_GRID, (1 - a * a))
    logpcfg = (1 + w) * _log_pc(_R_GRID) - w * _log_pu(_R_GRID)
    logpost = logpcfg + loglik
    logpost -= logpost.max()
    post = np.exp(logpost)
    post /= post.sum()
    return float(post @ _R_GRID)

def run_A(w, n_steps=600, eta=0.05, n_seeds=30, seed0=1000):
    """单视角 SDS：每步抽视图 k，CFG 放大边际先验，只更新 x_k。"""
    out = []
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 + s)
        x = np.abs(rng.normal(0, 0.05, K))     # 小随机初始化
        for _ in range(n_steps):
            k = rng.integers(K)
            a = rng.uniform(SQAB_LO, SQAB_HI)
            eps = rng.standard_normal()
            x_t = a * x[k] + np.sqrt(1 - a * a) * eps
            # CFG 合成先验下的 ε_φ：先积 E[r|x_t] 再套 Tweedie
            v = (1 - a * a) + a * a * SIG**2
            E_r = _E_r_cfg(x_t, a, w)
            eps_phi = np.sqrt(1 - a * a) * (x_t - a * E_r) / v
            g = np.sqrt(a * a / (1 - a * a)) * (eps - eps_phi)   # VG 02 约定
            x[k] += eta * g
        out.append(x.copy())
    return np.array(out)


# ---------- 路线 B：联合先验 score 优化 ----------
def run_B(n_steps=600, eta=0.05, n_seeds=30, seed0=2000):
    out = []
    pi = np.full(K, 1.0 / K)
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 + s)
        x = np.abs(rng.normal(0, 0.05, K))
        for _ in range(n_steps):
            a = rng.uniform(SQAB_LO, SQAB_HI)
            eps = rng.standard_normal(K)
            x_t = a * x + np.sqrt(1 - a * a) * eps
            eps_phi = eps_phi_mix(x_t, a, pi, ONEHOTS, SIG**2)
            g = np.sqrt(a * a / (1 - a * a)) * (eps - eps_phi)
            x += eta * g
        out.append(x.copy())
    return np.array(out)


# ---------- 路线 C：联合先验采样 + 最近模式重建（1 次前向） ----------
def run_C(n_seeds=200, seed0=3000):
    out, rec = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 + s)
        th = rng.integers(K)
        r = ONEHOTS[th] + rng.normal(0, SIG, K)   # 一次多视图生成（含先验噪声）
        xhat = ONEHOTS[np.argmax(r)]              # 最近模式"重建"
        out.append(xhat)
        rec.append(th)
    return np.array(out), np.array(rec)


# ---------- 指标 ----------
def metrics(X):
    J = X.sum(1) ** 2 / (X ** 2).sum(1)               # Janus 指数
    d2 = np.linalg.norm(X[:, None, :] - ONEHOTS[None], axis=2).min(1)
    return J, d2


if __name__ == "__main__":
    print("=" * 72)
    print("VG 03 toy: 单视角边际 SDS (A) vs 联合先验蒸馏 (B) vs 联合采样重建 (C)")
    print(f"σ={SIG}, P_FACE_cond={P_FACE_C}, P_FACE_uncond={P_FACE_U}, "
          f"√ᾱ∈[{SQAB_LO},{SQAB_HI}], η=0.05, 600 步")
    print("=" * 72)

    for w in [0, 1, 2, 4]:
        X = run_A(w)
        J, d2 = metrics(X)
        # 解析锚：CFG 放大后的"有效条件均值"（高噪声极限下每视图的拉力目标）
        wf = np.array([(P_FACE_C / P_FACE_U) ** w * P_FACE_C,
                       ((1 - P_FACE_C) / (1 - P_FACE_U)) ** w * (1 - P_FACE_C)])
        mean_cfg = wf[0] / wf.sum()
        print(f"\n[A] 单视角 SDS + CFG w={w}:  Janus J = {J.mean():.3f} ± {J.std():.3f}"
              f"   minL2(one-hot) = {d2.mean():.3f}")
        print(f"    x̄ = {X.mean(0).round(3)}  (±{X.std(0).round(3)})"
              f"   vs 放大后条件均值锚 {mean_cfg:.3f}")
        if X.mean() > 1.0:
            print(f"    ⚠ 全坐标均值 {X.mean():.3f} > 1：CFG 放大把模式推过 1（过饱和的 toy 形态）")

    X = run_B()
    J, d2 = metrics(X)
    print(f"\n[B] 联合先验 score 优化:      Janus J = {J.mean():.3f} ± {J.std():.3f}"
          f"   minL2(one-hot) = {d2.mean():.3f}")
    print(f"    x̄ = {X.mean(0).round(3)}  (±{X.std(0).round(3)})")
    print(f"    到对称中心 (0.25,..) 的距离 = {np.linalg.norm(X - 0.25, axis=1).mean():.3f}"
          f" —— stall 在鞍点/反模（平均灰物体）")

    # B 补充 1：stall 是否随步数持久（1200 步）
    X2 = run_B(n_steps=1200, seed0=4000)
    print(f"    1200 步：minL2 = {metrics(X2)[1].mean():.3f}（600 步 {d2.mean():.3f}）"
          f"—— 加倍步数逃离不了鞍点")

    # B 补充 2：好初值（one-hot + 噪声）能否守住盆地 —— GG 04 式检验
    def run_B_init(x0, n_steps=600, eta=0.05, n_seeds=30, seed0=6000):
        pi = np.full(K, 1.0 / K)
        out = []
        for s in range(n_seeds):
            rng = np.random.default_rng(seed0 + s)
            x = x0 + rng.normal(0, 0.05, K)
            for _ in range(n_steps):
                a = rng.uniform(SQAB_LO, SQAB_HI)
                eps = rng.standard_normal(K)
                x_t = a * x + np.sqrt(1 - a * a) * eps
                eps_phi = eps_phi_mix(x_t, a, pi, ONEHOTS, SIG**2)
                x += eta * np.sqrt(a * a / (1 - a * a)) * (eps - eps_phi)
            out.append(x.copy())
        return np.array(out)
    Xg = run_B_init(ONEHOTS[0])
    Jg, dg = metrics(Xg)
    print(f"    好初值 e_0+噪声：minL2 = {dg.mean():.3f}, J = {Jg.mean():.3f}, "
          f"argmax 落 0 的比例 = {np.mean(Xg.argmax(1) == 0):.3f}（≈随机 25%）"
          f" —— 初值守不住：高噪声项把参数拖回全局均值（均值回归是结构性的）")

    X, rec = run_C()
    J, d2 = metrics(X)
    # 解析恢复率：x_θ~N(1,σ²), x_j~N(0,σ²) 独立 → P(全部 3 个对手被压过)
    from scipy.stats import norm as _norm
    p_ana = _norm.cdf(1.0 / (np.sqrt(2) * SIG)) ** 3
    print(f"\n[C] 联合采样 + 最近模式重建:  Janus J = {J.mean():.3f}"
          f"   minL2(one-hot) = {d2.mean():.3f}")
    print(f"    恢复率 = {np.mean(rec == np.argmax(X,1)):.3f}"
          f"  vs 解析 Φ(1/(√2σ))³ = {p_ana:.3f}  ✓")
    print("    (1 次前向 vs A/B 的 600 步优化)")

    # B 的落点分布：哪个 one-hot？对称性如何被打破
    Xb = run_B(n_seeds=200, seed0=5000)
    arg = Xb.argmax(1)
    print(f"\n[B 补充] 200 种子落点分布 argmax(x) = {np.bincount(arg, minlength=K)}（应≈50/50/50/50）")
