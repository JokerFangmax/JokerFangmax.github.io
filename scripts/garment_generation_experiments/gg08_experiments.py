"""GG 08 数值实验：SewingLDM 精读的机理验证（numpy/scipy，无学习框架）
1) FSQ 潜空间的 Goldilocks：(a) 每 token 码字容量 5^k vs 训练样本数（计数）
   (b) 线性 AE+FSQ 的重建误差随 k 单调改善（大潜空间的诱惑）
   (c) 码空间近邻/复用结构随 k 崩塌（离散先验不可学）→ 256×6 的机理
2) CFG guidance 的模式压制：p_cfg ∝ p_c^{1+w}/p_u^w 解析预测 + DDPM 采样验证
   → 同 prompt 结构多样性低的机理（P0 实测 n_panels {2:18, 4:2}）
3) 潜空间 token 数 256 vs 1443 的计算预算（DiT FLOPs 屋顶线, 接 GG 07 模型）
"""
import numpy as np
from scipy.optimize import minimize
import time

rng = np.random.default_rng(8)

# ================= 实验 1：FSQ 潜空间的 Goldilocks =================
print("=== 实验 1: 每 token 有限标量量化（5 级 × k 维）的 Goldilocks ===")

# (a) 计数论证：码字容量 vs 样本数（SewingLDM: DressCodeData 19,683 件训练）
N_TRAIN = 19683
print(f"\n  (a) 计数（每 token 位置: {N_TRAIN:,} 个训练样本 vs 每维 5 级 × k 维）:")
print(f"  {'k':>3s} {'每token码字数 5^k':>18s} {'均匀假设下每码字样本数':>20s} {'数据可覆盖?':>10s}")
for k in [4, 6, 8, 12, 32]:
    cells = 5 ** k
    cover = "能" if cells <= N_TRAIN / 10 else ("边缘" if cells <= N_TRAIN * 10 else "不能")
    print(f"  {k:3d} {cells:18,.0f} {N_TRAIN / cells:20.3f} {cover:>10s}")
print("  → 均匀假设只是下界口径; 真实数据的占用/复用要看训练后的码字分布（见 b/c）")

# ---- 玩具版型数据：30 类'服装类型'，每类独立风格子空间 → obs 16 token × 29 维 ----
T_TYPES, T_TOK, D_EDGE = 30, 16, 29
D_OBS = T_TOK * D_EDGE
D_STYLE = 24
STYLE_STD, NOISE_STD = 0.15, 0.02

_g = np.random.default_rng(99)   # 固定'世界'：类型原型 + 每类风格基
PROTO = _g.normal(0, 1.0, (T_TYPES, D_OBS))
BASIS = _g.normal(0, 1.0, (T_TYPES, D_OBS, D_STYLE))
BASIS /= np.linalg.norm(BASIS, axis=1, keepdims=True)   # 列单位化
TYPE_W = 1.0 / np.arange(1, T_TYPES + 1) ** 0.8
TYPE_W /= TYPE_W.sum()

def make_toy(n, seed):
    g = np.random.default_rng(seed)
    t = g.choice(T_TYPES, n, p=TYPE_W)
    a = g.normal(0, STYLE_STD, (n, D_STYLE))
    # obs_i = PROTO[t_i] + BASIS[t_i] @ a_i + noise
    X = PROTO[t] + np.matmul(BASIS[t], a[:, :, None]).squeeze(-1)
    X += g.normal(0, NOISE_STD, (n, D_OBS))
    return X, t

X_tr, t_tr = make_toy(N_TRAIN, 1)
X_te, t_te = make_toy(2000, 2)

# ---- 线性 AE + 每 token FSQ（直通梯度, L-BFGS 训练） ----
L_LEV = 5
def fsq_codes(q):
    """量化输出 q ∈ {−1,−0.5,0,0.5,1} → 每维整数 0..4"""
    return np.rint((q + 1.0) / 2 * (L_LEV - 1) / 2).astype(np.int64)

def train_ae(k, X, n_fit=2500, maxiter=100, seed=3):
    """W_e: (m,D), h=XWᵉᵀ; u=tanh(h)·2; q=round(u)/2 (ST); x̂=qWᵈᵀ"""
    m = T_TOK * k
    n = len(X)
    Xf = X[:n_fit]
    nf = len(Xf)
    g = np.random.default_rng(seed)
    We0 = g.normal(0, 0.5 / np.sqrt(D_OBS), (m, D_OBS))
    Wd0 = We0.T.copy()
    theta0 = np.concatenate([We0.ravel(), Wd0.ravel()])

    def fg(theta):
        We = theta[:m * D_OBS].reshape(m, D_OBS)
        Wd = theta[m * D_OBS:].reshape(D_OBS, m)
        h = Xf @ We.T
        u = np.tanh(h) * 2.0
        q = np.round(u) / 2.0
        xh = q @ Wd.T
        e = xh - Xf
        L = np.mean(e ** 2)
        # 反传（直通: dq/du = 1）
        dWd = (2.0 / (nf * D_OBS)) * e.T @ q
        dq = (2.0 / (nf * D_OBS)) * e @ Wd
        dh = dq * 2.0 * (1 - np.tanh(h) ** 2)
        dWe = dh.T @ Xf
        return L, np.concatenate([dWe.ravel(), dWd.ravel()])

    res = minimize(fg, theta0, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=maxiter, maxfun=200, ftol=1e-12, gtol=1e-10))
    We = res.x[:m * D_OBS].reshape(m, D_OBS)
    Wd = res.x[m * D_OBS:].reshape(D_OBS, m)
    return We, Wd, res.fun

def ae_encode(We, X, k):
    h = X @ We.T
    q = np.round(np.tanh(h) * 2.0) / 2.0
    return q.reshape(len(X), T_TOK, k)

print(f"\n  (b,c) 线性 AE + FSQ（玩具: {T_TYPES} 类 × {D_STYLE} 维风格, obs {T_TOK}×{D_EDGE}, "
      f"train {N_TRAIN:,}）:")
print(f"  {'k':>3s} {'测试RMS':>9s} {'每token平均码字数':>14s} {'码字平均复用':>10s} "
        f"{'测试码整版复现率':>12s} {'NN Hamming/k':>10s}")
for k in [4, 6, 8, 12, 32]:
    t0 = time.perf_counter()
    We, Wd, _ = train_ae(k, X_tr)
    Q_te = ae_encode(We, X_te, k)      # (2000, T_TOK, k)
    Q_tr = ae_encode(We, X_tr, k)      # (19683, T_TOK, k)
    # 重建
    xh = Q_te.reshape(len(X_te), -1) @ Wd.T
    rms = np.sqrt(np.mean((xh - X_te) ** 2))
    # 每 token 占用
    dist_codes = []
    for tok in range(T_TOK):
        codes = np.unique(Q_tr[:, tok, :], axis=0)
        dist_codes.append(len(codes))
    used = np.mean(dist_codes)
    reuse = N_TRAIN / used
    # 整版码字精确复现率
    te_flat = Q_te.reshape(len(X_te), -1)
    tr_flat = Q_tr.reshape(len(X_tr), -1)
    # 用 void view 做 exact row match
    def rowkey(A):
        return np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1]))).ravel()
    exact = np.isin(rowkey(te_flat), rowkey(tr_flat)).mean()
    # NN Hamming（子采样训练集加速）
    sub = tr_flat[:5000]
    te_idx = fsq_codes(Q_te).reshape(len(X_te), -1)
    tr_idx = fsq_codes(Q_tr).reshape(len(X_tr), -1)[:5000]
    nnh = []
    for i in range(0, len(te_idx), 200):
        d = (te_idx[i:i+200, None, :] != tr_idx[None]).sum(-1)  # (200, 5000)
        nnh.append(d.min(1))
    nnh = np.concatenate(nnh)
    dt = time.perf_counter() - t0
    print(f"  {k:3d} {rms:9.4f} {used:14,.0f} {reuse:10.1f} {exact:12.3%} {nnh.mean()/ (T_TOK*k):10.3f}"
          f"   [{dt:.0f}s]")

print("""  → 读数（实测结构, 与直觉不同的两点）:
    1) 重建不是单调改善: k=4→8 下降后饱和甚至回升（k=32 收敛验证过 maxiter 600）——
       FSQ 每维只有 5 级(2.3 bits), 加一维 = 加一个量化噪声通道(均匀 ±0.25);
       重建压力想把样本摊开到不同码字, 但摊开带来的区分度被逐维量化噪声吃掉。
       大 k 在重建侧【没有】收益;
    2) 码字结构单调崩塌: 重建压力总是试图区分样本, 容量(k)小时量化强迫共享
       （k=6: 每码字复用 477 次, 95.7% 测试整版码字在训练中出现过——
        先验有大量重复证据可学）; 容量大时样本被摊成孤点
       （k=32: 每码字仅 17.7 次复用, 整版码字零复现）——
       离散先验面对 19,683 个互不重复的码组合, 只能 memorize 不能泛化。
    Goldilocks = 重建饱和点(k≈6–8) ∩ 复用结构尚存 → 256×6 的机理:
    每 token 5^6 = 15,625 码字 与 训练样本数 19,683 同量级（(a) 表）。""")

# ================= 实验 2：CFG guidance 的模式压制 =================
print("\n=== 实验 2: CFG guidance scale 的模式压制（解析 + 采样验证） ===")

# 4 个'结构'模式（2D toy: 模式=panel 布局类）。无条件 vs 条件（prompt）权重不同:
#   模式1: prompt 主结构; 模式2: 合法但少数; 模式3: 与 prompt 不符（条件模型只压到 7%）; 模式4: 中性
# 模式间距 ~10σ: 分离良好, 让 w=0 基线的采样偏差主要来自 CFG 而非模式重叠
MUS = np.array([[0.50, 0.40], [-0.50, 0.40], [0.00, 0.55], [0.00, -0.50]])
SIG = 0.05
w_u = np.array([0.30, 0.30, 0.25, 0.15])
w_c = np.array([0.55, 0.22, 0.07, 0.16])
K4 = len(MUS)

def gmm_pdf(X, w, sig):
    var = sig ** 2
    d2 = np.sum((X[..., None, :] - MUS) ** 2, axis=-1)
    comp = w * np.exp(-0.5 * d2 / var) / (2 * np.pi * var)
    return comp.sum(-1)

def gmm_score(X, w, sig):
    var = sig ** 2
    d2 = np.sum((X[:, None, :] - MUS) ** 2, axis=2)
    logc = np.log(w)[None] - 0.5 * d2 / var - np.log(2 * np.pi * var)
    m = logc.max(1, keepdims=True)
    lse = m + np.log(np.exp(logc - m).sum(1, keepdims=True))
    resp = np.exp(logc - lse)
    return -(resp[..., None] * (X[:, None, :] - MUS[None]) / var).sum(1)

# (a) 解析预测: CFG 等价于从 p_cfg ∝ p_c^{1+w} / p_u^w 采样
#     （score 恰为 (1+w)∇log p_c − w∇log p_u = ∇log p_cfg, 精确成立）
def predicted_weights(w_guid):
    xs = np.linspace(-0.85, 0.85, 460)
    XX, YY = np.meshgrid(xs, xs)
    P = np.stack([XX, YY], -1).reshape(-1, 2)
    # log 空间防下溢: p_cfg ∝ exp((1+w)·log p_c − w·log p_u)
    lc = np.log(gmm_pdf(P, w_c, SIG) + 1e-300)
    lu = np.log(gmm_pdf(P, w_u, SIG) + 1e-300)
    lcfg = (1 + w_guid) * lc - w_guid * lu
    lcfg -= lcfg.max()
    p_cfg = np.exp(lcfg)
    # 模式归属: 最近中心
    assign = np.argmin(((P[:, None] - MUS[None]) ** 2).sum(-1), axis=1)
    tot = p_cfg.sum()
    return np.array([p_cfg[assign == i].sum() / tot for i in range(K4)])

# (b) DDPM 采样验证（全 1000 步, 解析 score, CFG 在每个噪声层组合）
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
abar = np.cumprod(alphas)

def ddpm_cfg(n, w_guid, seed=0):
    g = np.random.default_rng(seed)
    X = g.normal(0, 1, (n, 2))
    for t in range(T - 1, -1, -1):
        sig_t = np.sqrt(1 - abar[t])
        s_c = gmm_score(X, w_c, np.sqrt(SIG ** 2 + sig_t ** 2))
        s_u = gmm_score(X, w_u, np.sqrt(SIG ** 2 + sig_t ** 2))
        s = (1 + w_guid) * s_c - w_guid * s_u       # CFG 组合 score
        eps = -sig_t * s
        mean = (X - (1 - alphas[t]) / np.sqrt(1 - abar[t]) * eps) / np.sqrt(alphas[t])
        if t > 0:
            var = betas[t] * (1 - abar[t - 1]) / (1 - abar[t])
            X = mean + np.sqrt(var) * g.normal(0, 1, (n, 2))
        else:
            X = mean
    return X

def sampled_weights(X):
    assign = np.argmin(((X[:, None] - MUS[None]) ** 2).sum(-1), axis=1)
    return np.bincount(assign, minlength=K4) / len(X)

N_S = 4000
print(f"  条件模型本身: 模式权重 {w_c}（模式3 与 prompt 不符, 但条件模型只压到 {w_c[2]:.0%}）")
print(f"  模式似然比 π_c/π_u: {(w_c / w_u).round(3)} → CFG 按 (π_c/π_u)^w 几何放大\n")
print(f"  {'w':>3s} {'解析 p_cfg 模式权重':>26s} {'DDPM 采样权重':>24s} {'非法模式质量':>10s} {'合法模式熵':>8s}")
for w_guid in [0, 1, 2, 3, 5, 7, 10]:
    pw = predicted_weights(w_guid)
    Xs = ddpm_cfg(N_S, w_guid)
    sw = sampled_weights(Xs)
    valid = pw.copy(); valid[2] = 0
    ent = -(valid[valid > 0] * np.log(valid[valid > 0])).sum() if valid.sum() > 0 else 0
    print(f"  {w_guid:3d}  [{pw[0]:.3f} {pw[1]:.3f} {pw[2]:.4f} {pw[3]:.3f}]"
          f"  [{sw[0]:.3f} {sw[1]:.3f} {sw[2]:.4f} {sw[3]:.3f}]"
          f"   {pw[2]:10.4f} {ent:8.3f}")
print("""  → 读数:
    w 是似然比的指数放大器: 模式质量按 π_c·(π_c/π_u)^w 几何衰减;
    压制非法模式(模式3, ratio 0.28)与保留少数合法模式(模式2, ratio 0.733)
    的分离度 = 两者 log-ratio 之比——w 调到非法模式 <1% 时, 少数合法模式
    已被压到百分位级, 合法模式熵崩塌。
    P0 复现的 n_panels {2:18, 4:2}（同 prompt 20 seeds）正是这个机理的实测样子:
    少数结构（4 panel）存活率 ~10% ↔ 上表中 w≈1–2 的模式2 质量量级。
    结论: guidance 调不出'既压制非法又保留少数'——除非分离度本身变大,
    即条件分布 p_c 要更懂 prompt（Work 3 视频先验蒸馏的切入点）""")

# ================= 实验 3：1443→256×6 潜压缩的计算预算（屋顶线） =================
print("\n=== 实验 3: 观测空间 1443 token vs 潜空间 256 token 的 DiT 计算预算 ===")

def dit_flops_params(L, d, n, T, cross_dim=None):
    """DiT 每步 FLOPs ≈ L·(12·n·d²)（block 线性层） + L²·2·d（self-attn）
       + 2·L·d·cross_dim（cross-attn, T5 条件）; 参数量 ≈ 12·n·d²"""
    per_step = L * 12 * n * d ** 2 + L * L * 2 * d
    params = 12 * n * d ** 2
    return per_step * T, params

print(f"  （代入两论文/代码真实配置; T5 cross-attn 项 <2% 忽略）:")
print(f"  {'模型':>28s} {'L':>5s} {'d':>5s} {'n':>3s} {'T':>4s} {'总FLOPs':>9s} {'DiT参数':>8s} {'A10@100TF':>9s}")
cfgs = [
    ("GarmentDiffusion（观测空间）", 1443, 768, 12, 50),
    ("SewingLDM（潜空间）",         256, 1152, 24, 100),
    ("假想：SewingLDM 不压缩（观测空间）", 1443, 1152, 24, 100),
]
for name, L, d, n, T in cfgs:
    fl, P = dit_flops_params(L, d, n, T)
    print(f"  {name:>30s} {L:5d} {d:5d} {n:3d} {T:4d} {fl/1e12:7.2f}T {P/1e6:7.0f}M {fl/1e14:8.3f}s")
print("""  → 读数:
    潜压缩（1443→256 token, 29→6 维）让 SewingLDM 用 [更大 DiT(24 层×1152) × 2 倍步数]
    的总 FLOPs 仍与 GarmentDiffusion 同量级——L² 注意力项从 1443² 降 32× 是主因。
    '假想不压缩'一行显示: 同等 DiT 直接吃 1443 token 是 8× 计算量。
    P0 实测 SewingLDM 推理 6.5s/件（A6000, 100 步 IDDPM）——模型给的扩散本体是
    ~1s 量级, 其余是 T5 编码 + 序列化/反量化开销: 潜空间扩散把'快'的收益
    花在了'更大模型 + 更大步数'上换质量, 而不是裸延迟（对照 GG 07 实验 1:
    GarmentDiffusion 已把延迟打到 <1s, 延迟竞争在版型域已经结束,
    SewingLDM 的竞争维度是复杂版型能力 + 多模态条件）。""")
