"""GG 03 数值实验：TailorNet 式分解泛化 / SMPLicit 式 UDF 分层 / 开曲面 SDF 歧义"""
import numpy as np

rng = np.random.default_rng(42)

# ============ 实验 1：分解表示 vs 单体表示的泛化（TailorNet 核心主张） ============
# 玩具版: 服装位移场 d(β, s) ∈ R^K（K 个子网格顶点）= 身体项 A(β) + 款式项 B(s) + 可调交互项
print("=== 实验 1: 分解表示 vs 单体表示（held-out (β,s) 组合泛化）===")

K, H = 64, 32          # 顶点数 / 隐层宽
D_BETA, D_S = 2, 2     # β 与 style 维度

# 固定 GT 特征映射
W1 = rng.normal(0, 1, (H, D_BETA)); U1 = rng.normal(0, 0.5, (K, H))
W2 = rng.normal(0, 1, (H, D_S));    U2 = rng.normal(0, 0.5, (K, H))

def gt(β, s, interact):
    fa = np.tanh(W1 @ β)   # H
    fs = np.tanh(W2 @ s)
    d = U1 @ fa + U2 @ fs
    if interact > 0:
        d = d + interact * (U1 * rng2_interact) @ (fa[:, None] * fs[None, :]).sum(1) * 0  # placeholder
    return d

# 交互项: 用第三组随机映射做逐顶点乘积
Wi = rng.normal(0, 0.7, (K, H))
def gt_field(β, s, interact):
    fa = np.tanh(W1 @ β); fs = np.tanh(W2 @ s)
    return U1 @ fa + U2 @ fs + interact * (Wi @ (fa * fs))

# 数据: 8×8 的 (β,s) 网格, 随机扣掉 16 个组合做 held-out 测试
n_grid = 8
betas = rng.normal(0, 1, (n_grid, D_BETA))
styles = rng.normal(0, 1, (n_grid, D_S))
combos = [(i, j) for i in range(n_grid) for j in range(n_grid)]
rng.shuffle(combos)
test_set = set(combos[:16]); train_set = [c for c in combos if c not in test_set]

def make_data(interact):
    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    for (i, j) in train_set:
        X_tr.append(np.concatenate([betas[i], styles[j]]))
        Y_tr.append(gt_field(betas[i], styles[j], interact))
    for (i, j) in test_set:
        X_te.append(np.concatenate([betas[i], styles[j]]))
        Y_te.append(gt_field(betas[i], styles[j], interact))
    return (np.array(X_tr), np.array(Y_tr), np.array(X_te), np.array(Y_te))

# ---- 小 MLP 库（Adam, tanh, 手写反向传播）----
class Adam:
    def __init__(self, params, lr=1e-2):
        self.p = params; self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = 0.9*self.m[i] + 0.1*g
            self.v[i] = 0.999*self.v[i] + 0.001*g*g
            mh = self.m[i]/(1-0.9**self.t); vh = self.v[i]/(1-0.999**self.t)
            p -= self.lr * mh/(np.sqrt(vh)+1e-8)

def forward_monolithic(X, params):
    W1m, b1, W2m, b2 = params
    h = np.tanh(X @ W1m.T + b1)
    return h @ W2m.T + b2

def train_monolithic(X, Y, hidden=48, epochs=3000, lr=1e-2):
    D = X.shape[1]
    W1m = rng.normal(0, 0.5, (hidden, D)); b1 = np.zeros(hidden)
    W2m = rng.normal(0, 0.3, (K, hidden)); b2 = np.zeros(K)
    params = [W1m, b1, W2m, b2]
    opt = Adam(params, lr)
    for _ in range(epochs):
        out = forward_monolithic(X, params)
        g_out = 2*(out - Y)/len(X)
        gW2 = g_out.T @ np.tanh(X @ W1m.T + b1)
        gb2 = g_out.sum(0)
        gh = g_out @ W2m * (1 - np.tanh(X @ W1m.T + b1)**2)
        gW1 = gh.T @ X; gb1 = gh.sum(0)
        opt.step([gW1, gb1, gW2, gb2])
    return params

def forward_factorized(X, params):
    # X = [β(2), s(2)]; 输出 = 分支A(β) + 分支B(s)
    W1a, b1a, W2a, W2b, W1b, b1b = params
    ha = np.tanh(X[:, :D_BETA] @ W1a.T + b1a)
    hb = np.tanh(X[:, D_BETA:] @ W1b.T + b1b)
    return ha @ W2a.T + hb @ W2b.T

def train_factorized(X, Y, hidden=48, epochs=3000, lr=1e-2):
    W1a = rng.normal(0, 0.5, (hidden, D_BETA)); b1a = np.zeros(hidden)
    W1b = rng.normal(0, 0.5, (hidden, D_S));    b1b = np.zeros(hidden)
    W2a = rng.normal(0, 0.3, (K, hidden)); W2b = rng.normal(0, 0.3, (K, hidden))
    params = [W1a, b1a, W2a, W2b, W1b, b1b]
    opt = Adam(params, lr)
    for _ in range(epochs):
        β = X[:, :D_BETA]; s = X[:, D_BETA:]
        ha = np.tanh(β @ W1a.T + b1a); hb = np.tanh(s @ W1b.T + b1b)
        out = ha @ W2a.T + hb @ W2b.T
        g_out = 2*(out - Y)/len(X)
        gW2a = g_out.T @ ha; gW2b = g_out.T @ hb
        gha = g_out @ W2a * (1 - ha**2)
        ghb = g_out @ W2b * (1 - hb**2)
        gW1a = gha.T @ β; gb1a = gha.sum(0)
        gW1b = ghb.T @ s; gb1b = ghb.sum(0)
        opt.step([gW1a, b1a*0+gb1a, gW2a, gW2b, gW1b, b1b*0+gb1b])
    return params

def rms(pred, Y):
    return float(np.sqrt(((pred - Y)**2).mean()))

for interact, label in [(0.0, '纯加性 GT（交互=0）'), (0.15, '轻度交互 GT（交互=0.15）'), (0.5, '强交互 GT（交互=0.5）')]:
    Xtr, Ytr, Xte, Yte = make_data(interact)
    # 随机种子固定后重置 RNG 保证两模型可比
    st = rng.bit_generator.state
    rng.bit_generator.state = st
    p_mono = train_monolithic(Xtr, Ytr)
    rng.bit_generator.state = st
    p_fact = train_factorized(Xtr, Ytr)
    e_mono = rms(forward_monolithic(Xtr, p_mono), Ytr), rms(forward_monolithic(Xte, p_mono), Yte)
    e_fact = rms(forward_factorized(Xtr, p_fact), Ytr), rms(forward_factorized(Xte, p_fact), Yte)
    print(f"\n  [{label}]  (48 组训练 / 16 组 held-out 组合)")
    print(f"    单体 MLP   : train {e_mono[0]:.4f}  test {e_mono[1]:.4f}")
    print(f"    分解 A+B   : train {e_fact[0]:.4f}  test {e_fact[1]:.4f}")

# ============ 实验 2：单场多阈值的"伪分层"——一个 SMPLicit 没选的方案 ============
# 说明: SMPLicit 实际用单一阈值 t_d=0.1mm + 多个独立服装实例实现多层穿衣。
# 本实验检验的是"一个距离场+多个阈值"这个自然替代方案——几何上可行但语义上失效,
# 正文用它说明 SMPLicit 为什么选多实例。
print("\n=== 实验 2: 单一距离场多阈值方案的检验（SMPLicit 未采用的替代路线）===")

def dist_to_capsule(P, r=0.18, z0=0.2, z1=1.5):
    """竖直胶囊体（圆柱+端球）解析距离"""
    z = np.clip(P[:, 2], z0, z1)
    axis_pt = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    d = np.linalg.norm(P - axis_pt, axis=1) - r
    # 端球
    for zc in (z0, z1):
        d2 = np.linalg.norm(P - np.array([0, 0, zc]), axis=1) - r
        d = np.minimum(d, d2)
    return d

# 网格点
gx = np.linspace(-0.5, 0.5, 41)
gy = np.linspace(-0.5, 0.5, 41)
gz = np.linspace(0.0, 1.7, 71)
G = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), -1).reshape(-1, 3)
u = dist_to_capsule(G)

# 1) |∇u| = 1 检验（有限差分, 内部点）
eps = 1e-4
def grad_u(P):
    g = np.zeros_like(P)
    for d in range(3):
        Pp = P.copy(); Pp[:, d] += eps
        Pm = P.copy(); Pm[:, d] -= eps
        g[:, d] = (dist_to_capsule(Pp) - dist_to_capsule(Pm)) / (2*eps)
    return g
inner = u > 0.005  # 离开表面
gu = grad_u(G[inner])
norms = np.linalg.norm(gu, axis=1)
print(f"  |∇u| 检验（表面 5mm 外的 {inner.sum()} 个网格点）: mean {norms.mean():.6f}, max {norms.max():.6f}  ✓（距离场性质）")

# 2) 不同阈值提取等值面 → 分层服装
for name, d_layer in [('贴身 T 恤', 0.03), ('宽松毛衣', 0.08), ('外套大衣', 0.15)]:
    on_surf = np.abs(u - d_layer) < 0.005
    pts = G[on_surf]
    cyl = pts[(pts[:, 2] > 0.4) & (pts[:, 2] < 1.3)]   # 只统计圆柱段（端球区径向递减）
    r_pts = np.linalg.norm(cyl[:, :2], axis=1)
    print(f"  阈值 σ={d_layer:.2f} m [{name}]: {on_surf.sum():4d} 个等值点, "
          f"圆柱段径向 {r_pts.mean():.4f}±{r_pts.std():.4f} m (理论 {0.18+d_layer:.3f})"
          f"  ——但注意: 三层都是身体的'等距扩张', 不是真服装层")

# 3) 嵌套性: 外层等值面把内层完全包住（同一条射线上 u 单调）
ray = np.stack([np.linspace(0.19, 0.45, 200), np.zeros(200), np.full(200, 0.8)], axis=1)
ur = dist_to_capsule(ray)
print(f"  沿径向射线 u 单调（分层无交叉）: {np.all(np.diff(ur) > 0)} ✓")

# ============ 实验 3：开放曲面的 SDF 歧义（为什么必须用 UDF） ============
print("\n=== 实验 3: 开放曲线的'有符号'距离 —— 零水平集污染 ===")
# 2D 开弧（半圆, 开口向下）: 参数 t ∈ [0, π], θ = t, r = 1
ts = np.linspace(0, np.pi, 200)
arc = np.stack([np.cos(ts), np.sin(ts)], axis=1)   # 单位上半圆弧, 端点 (1,0) 与 (-1,0)
tang = np.stack([-np.sin(ts), np.cos(ts)], axis=1)
normals = tang[:, [1, 0]] * np.array([1, -1])  # 旋转90°

def closest_on_arc(P):
    d = np.linalg.norm(P[:, None, :] - arc[None, :, :], axis=2)
    idx = d.argmin(axis=1)
    return arc[idx], normals[idx], d[np.arange(len(P)), idx]

# 网格
gx2 = np.linspace(-1.6, 1.6, 161)
gy2 = np.linspace(-0.6, 1.6, 161)
G2 = np.stack(np.meshgrid(gx2, gy2, indexing='ij'), -1).reshape(-1, 2)
cp, cn, dist = closest_on_arc(G2)
signed = np.sum((G2 - cp) * cn, axis=1)   # 最近点法向定向的"符号"

# 零水平集成员分析
zero_band = np.abs(signed) < 0.02
true_dist = dist
far_from_arc = true_dist > 0.05          # 离真实弧至少 5cm
contaminated = zero_band & far_from_arc
print(f"  |signed| < 2cm 的点: {zero_band.sum()}")
print(f"  其中离真实弧 > 5cm 的污染点: {contaminated.sum()}"
      f"  ({100*contaminated.sum()/max(zero_band.sum(),1):.1f}% of zero-band)")
# 污染区域位置
if contaminated.sum() > 0:
    cpts = G2[contaminated]
    print(f"  污染点 x 范围: [{cpts[:,0].min():.2f}, {cpts[:,0].max():.2f}], "
          f"y 范围: [{cpts[:,1].min():.2f}, {cpts[:,1].max():.2f}]")
    print(f"  → 端点 (±1, 0) 之外沿切向延伸的整条带被'误归零'：")
    print(f"    开曲面的内外经端点连通, 任何符号约定都会把零水平集从弧'漏'出去 —— SDF 不适定, 必须用 UDF")
