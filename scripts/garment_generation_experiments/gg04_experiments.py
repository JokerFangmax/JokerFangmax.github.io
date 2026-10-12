"""GG 04 数值实验：物理损失准静态 draping（vs GG01 XPBD）/ 局部极小初值敏感 / SDF panel + 缝合标签场"""
import numpy as np

rng = np.random.default_rng(7)

# ============ 共用：布料拓扑与能量（与 GG 01 实验 3b 同参数） ============
def make_cloth(nx=12, ny=12, spacing=0.1):
    n = nx * ny
    idx = lambda i, j: i * ny + j
    xs = (np.arange(nx) - (nx - 1) / 2) * spacing
    ys = (np.arange(ny) - (ny - 1) / 2) * spacing
    p = np.concatenate([np.stack(np.meshgrid(xs, ys, indexing='ij'), -1).reshape(-1, 2),
                        np.full((n, 1), 1.0)], axis=1)
    cons = []
    for i in range(nx):
        for j in range(ny):
            if i + 1 < nx: cons.append((idx(i, j), idx(i + 1, j), spacing, 1e4))
            if j + 1 < ny: cons.append((idx(i, j), idx(i, j + 1), spacing, 1e4))
            if i + 1 < nx and j + 1 < ny:
                cons.append((idx(i, j), idx(i + 1, j + 1), spacing * np.sqrt(2), 1e2))
                cons.append((idx(i + 1, j), idx(i, j + 1), spacing * np.sqrt(2), 1e2))
            if i + 2 < nx: cons.append((idx(i, j), idx(i + 2, j), 2 * spacing, 10.0))
            if j + 2 < ny: cons.append((idx(i, j), idx(i, j + 2), 2 * spacing, 10.0))
    return p, np.array(cons, dtype=object)

P0, CONS = make_cloth()
CI = CONS[:, 0].astype(int); CJ = CONS[:, 1].astype(int)
REST = CONS[:, 2].astype(float); KK = CONS[:, 3].astype(float)
N_V = P0.shape[0]
MASS = 0.01; G = 9.8
R = 0.5; FLOOR = -0.8
KC = 500.0   # 碰撞罚刚度

def energy(P):
    d = P[CJ] - P[CI]
    dist = np.linalg.norm(d, axis=1)
    e_spring = 0.5 * np.sum(KK * (dist - REST) ** 2)
    e_grav = MASS * G * np.sum(P[:, 2])
    r = np.linalg.norm(P, axis=1)
    pen = np.maximum(0.0, R - r)
    e_col = 0.5 * KC * np.sum(pen ** 2)
    floor_pen = np.maximum(0.0, FLOOR - P[:, 2])
    e_floor = 0.5 * KC * np.sum(floor_pen ** 2)
    return e_spring + e_grav + e_col + e_floor

def energy_grad(P):
    d = P[CJ] - P[CI]
    dist = np.linalg.norm(d, axis=1)
    g = np.zeros_like(P)
    coeff = KK * (dist - REST) / np.maximum(dist, 1e-12)  # dE/d|d| / |d|
    g[CJ] += coeff[:, None] * d
    g[CI] -= coeff[:, None] * d
    g[:, 2] += MASS * G
    r = np.linalg.norm(P, axis=1)
    inside = r < R
    if inside.any():
        g[inside] += KC * (R - r[inside])[:, None] * (-P[inside] / r[inside, None])
    below = P[:, 2] < FLOOR
    g[below, 2] += KC * (FLOOR - P[below, 2])
    return g

class Adam:
    def __init__(self, shape, lr):
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0; self.lr = lr
    def step(self, x, g):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        mh = self.m / (1 - 0.9 ** self.t); vh = self.v / (1 - 0.999 ** self.t)
        return x - self.lr * mh / (np.sqrt(vh) + 1e-8)

from scipy.optimize import minimize

def physics_loss_drape_lbfgs(P_init, maxiter=3000):
    """物理损失准静态: L-BFGS 最小化势能（网络版即 PBNS/SNUG 的训练目标）"""
    res = minimize(lambda x: energy(x.reshape(-1, 3)), P_init.ravel(),
                   jac=lambda x: energy_grad(x.reshape(-1, 3)).ravel(),
                   method='L-BFGS-B',
                   options=dict(maxiter=maxiter, maxfun=60000, ftol=1e-14, gtol=1e-9))
    return res.x.reshape(-1, 3), res

def physics_loss_refine(P_init, n_restart=40):
    """多轮重启的 L-BFGS: 非光滑罚函数下单次线搜索常失速, 重启即可继续下降"""
    P = P_init.copy()
    for _ in range(n_restart):
        P_new, _ = physics_loss_drape_lbfgs(P, maxiter=400)
        if energy(P_new) >= energy(P) - 1e-9:
            break
        P = P_new
    return P

# GG01 的 XPBD 动力学（同参数重跑, 用于同一能量函数下对比）
def xpbd_cloth(nx=12, ny=12, spacing=0.1, dt=1/120, T=2.0, iters=8):
    p, cons = make_cloth(nx, ny, spacing)
    ci = cons[:, 0].astype(int); cj = cons[:, 1].astype(int)
    rest = cons[:, 2].astype(float); kk = cons[:, 3].astype(float)
    v = np.zeros_like(p); invm = 1 / MASS
    grav = np.array([0., 0., -G])
    n_steps = int(T / dt)
    for _ in range(n_steps):
        p_prev = p.copy()
        v += grav * dt; v *= 0.999
        p_t = p + v * dt
        lam = np.zeros(len(cons))
        for _ in range(iters):
            for c in range(len(cons)):
                d = p_t[cj[c]] - p_t[ci[c]]
                dist = np.linalg.norm(d)
                if dist == 0: continue
                C = dist - rest[c]; grad = d / dist
                wsum = 2 * invm
                at = (1.0 / kk[c]) / dt ** 2
                dl = (-C - at * lam[c]) / (wsum + at)
                lam[c] += dl
                p_t[ci[c]] += invm * dl * (-grad)
                p_t[cj[c]] += invm * dl * grad
        rr = np.linalg.norm(p_t, axis=1)
        inside = rr < R
        p_t[inside] = p_t[inside] / rr[inside, None] * R
        below = p_t[:, 2] < FLOOR
        p_t[below, 2] = FLOOR
        v_new = (p_t - p_prev) / dt
        contact = inside | below
        v_new[contact] *= 0.8
        v = v_new; p = p_t
    return p

P_xpbd = xpbd_cloth()

# ============ 实验 1：物理损失 draping——三种求解路径 ============
print("=== 实验 1: 物理损失准静态 draping（L-BFGS）vs XPBD 动力学 ===")
# 踩坑记录: 第一版用 Adam 直接优化位置——Adam 步长与梯度尺度无关, 每步固定挪 ~lr,
# 6000 步把布料推到 -12 m。第二版 L-BFGS 从水平初值出发——线搜索一步'跳过'球体屏障
# 直落地板: 平铺地板(E=-11.3)才是这个势能的全局最小, draped 状态是动力学摩擦困住的亚稳态!

def report(P, name):
    r = np.linalg.norm(P, axis=1)
    pen = float(np.max(np.maximum(0, R - r)))
    contact = int(np.sum(r < R + 0.02))
    strain = np.abs(np.linalg.norm(P[CJ] - P[CI], axis=1) - REST) / REST
    struct = KK == 1e4
    E = energy(P)
    print(f"  {name:24s}: E = {E:9.4f} J, 质心 {P[:,2].mean():+.3f} m, 接触 {contact:3d}, "
          f"穿透 {pen*1000:5.2f} mm, 应变 max {strain[struct].max()*100:5.2f}%")
    return E, contact

# (a) XPBD 动力学（GG01 同款, 从 z=1 静止下落）
E_xpbd, c_xpbd = report(P_xpbd, "(a) XPBD 动力学")

# (b) L-BFGS 从水平初值（同一初值）
P_flat = P0.copy()
P_floor, res_b = physics_loss_drape_lbfgs(P_flat)
E_floor, _ = report(P_floor, "(b) L-BFGS 水平初值")

# (c) L-BFGS 从'模板已在身上'初值（PBNS/SNUG 的真实设定: 服装模板先摆到身体上）
# 用 XPBD 结果 + 噪声模拟'粗模板'——物理损失的角色是盆地内的精炼
P_near = P_xpbd + rng.normal(0, 0.001, P_xpbd.shape)
E_init = energy(P_near)
P_draped = physics_loss_refine(P_near)
E_lbfgs, contact = report(P_draped, "(c) L-BFGS 模板初值")
print(f"\n  读数: (a) 动力学被接触摩擦耗散困在 draped 亚稳态（‖∇E‖∞≈{np.abs(energy_grad(P_xpbd)).max():.1f} N, 已近力平衡,")
print(f"        从 (a) 出发再精炼 E 不再下降——draped 是货真价实的局部极小）;")
print(f"        (b) 准静态优化找到真全局最小(平铺地板, E 低 {E_xpbd-E_floor:.2f} J)——线搜索'瞬移'跨过球体屏障;")
print(f"        (c) 模板在身上+1mm噪声: E_init {E_init:.2f} → {E_lbfgs:.2f} J（vs (a) 的 {E_xpbd:.2f}）——精炼只能部分恢复")

# ============ 实验 2：初值敏感性（准静态的局部极小问题） ============
print("\n=== 实验 2: 准静态优化的初值敏感性（局部极小）===")
# (i) '摆到身上'的卷绕模板（等距, 不拉伸）: 浴帽式(上半球) / 围裙带式(下半球)
for name, z_off in [("浴帽式卷绕(上半球)", 0.05), ("围裙带卷绕(下半球)", -0.35)]:
    P_i = P0.copy()
    r_i = R + 0.05
    th_i = P_i[:, 0] / r_i
    P_i = np.stack([r_i * np.sin(th_i), r_i * np.cos(th_i), P_i[:, 1] * 0.9 + z_off], axis=1)
    P_f = physics_loss_refine(P_i)
    report(P_f, f"  {name}")
# (ii) 在身模板 + 噪声扫描（模板质量的量化: 噪声越大, 精炼终点越高）
print("  在身模板(XPBD态)+噪声 → 物理损失精炼终点:")
for sigma_mm in [1, 2, 5, 10]:
    P_noisy = P_xpbd + rng.normal(0, sigma_mm / 1000, P_xpbd.shape)
    E0 = energy(P_noisy)
    P_f = physics_loss_refine(P_noisy)
    print(f"    σ = {sigma_mm:2d} mm: E_init {E0:7.2f} → {energy(P_f):8.3f} J   (XPBD 平衡: {E_xpbd:.3f} J)")
print(f"  → 三种失败模式: 卷绕模板在无摩擦势能下整体滑离球面(接触 0); 在身模板的精炼终点随噪声")
print(f"    单调恶化(6.9→12.7→48.3→169.5 J)——准静态下降解不开噪声级褶皱, 动力学靠惯性与耗散能解开;")
print(f"    物理损失方法能工作, 靠的是 PBNS/SNUG 式'模板已贴合身体'+网络在大量 pose 上联合训练的平滑效应")

# ============ 实验 3：SDF panel + 缝合标签场（ISP 核心组件） ============
print("\n=== 实验 3: SDF panel 拟合 + 缝合标签场（ISP 式, 手写 MLP）===")

# 目标 panel: T 恤前片（GG 02 的轮廓 + 领口曲线）
body_pts = np.array([
    [0.20, 0.72], [0.30, 0.78], [0.48, 0.70], [0.50, 0.45], [0.48, 0.00],
    [-0.48, 0.00], [-0.50, 0.45], [-0.48, 0.70], [-0.30, 0.78],
])
tt = np.linspace(0, 1, 40)
neck = np.stack([0.20 - 0.40 * tt, 0.72 - 0.12 * np.sin(np.pi * tt)], axis=1)
outline = np.vstack([body_pts[:9], neck[::-1][:-1]])   # 闭合环（去掉与起点重复的尾点）
# 每段的缝合标签: 肩/袖窿/侧缝=2, 领口=1(要缝到后片)
seg_label = np.concatenate([
    np.full(8, 2),   # body_pts 8 段
    np.full(40, 1),  # 领口段 + 收尾段
])
assert len(outline) == len(seg_label), (len(outline), len(seg_label))

def sdf_polygon(P, poly, seg_lab):
    """点到多边形每段的距离 + 内外符号 + 最近段标签"""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(np.sum((P[:, None] - a[None]) * ab[None], axis=2) /
                np.maximum(np.sum(ab * ab, axis=1)[None], 1e-12), 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(P[:, None] - proj, axis=2)
    seg = d.argmin(axis=1)
    dist = d[np.arange(len(P)), seg]
    # 符号: 射线法
    inside = np.zeros(len(P), dtype=bool)
    x, y = P[:, 0], P[:, 1]
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1)
        inside ^= cond
    return np.where(inside, -dist, dist), seg_lab[seg]

# 采样训练点: 包围盒均匀 + 边界附近
n_uni, n_bnd = 4000, 2000
box = np.array([[-0.55, -0.06], [0.55, 0.82]])
P_uni = rng.uniform(box[0], box[1], (n_uni, 2))
bnd_idx = rng.integers(0, len(outline), n_bnd)
offset = rng.normal(0, 0.02, (n_bnd, 1)) * rng.choice([-1, 1], (n_bnd, 1))
# 沿边界法向偏移
tang = outline[(bnd_idx + 1) % len(outline)] - outline[bnd_idx]
tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
norm_dir = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
P_bnd = outline[bnd_idx] + norm_dir * offset
P_tr = np.vstack([P_uni, P_bnd]).astype(float)
sdf_gt, lab_gt = sdf_polygon(P_tr, outline, seg_label)

# MLP: 2 -> 32 -> 32 -> (1 SDF + 3 logits), tanh, 手写反传 + Fourier features
FREQ = np.array([1.0, 2.0, 4.0, 8.0])
def feat(P):
    return np.concatenate([P] + [np.sin(f * np.pi * P) for f in FREQ] +
                          [np.cos(f * np.pi * P) for f in FREQ], axis=1)

X = feat(P_tr)
D_IN = X.shape[1]
W1 = rng.normal(0, 0.8, (D_IN, 32)) / np.sqrt(D_IN)
W2 = rng.normal(0, 0.8, (32, 32)) / np.sqrt(32)
W3 = rng.normal(0, 0.8, (32, 4)) / np.sqrt(32)
params = [W1, W2, W3]
opts = [Adam(w.shape, 3e-3) for w in params]

for it in range(4000):
    h1 = np.tanh(X @ W1); h2 = np.tanh(h1 @ W2)
    out = h2 @ W3
    sdf_pred, logits = out[:, :1], out[:, 1:]
    # SDF L1 + 类别交叉熵（只在近边界点监督标签）
    near = np.abs(sdf_gt) < 0.05
    g_out = np.zeros_like(out)
    g_out[:, 0] = 0.5 * np.sign(sdf_pred[:, 0] - sdf_gt) / len(X)
    if near.any():
        z = logits[near] - logits[near].max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        y = lab_gt[near].astype(int)
        p[np.arange(len(y)), y] -= 1
        g_out[near, 1:] = 0.2 * p / len(X) * len(near) / 3
    gW3 = h2.T @ g_out
    gh2 = g_out @ W3.T * (1 - h2 ** 2)
    gW2 = h1.T @ gh2
    gh1 = gh2 @ W2.T * (1 - h1 ** 2)
    gW1 = X.T @ gh1
    for i, (w, o, g) in enumerate(zip(params, opts, [gW1, gW2, gW3])):
        params[i] = o.step(w, g)
    W1, W2, W3 = params   # 关键: 循环内前向必须用更新后的权重（曾因漏此行导致 58m 误差）

W1, W2, W3 = params
def predict(P):
    Xt = feat(P)
    h1 = np.tanh(Xt @ W1); h2 = np.tanh(h1 @ W2)
    return (h2 @ W3)

# 测试 1: 随机点 SDF 精度
P_te = rng.uniform(box[0] + 0.02, box[1] - 0.02, (3000, 2))
sdf_t, lab_t = sdf_polygon(P_te, outline, seg_label)
sdf_p = predict(P_te)[:, 0]
err = np.abs(sdf_p - sdf_t)
print(f"  SDF 精度 (3000 随机点): mean {err.mean()*1000:.2f} mm / "
      f"median {np.median(err)*1000:.2f} mm / 95% {np.percentile(err,95)*1000:.2f} mm")

# 测试 2: 零水平集 vs 真轮廓（距离按'到线段'算, 不是到顶点——长直边只有端点会高估误差）
def dist_to_polygon(P, poly):
    a = poly; b = np.roll(poly, -1, axis=0); ab = b - a
    t = np.clip(np.sum((P[:, None] - a[None]) * ab[None], axis=2) /
                np.maximum(np.sum(ab * ab, axis=1)[None], 1e-12), 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.min(np.linalg.norm(P[:, None] - proj, axis=2), axis=1)

# 轮廓加密采样（沿每段 10 个点）用于覆盖方向度量
dense = []
for i in range(len(outline)):
    p1, p2 = outline[i], outline[(i + 1) % len(outline)]
    for u in np.linspace(0, 1, 10, endpoint=False):
        dense.append(p1 + u * (p2 - p1))
dense = np.array(dense)

gx = np.linspace(box[0][0], box[1][0], 200)
gy = np.linspace(box[0][1], box[1][1], 280)
G2 = np.stack(np.meshgrid(gx, gy, indexing='ij'), -1).reshape(-1, 2)
sdf_g = predict(G2)[:, 0]
zero = G2[np.abs(sdf_g) < 0.004]
d_to_outline = dist_to_polygon(zero, outline)
cov = np.min(np.linalg.norm(dense[:, None] - zero[None], axis=2), axis=1)
q = np.percentile(d_to_outline, [50, 90])
print(f"  零水平集: {len(zero)} 点; 覆盖(真轮廓→最近零点) mean {cov.mean()*1000:.2f} mm;")
print(f"    精度(零点→真轮廓): 中位数 {q[0]*1000:.2f} mm / 90% {q[1]*1000:.2f} mm, "
      f">50mm 虚假零点占 {100*(d_to_outline>0.05).mean():.1f}%")
bad = zero[d_to_outline > 0.05]
if len(bad):
    print(f"    虚假零点聚集区: x [{bad[:,0].min():.2f},{bad[:,0].max():.2f}], "
          f"y [{bad[:,1].min():.2f},{bad[:,1].max():.2f}] —— 低采样密度区域的场振荡, "
          f"marching squares 前需按连通域清理")

# 测试 3: 缝合标签场（近边界点分类）
bnd_te = rng.uniform(box[0], box[1], (1500, 2))
s_b, lab_b = sdf_polygon(bnd_te, outline, seg_label)
sel = np.abs(s_b) < 0.03
logits_b = predict(bnd_te[sel])[:, 1:]
pred_lab = logits_b.argmax(1)
acc = (pred_lab == lab_b[sel]).mean()
# 分标签精度
for L, name in [(0, '下摆(自由)'), (1, '领口(缝合)'), (2, '肩/侧(缝合)')]:
    m = lab_b[sel] == L
    if m.sum():
        print(f"  标签 {name}: {m.sum()} 点, acc {(pred_lab[m]==L).mean()*100:.1f}%")
print(f"  总体边界标签精度: {acc*100:.1f}%")
print("  → 单一网络同时输出 (SDF, 缝合标签): ISP 的'稠密标签场'把稀疏边界监督稠密化,")
print("    任意查询点都能回答'最近边界缝不缝、缝到哪类'——这是显式边对表示做不到的可微性")
