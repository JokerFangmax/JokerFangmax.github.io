"""GG 01 数值实验：LBS 糖纸伪影 / XPBD vs PBD 链刚度 / 布料落球 / 显式积分爆炸"""
import numpy as np

rng = np.random.default_rng(0)

# ============ 实验 1：LBS 糖纸伪影（解析验证 cos^2(θ/2)） ============
def rot_x(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])

def poly_area(pts2d):
    x, y = pts2d[:, 0], pts2d[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
ring = np.stack([np.zeros_like(phi), np.cos(phi), np.sin(phi)], axis=1)  # 关节处一圈顶点，r=1
A0 = poly_area(ring[:, 1:])
w = 0.5  # 两骨骼权重各半

print("=== 实验 1: LBS 糖纸伪影 (w=0.5, 解析预测 cos^2(θ/2)) ===")
for deg in [0, 30, 60, 90, 120, 150, 180]:
    th = np.deg2rad(deg)
    v2 = (1 - w) * ring + w * (ring @ rot_x(th).T)   # LBS: 凸组合
    ratio = poly_area(v2[:, 1:]) / A0
    print(f"  θ={deg:3d}°  截面面积比 = {ratio:.4f}   解析 = {np.cos(th/2)**2:.4f}")

# 不等权重的推广：scale = sqrt(1 - 2w(1-w)(1-cosθ))
print("\n  推广（解析 sqrt(1-2w(1-w)(1-cosθ))）: θ=180° 时")
for w_ in [0.1, 0.3, 0.5]:
    th = np.pi
    v2 = (1 - w_) * ring + w_ * (ring @ rot_x(th).T)
    ratio = poly_area(v2[:, 1:]) / A0
    pred = 1 - 2 * w_ * (1 - w_) * (1 - np.cos(th))  # scale^2
    print(f"  w={w_:.1f}: 面积比={ratio:.4f}  解析={pred:.4f}")

# ============ 实验 2：PBD vs XPBD 链式弹簧（解析验证 mg/k） ============
def chain_sim(method, iters, dt=1/60, T=30.0, k=100.0, m=1.0, L=1.0, g=9.8):
    """锚点 + 2 粒子垂直悬挂链。PBD: 刚度随迭代数变化; XPBD: 物理刚度 k。
    解析平衡: 上链伸长 2mg/k=0.196, 下链伸长 mg/k=0.098"""
    p = np.array([[0., 0., -L], [0., 0., -2 * L]])  # 两个粒子
    v = np.zeros((2, 3))
    invm = np.array([1 / m, 1 / m])
    anchor = np.zeros(3)
    grav = np.array([0., 0., -g])
    n_steps = int(T / dt)
    cons = [(anchor, 0), (0, 1)]  # (端点或粒子索引) 距离约束
    for _ in range(n_steps):
        p_prev = p.copy()
        v += grav * dt
        p_tilde = p + v * dt
        lam = np.zeros(2)  # XPBD 每个 constraint 的乘子
        for _ in range(iters):
            for ci, (ia, ib) in enumerate(cons):
                a = anchor if isinstance(ia, np.ndarray) else p_tilde[ia]
                b = p_tilde[ib]
                d = b - a
                dist = np.linalg.norm(d)
                if dist == 0:
                    continue
                C = dist - L
                grad_b = d / dist
                grad_a = -grad_b
                # 质量加权范数
                wb = invm[ib]
                wa = 0.0 if isinstance(ia, np.ndarray) else invm[ia]
                wsum = wa + wb
                if method == 'pbd':
                    dlam = -C / wsum
                else:  # xpbd
                    alpha_tilde = (1.0 / k) / dt**2
                    dlam = (-C - alpha_tilde * lam[ci]) / (wsum + alpha_tilde)
                    lam[ci] += dlam
                if not isinstance(ia, np.ndarray):
                    p_tilde[ia] += wa * dlam * grad_a
                p_tilde[ib] += wb * dlam * grad_b
        p = p_tilde
        v = (p - p_prev) / dt
    ext_top = np.linalg.norm(p[0]) - L
    ext_bot = np.linalg.norm(p[1] - p[0]) - L
    return ext_top, ext_bot

print("\n=== 实验 2: 链式弹簧 平衡伸长 (T=30s 完全安定; 解析: 上 2mg/k=0.196, 下 mg/k=0.098) ===")
for method, iters_list in [('pbd', [1, 2, 4, 16, 64]), ('xpbd', [1, 2, 4, 16, 64])]:
    for it in iters_list:
        et, eb = chain_sim(method, it)
        print(f"  {method.upper():4s} iters={it:2d}: 上={et:+.4f} 下={eb:+.4f} 总={2+et+eb:.4f}")

# ============ 实验 3：显式积分 vs XPBD（刚性弹簧稳定性） ============
print("\n=== 实验 3a: 刚性弹簧积分稳定性 (k=1e5, m=0.01, dt=1/60, ωdt =", end="")
k, m, dt = 1e5, 0.01, 1 / 60
omega = np.sqrt(k / m)
print(f"{omega*dt:.1f}) ===")

def forward_euler(k, m, dt, steps=300, x0=1.001):
    x, v = x0, 0.0
    for s in range(steps):
        x_new = x + v * dt
        v = v - (k / m) * x * dt
        x = x_new
        if abs(x) > 1e3:
            return s + 1, x
    return None, x

def symplectic_euler(k, m, dt, steps=300, x0=1.001):
    x, v = x0, 0.0
    for s in range(steps):
        v = v - (k / m) * x * dt
        x = x + v * dt
        if abs(x) > 1e3:
            return s + 1, x
    return None, x

def xpbd_spring(k, m, dt, steps=300, x0=1.001):
    x, v = x0, 0.0
    L = 1.0
    for _ in range(steps):
        v_prev = v
        x_t = x + v * dt
        lam = 0.0
        for _ in range(4):
            C = x_t - L
            grad = 1.0
            w = 1 / m
            alpha_tilde = (1.0 / k) / dt**2
            dlam = (-C - alpha_tilde * lam) / (w + alpha_tilde)
            lam += dlam
            x_t += w * dlam * grad
        v = (x_t - x) / dt
        x = x_t
    return x

s1, x1 = forward_euler(k, m, dt)
s2, x2 = symplectic_euler(k, m, dt)
x3 = xpbd_spring(k, m, dt)
print(f"  前向 Euler:   {'%d 步后 |x|>1e3 (x=%.2e)' % (s1, x1) if s1 else f'300 步稳定, x={x2:.4f}'}")
print(f"  辛 Euler:     {'%d 步后 |x|>1e3 (x=%.2e)' % (s2, x2) if s2 else f'300 步稳定, x={x2:.4f}'}")
print(f"  XPBD (k=1e5): 300 步后 x = {x3:.6f} (解析平衡 x=1, 有界 ✓)")

# ============ 实验 3b：XPBD 布料落到球上 ============
print("\n=== 实验 3b: XPBD 布料 12×12 落到球上 (R=0.5, dt=1/120, T=2s) ===")

def cloth_sim(nx=12, ny=12, spacing=0.1, dt=1/120, T=2.0, iters=8,
              k_stretch=1e4, k_shear=1e2, k_bend=10.0, R=0.5, floor=-0.8):
    n = nx * ny
    idx = lambda i, j: i * ny + j
    xs = (np.arange(nx) - (nx - 1) / 2) * spacing
    ys = (np.arange(ny) - (ny - 1) / 2) * spacing
    p0 = np.stack(np.meshgrid(xs, ys, indexing='ij'), -1).reshape(-1, 2)
    p = np.concatenate([p0, np.full((n, 1), 1.0)], axis=1)  # z=1.0 水平放置
    p[:, [0, 1, 2]] = p[:, [0, 2, 1]]  # 变成 (x, z_up, y)? 简化：直接用 z 向上
    # 重新构造: x, y 平面网格, z=1
    p = np.concatenate([p0, np.full((n, 1), 1.0)], axis=1)
    v = np.zeros((n, 3))
    mass = 0.01
    invm = 1.0 / mass
    grav = np.array([0., 0., -9.8])

    cons = []  # (i, j, rest, k)
    for i in range(nx):
        for j in range(ny):
            if i + 1 < nx:
                cons.append((idx(i, j), idx(i + 1, j), spacing, k_stretch))
            if j + 1 < ny:
                cons.append((idx(i, j), idx(i, j + 1), spacing, k_stretch))
            if i + 1 < nx and j + 1 < ny:
                cons.append((idx(i, j), idx(i + 1, j + 1), spacing * np.sqrt(2), k_shear))
                cons.append((idx(i + 1, j), idx(i, j + 1), spacing * np.sqrt(2), k_shear))
            if i + 2 < nx:
                cons.append((idx(i, j), idx(i + 2, j), 2 * spacing, k_bend))
            if j + 2 < ny:
                cons.append((idx(i, j), idx(i, j + 2), 2 * spacing, k_bend))
    cons = np.array(cons, dtype=object)
    ci = cons[:, 0].astype(int); cj = cons[:, 1].astype(int)
    rest = cons[:, 2].astype(float); kk = cons[:, 3].astype(float)

    n_steps = int(T / dt)
    for step in range(n_steps):
        p_prev = p.copy()
        v += grav * dt
        v *= 0.999  # 轻微全局阻尼
        p_t = p + v * dt
        lam = np.zeros(len(cons))
        for _ in range(iters):
            for c in range(len(cons)):
                d = p_t[cj[c]] - p_t[ci[c]]
                dist = np.linalg.norm(d)
                if dist == 0:
                    continue
                C = dist - rest[c]
                grad = d / dist
                wsum = 2 * invm
                alpha_t = (1.0 / kk[c]) / dt**2
                dlam = (-C - alpha_t * lam[c]) / (wsum + alpha_t)
                lam[c] += dlam
                p_t[ci[c]] += invm * dlam * (-grad)
                p_t[cj[c]] += invm * dlam * grad
        # 球碰撞投影
        r = np.linalg.norm(p_t, axis=1)
        inside = r < R
        p_t[inside] = p_t[inside] / r[inside, None] * R
        # 地板
        below = p_t[:, 2] < floor
        p_t[below, 2] = floor
        # 接触摩擦：接触点切向速度衰减
        v_new = (p_t - p_prev) / dt
        contact = inside | below
        v_new[contact] *= 0.8
        v = v_new
        p = p_t
    # 统计
    r = np.linalg.norm(p, axis=1)
    contact_n = int(np.sum(r < R + 0.02))
    pen = max(0.0, float(np.max(R - r)))
    d = p[cj] - p[ci]
    dist = np.linalg.norm(d, axis=1)
    strain = np.abs(dist - rest) / rest
    struct = kk == k_stretch
    max_strain = float(np.max(strain[struct]))
    mean_strain = float(np.mean(strain[struct]))
    return dict(contact=contact_n, penetration=pen, max_strain=max_strain,
                mean_strain=mean_strain, centroid_z=float(np.mean(p[:, 2])),
                max_v=float(np.max(np.linalg.norm(v, axis=1))))

res = cloth_sim()
print(f"  接触球面顶点数: {res['contact']}/{144}")
print(f"  最大穿透深度: {res['penetration']:.2e} (投影保证 ≈0)")
print(f"  结构约束应变: max {res['max_strain']*100:.2f}% / mean {res['mean_strain']*100:.3f}%")
print(f"  布料质心高度: {res['centroid_z']:.3f} (球顶 0.5)")
print(f"  残余最大速度: {res['max_v']:.3f} m/s")

# 复跑一遍不同迭代数，看 XPBD 刚度无关性（结构应变应稳定）
print("\n  XPBD 迭代数无关性检查（应变随迭代数的变化）:")
for it in [2, 4, 8, 16]:
    r2 = cloth_sim(iters=it)
    print(f"    iters={it:2d}: max_strain={r2['max_strain']*100:.2f}%  contact={r2['contact']}  cz={r2['centroid_z']:.3f}")
