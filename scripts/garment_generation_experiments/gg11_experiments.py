"""GG 11 数值实验：无版型路线 vs 版型路线——机理对照（numpy/scipy，无学习框架）
1) 可展性：版型↔3D 的桥只在 Gaussian 曲率 K=0 时免费
   (a) 多面体 Gauss–Bonnet（离散恒等式）：icosphere 角缺陷总和 = 4π（机器精度）
   (b) 测地圆周亏损（Bertrand–Puiseux）：球冠（K=1）网格测地半径 vs 环周长，
       亏损 Δ(s)=2πs−C(s) 与解析 2π(s−R sin(s/R)) 吻合、小 s 时 →(π/3)K̄s³；
       圆柱（K=0，可展）同机械量亏损 ≈ 0 —— 周长是否落后于半径 = 可展性判据
   (c) 内蕴长度平展（least-squares flattening）：手风琴条带（K=0，可展）边长残差
       压到机器零、复原真值平铺；鼓包条带（K≠0，非可展）残差不可压零、随幅度增长
2) rest shape 缺失的下游代价（控制变量实验）：
   真衣 = 平铺布条；"生成的 3D 服装" = 同等 RMS 位移的两种扰动：
     可展扰动（手风琴，褶皱）vs 非可展扰动（鼓包，内在拉伸）
   管线 A（平展→版型→仿真，版型路线的机械化）与
   管线 B（直接拿 3D 边长当 rest，无版型路线的常见 hack）：
   可展扰动两条管线都还原真值 drape（求解器地板级误差）；
   非可展扰动两条都伤、随幅度单调增长 —— 布料损伤只来自非可展分量
3) SDS 蒸馏机理（Tweedie 解析去噪器 + K=3 GMM 先验, render=恒等）：
   (a) 单步梯度 g = ε̂(θ+σε)−ε 的精确分解: g = σ(θ−μ₀)/v − ε·σ_c²/v
       → |E g| = σr/(σ_c²+σ²)（峰值在 σ=σ_c）, SNR = σr/(√2σ_c²)（∝σ）——均机器精度吻合
   (b) guidance λ 的代价: 有偏不动点沿"远离干扰模式质量"方向偏移
       （偏移源 = 先验混叠残差 m₀ = E[ε_mix](μ₀), |m₀|≈σ_c）; 线性化
       x* = (λ−1)m₀/(λā₁−(λ−1)β) 在 λ≤1.3 内吻合 <5%;
       主支在 λ*≈1.71 处跑向无穷远消失（线性化预测 1.98, 非线性提前）;
       λ≥λ* 只剩远处鞍点 + 远场曲率 ā(λ)=E_σ[−(λ−1)/σ+λσ/v]<0（反恢复）→
       随机动力学逃逸。论文 guidance=100 ≫ 两个临界值（分支死亡 ~1.71 / 远场失稳
       ~1.80）: 1200 步固定预算是截停, 不是收敛
   (c) 同冻结先验、同查询预算（1 查询 = 1 次去噪器前向）的三个估计器:
       SDS(λ=1, 24 视角平均) / 重采样伪 GT（链长 50）/ 固定伪 GT（K=24 重用）
       → 误差-预算全部落在理论线上; SDS 每查询效率 ≈ 6.2× 重采样伪 GT
       （= √50 链成本优势 × 0.87 Tweedie 噪声惩罚）; 固定伪 GT 有预算无关地板

实现注记（踩坑记录）:
- 手风琴折痕必须与网格线对齐（GG 10 教训），四边形对角线不得跨折痕；本篇折痕沿
  水平网格线、**顶带（挂钉边）不折**——钉住不引入强制拉伸。关键不变量：终止在
  折痕行上的对角线也保长（刚体旋转的转轴包含端点之一时两点距离不变）。
- 平展是非凸问题；初值取 (x, y3) 投影 + 微扰。可展面残差→0 且回到真值，非可展
  面残差>0——对照的是"能否无残差"，不是全局最优性。
- **软规范权重必须与残差同量级**（1e3 权重 vs 1e-3 残差 → trf 信任域停摆打满
  max_nfev，残差卡在 5.9e-3 伪影——健全性检查：输入=平铺真值本身也压不下去，
  才定位到是求解器问题而非几何问题；W_G=1 时 71 步收敛到 1e-16）。
- 平展解的规范是任意的（可旋转 90°/镜像）；进仿真前须做 2D Procrustes 对齐
  （刚体+镜像，无缩放）——版型放码/摆位是任何管线的必备步骤。
- 钉**顶**边（y=H）让布条受张悬挂——钉底边会得到受压直立态：完美对称初值停在
  不稳定平衡（真值 z≡0 装作稳定），非对称初值（B 管线）翻折下垂 y≈−0.5，
  A/B 对比变成 apples-to-oranges。受张悬挂的平衡态唯一、逐顶点可对照。
- XPBD 用边着色 Gauss–Seidel 向量化（水平/垂直各 2 色 + 对角 4 色，每色内约束
  互不共享顶点，numpy 批量投影 = 精确 GS）。
- 读平衡值前检查残余速度（GG 01 教训：残余振荡会伪装成 ~4% 系统误差）。
- **实验 3 踩坑（SDS toy）**：
  ① 方向符号——DreamFusion 是最小化 SDS 损失，更新必须 θ ← θ − η·g；写成 += 目标
  变斥子（λ=1 终点 5e12）。② 求根不能用固定 MC 样本集——λ 放大后 (ε_c−ε_mix) 的
  MC 噪声 ~λ/√N，"不动点"全是噪声伪根；改 Gauss–Hermite 求积（41²×9 节点，
  确定性、光滑被积函数下机器精度）。③ 多起点求根会跳支——延拓（λ 小步长、上一步
  解做初值）才能追踪主支到分岔；λ=2 的"根"其实是近零 Jacobian 方向的浅谷（残差
  1e-9 阈值要卡 cost 而非只看返回值）。④ 高 λ 动力学：梯度幅值 ∝λ 且远场反恢复
  （h(θ)≈ā(λ)θ, ā<0）→ 指数逃逸；梯度裁剪（‖g‖≤3）+ η 随 λ 缩小只能延缓，
  治本要靠"该 regime 本无吸引子"的结论本身。⑤ 固定统计量（预算无关量）的 200
  种子平均有 ±1.6σ 波动（实测 0.1524 vs 理论 0.1443），加到 20000 种子才贴线。
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

rng = np.random.default_rng(11)

# ================= 实验 1：可展性 =================
print("=== 实验 1: 可展性 —— 版型↔3D 的桥只在 K=0 时免费 ===")


def icosphere(sub=3):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], int)
    for _ in range(sub):
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        cache, newV, newF = {}, [v for v in V], []
        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = (V[a] + V[b]) / 2
                newV.append(m / np.linalg.norm(m))
                cache[key] = len(newV) - 1
            return cache[key]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        V, F = np.array(newV), np.array(newF, int)
    return V, F


def angle_defects(V, F):
    angsum = np.zeros(len(V))
    for f in F:
        for k in range(3):
            i, j, l = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
            e1, e2 = V[j] - V[i], V[l] - V[i]
            angsum[i] += np.arccos(np.clip(e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2)), -1, 1))
    return 2 * np.pi - angsum

V_ico, F_ico = icosphere(3)
def_ico = angle_defects(V_ico, F_ico)
print(f"\n  (a) icosphere Gauss–Bonnet: {len(V_ico)} 顶点, 角缺陷总和 = {def_ico.sum():.12f}"
      f"（2πχ = 4π = {4 * np.pi:.12f}, 偏差 {abs(def_ico.sum() - 4 * np.pi):.1e}）")
print(f"      全部缺陷 > 0（球面 K=1: 每顶点缺陷 ≈ K·A_v, 均值 {def_ico.mean():.2e}）")


# ---- (b) 测地圆周亏损: 球冠 vs 圆柱 ----
def geodesic_disc_mesh(map_fn, s_max, nr, nphi):
    """测地圆盘网格: 环 i 在测地半径 s_i = i·s_max/nr, 每环 nphi 段; 顶点 0 = 圆心"""
    P = [np.asarray(map_fn(0.0, 0.0), float)]   # 圆心 = map_fn(0,0), 非占位符
    ring0 = []
    for i in range(1, nr + 1):
        s = i * s_max / nr
        ring0.append(1 + (i - 1) * nphi)
        for k in range(nphi):
            ph = k * 2 * np.pi / nphi
            P.append(map_fn(s * np.cos(ph), s * np.sin(ph)))
    return np.array(P), ring0


def geodesic_deficit(P, ring0, nphi):
    """加权 Dijkstra 图测地 + 环周长 → 逐环亏损 Δ = 2πs − C（权重 = 3D 边长, 非 hop 数）"""
    rows, cols, data = [], [], []
    def add(a, b):
        w_ = np.linalg.norm(P[a] - P[b])
        rows.append(a); rows.append(b); cols.append(b); cols.append(a)
        data.append(w_); data.append(w_)
    for k in range(nphi):
        add(0, 1 + k)
    for i in range(len(ring0)):
        base = ring0[i]
        for k in range(nphi):
            add(base + k, base + (k + 1) % nphi)                  # 环边
            if i + 1 < len(ring0):
                add(base + k, ring0[i + 1] + k)                   # 径向边
                add(base + k, ring0[i + 1] + (k + 1) % nphi)      # 对角边
    G = coo_matrix((data, (rows, cols)), shape=(len(P), len(P))).tocsr()
    d = shortest_path(G, directed=False, indices=0)
    out = []
    for base in ring0:
        s = np.median(d[base: base + nphi])
        C = sum(np.linalg.norm(P[base + (k + 1) % nphi] - P[base + k]) for k in range(nphi))
        out.append((s, C))
    return out

R = 1.0
def cap_map(u, w):
    """测地极坐标 → 球面: (s,φ) ↦ R(sin(s/R)cosφ, sin(s/R)sinφ, cos(s/R))"""
    s = np.hypot(u, w)
    if s < 1e-12:
        return np.array([0.0, 0.0, R])
    th = s / R
    return R * np.array([np.sin(th) * u / s, np.sin(th) * w / s, np.cos(th)])

P_cap, r0_cap = geodesic_disc_mesh(cap_map, 0.8, 40, 180)
cap = geodesic_deficit(P_cap, r0_cap, 180)
print(f"\n  (b) 测地圆周亏损 Δ(s)=2πs−C(s), 球冠 R=1（K=1）网格 vs 解析 2π(s−sin s):")
for i in [2, 4, 8, 16, 32]:
    s, C = cap[i]
    ana = 2 * np.pi * (s - np.sin(s))
    print(f"      s={s:.3f}: 网格 Δ={2 * np.pi * s - C:.6f}, 解析={ana:.6f}"
          f"（相对偏差 {abs((2 * np.pi * s - C) - ana) / ana:.1e}）")
print( "      小圆极限 Δ/((π/3)s³)（→K̄=1; 最小环残存 O(dφ²) 弦长偏差, 随 s 消失）: "
      + ", ".join(f"s={cap[i][0]:.3f}: {(2 * np.pi * cap[i][0] - cap[i][1]) / ((np.pi / 3) * cap[i][0] ** 3):.3f}"
                  for i in [1, 2, 3, 5, 8]))

RC = 1.0
P_cyl, r0_cyl = geodesic_disc_mesh(
    lambda u, w: np.array([RC * np.sin(u / RC), RC * np.cos(u / RC), w]), 0.8, 40, 180)
cyl = geodesic_deficit(P_cyl, r0_cyl, 180)
worst = max(abs(2 * np.pi * s - C) / (2 * np.pi * s) for s, C in cyl[4:])
print(f"      圆柱（K=0, 可展）同机械量: |Δ|/(2πs) 最大 {worst:.1e} —— 周长不落后（展开即平面）")


# ---- (c) 内蕴长度平展: 手风琴 vs 鼓包 ----
L, H = 1.0, 0.5
NU, NV = 41, 21
xs = np.linspace(0, L, NU)
ys_flat = np.linspace(0, H, NV)          # j=0 → y=0（自由底边）, j=NV−1 → y=H（钉住的顶边）
dx, dy = L / (NU - 1), H / (NV - 1)

def vid(i, j):
    return i * NV + j

edges = []
for i in range(NU):
    for j in range(NV):
        if i + 1 < NU: edges.append((vid(i, j), vid(i + 1, j)))
        if j + 1 < NV: edges.append((vid(i, j), vid(i, j + 1)))
        if i + 1 < NU and j + 1 < NV: edges.append((vid(i, j), vid(i + 1, j + 1)))
edges = np.array(edges)

def flat_strip():
    V = np.zeros((NU * NV, 3))
    for i in range(NU):
        for j in range(NV):
            V[vid(i, j)] = [xs[i], ys_flat[j], 0.0]
    return V

# 手风琴: 折痕沿水平网格线, **顶带（j=NV−1 侧, 挂钉边）不折**; 自上而下 0,+α,−α,...
N_FOLD = 5                    # 带数, NV−1 = 20 = 5 带 × 4 行
RPB = (NV - 1) // N_FOLD
ALPHA = np.deg2rad(20.0)

def accordion_strip():
    theta = np.zeros(N_FOLD)
    for b in range(1, N_FOLD):
        theta[b] = ALPHA * (1 if b % 2 == 1 else -1)
    d = np.stack([np.cos(theta), np.sin(theta)], 1)     # 各带 (y,z) 方向
    bnd = np.zeros((N_FOLD + 1, 2))
    bnd[0] = [H, 0.0]                                    # 顶边 (y=H, z=0) 不动
    for b in range(N_FOLD):
        bnd[b + 1] = bnd[b] - RPB * dy * d[b]
    V = np.zeros((NU * NV, 3))
    for j in range(NV):
        m = (NV - 1) - j                                 # 距顶边的行数
        b = min(m // RPB, N_FOLD - 1)
        yz = bnd[b] - (m - b * RPB) * dy * d[b]
        for i in range(NU):
            V[vid(i, j)] = [xs[i], yz[0], yz[1]]
    return V

def bump_strip(a):
    V = np.zeros((NU * NV, 3))
    for i in range(NU):
        for j in range(NV):
            y = ys_flat[j]
            V[vid(i, j)] = [xs[i], y, a * np.sin(np.pi * xs[i] / L) * np.sin(np.pi * y / H)]
    return V

V_flat, V_acc = flat_strip(), accordion_strip()

l_acc = np.linalg.norm(V_acc[edges[:, 0]] - V_acc[edges[:, 1]], axis=1)
l_flat = np.linalg.norm(V_flat[edges[:, 0]] - V_flat[edges[:, 1]], axis=1)
iso_err = np.abs(l_acc - l_flat).max()
assert iso_err < 1e-12, f"手风琴不保长: {iso_err}"
rms_z_acc = np.sqrt((V_acc[:, 2] ** 2).mean())


def mesh_area(V):
    A = 0.0
    for i in range(NU - 1):
        for j in range(NV - 1):
            a, b, c, dd = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            A += 0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[dd] - V[a]))
            A += 0.5 * np.linalg.norm(np.cross(V[b] - V[c], V[dd] - V[c]))
    return A

A_flat = mesh_area(V_flat)
A_acc = mesh_area(V_acc)
probe = bump_strip(1.0)
rms_unit = np.sqrt((probe[:, 2] ** 2).mean())
A0 = rms_z_acc / rms_unit
V_bump = bump_strip(A0)
A_bump = mesh_area(V_bump)
excess_ana = A0 ** 2 * np.pi ** 2 / 8 * (1 / L ** 2 + 1 / H ** 2)
print(f"\n  (c) 平展（least-squares, 内蕴边长约束）: 条带 {NU}×{NV}, 钉顶边 y=H")
print(f"      手风琴: {N_FOLD} 折, 折角 {np.rad2deg(ALPHA):.0f}°, RMS z = {rms_z_acc:.4f}, "
      f"等距断言通过（max|Δℓ| = {iso_err:.0e}）, 面积比 {A_acc / A_flat:.12f}（=1 精确）")
print(f"      鼓包:   幅度 a = {A0:.4f}（RMS z 对齐 {np.sqrt((V_bump[:, 2] ** 2).mean()):.4f}）, "
      f"面积比 {A_bump / A_flat:.5f}（解析 1+{excess_ana:.5f}）")


def flatten(V3, noise=0.002):
    """内蕴长度平展: 求 2D 嵌入 min Σ(‖p_i−p_j‖−ℓ_e)²; 软规范(同量级权重): v00 于原点 + 定旋转"""
    n = len(V3)
    l3 = np.linalg.norm(V3[edges[:, 0]] - V3[edges[:, 1]], axis=1)
    p = np.stack([V3[:, 0], V3[:, 1]], 1).copy()
    if noise:
        p += rng.normal(0, noise, p.shape)
    E = len(edges)
    W_G = 1.0

    def res(q):
        P2 = q.reshape(n, 2)
        dvec = P2[edges[:, 0]] - P2[edges[:, 1]]
        r = np.linalg.norm(dvec, axis=1) - l3
        return np.concatenate([r, W_G * np.array([P2[0, 0], P2[0, 1], P2[NV, 0]])])

    rows, cols = [], []
    for e, (a, b) in enumerate(edges):
        for v_ in (a, b):
            rows += [e, e]; cols += [2 * v_, 2 * v_ + 1]
    for k, c_ in enumerate([0, 1, 2 * NV]):
        rows.append(E + k); cols.append(c_)
    S = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(E + 3, 2 * n)).tocsr()
    # 非可展面残差有不可压零的地板：容差必须高于地板的梯度水平（1e-12 会永远
    # 打满 max_nfev 追逐机器零；1e-8 时 91 步 gtol 正常停机, 解与打满时逐位一致）
    sol = least_squares(res, p.ravel(), jac_sparsity=S, method='trf',
                        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=12000)
    assert sol.status != 0, f"平展求解器打满 max_nfev（status=0）——检查规范权重/量纲"
    P2 = sol.x.reshape(n, 2)
    resid = np.abs(np.linalg.norm(P2[edges[:, 0]] - P2[edges[:, 1]], axis=1) - l3)
    return P2, resid


def procrustes2d(P2, T2):
    """刚体+镜像对齐（无缩放）: 返回对齐后的 P2"""
    pc, tc = P2 - P2.mean(0), T2 - T2.mean(0)
    M = tc.T @ pc
    U, _, Vt = np.linalg.svd(M)
    Rt = U @ Vt
    if np.linalg.det(Rt) < 0:                       # 允许镜像（版型手性合法）
        Vt[-1] *= -1
        Rt = U @ Vt
    return pc @ Rt.T + T2.mean(0)

P2_acc, resid_acc = flatten(V_acc)
P2_acc_al = procrustes2d(P2_acc, V_flat[:, :2])
pos_err_acc = np.abs(P2_acc_al - V_flat[:, :2]).max()
P2_bump, resid_bump = flatten(V_bump)
P2_bump_al = procrustes2d(P2_bump, V_flat[:, :2])
print(f"\n      平展边长残差（能否造出贴合生成 mesh 的平布）: 手风琴 max {resid_acc.max():.1e}"
      f" / 中位 {np.median(resid_acc):.1e}  ← 可展: 机器零")
print(f"                    鼓包   max {resid_bump.max():.1e} / 中位 {np.median(resid_bump):.1e}"
      f"  ← 非可展: 残差地板")
l2_acc = np.linalg.norm(P2_acc[edges[:, 0]] - P2_acc[edges[:, 1]], axis=1)
l2_bump = np.linalg.norm(P2_bump[edges[:, 0]] - P2_bump[edges[:, 1]], axis=1)
rel_bump = np.abs(l2_bump - l_flat) / l_flat
print(f"      复原版型 vs 真值平铺的边长偏差（造出的衣服 ≠ 设计的那件）:")
print(f"        手风琴: max 相对偏差 {np.abs(l2_acc - l_flat).max() / l_flat.min():.1e}, "
      f"对齐后位置 max 偏差 {pos_err_acc:.1e}")
print(f"        鼓包:   max 相对偏差 {rel_bump.max():.4%}, 中位 {np.median(rel_bump):.4%}"
      f"（面积过剩解析 {excess_ana:.4%} 同量级）")

print(f"      鼓包幅度扫描（残差与版型畸变都应 ∝ a²）:")
for a in [A0 / 2, A0, 2 * A0]:
    Vb = bump_strip(a)
    P2b, rb = flatten(Vb)
    l2b = np.linalg.norm(P2b[edges[:, 0]] - P2b[edges[:, 1]], axis=1)
    relb = np.abs(l2b - l_flat) / l_flat
    ex = a ** 2 * np.pi ** 2 / 8 * (1 / L ** 2 + 1 / H ** 2)
    print(f"        a={a:.4f}（RMS z={a * rms_unit:.4f}）: 残差 max {rb.max():.2e}, "
          f"版型边长偏差 max {relb.max():.4%}, 面积过剩解析 {ex:.4%}")

# ================= 实验 2：rest shape 缺失的下游代价 =================
print("\n=== 实验 2: 同等 3D 误差的可展/非可展扰动 —— drape 下游代价 ===")


def edge_colors():
    """边着色 GS: 每色内约束互不共享顶点 → numpy 批量投影 = 精确 Gauss–Seidel"""
    cols = {}
    for i in range(NU):
        for j in range(NV):
            if i + 1 < NU:
                cols.setdefault(('h', i % 2), []).append((vid(i, j), vid(i + 1, j)))
            if j + 1 < NV:
                cols.setdefault(('v', j % 2), []).append((vid(i, j), vid(i, j + 1)))
            if i + 1 < NU and j + 1 < NV:
                cols.setdefault(('d', i % 2 * 2 + j % 2), []).append((vid(i, j), vid(i + 1, j + 1)))
    return [(np.array(c)[:, 0], np.array(c)[:, 1]) for c in
            (cols[k] for k in sorted(cols, key=str))]

COLORS = edge_colors()
PIN = np.array([vid(i, NV - 1) for i in range(NU)])   # 钉顶边 y=H → 受张悬挂, 平衡唯一
PIN_POS = V_flat[PIN].copy()
BOTTOM = np.array([vid(i, 0) for i in range(NU)])      # 自由底边（真值 y≈0）


def xpbd(x0, rest, g=9.8, dt=1 / 120, iters=12, comp=1e-7, damp=0.99,
         max_steps=4800, chunk=600, v_tol=5e-3):
    """边着色 XPBD; 钉顶边于真值位置; 自适应跑到准静态"""
    x = x0.copy(); x[PIN] = PIN_POS
    v = np.zeros_like(x)
    invm = np.ones(len(x)); invm[PIN] = 0.0
    lam = [np.zeros(len(a)) for a, _ in COLORS]
    comp_t = comp / dt ** 2
    steps_done = 0
    while steps_done < max_steps:
        for _ in range(chunk):
            v *= damp
            v[:, 1] -= g * dt
            xt = x + v * dt
            xt[PIN] = PIN_POS
            xold = x.copy()
            for l_ in lam: l_[:] = 0.0
            for _ in range(iters):
                for ci, (a, b) in enumerate(COLORS):
                    dvec = xt[a] - xt[b]
                    dist = np.maximum(np.linalg.norm(dvec, axis=1), 1e-12)
                    nvec = dvec / dist[:, None]
                    C = dist - rest[ci]
                    w = invm[a] + invm[b]
                    dl = (-C - comp_t * lam[ci]) / (w + comp_t)
                    lam[ci] += dl
                    np.add.at(xt, a, invm[a][:, None] * nvec * dl[:, None])
                    np.add.at(xt, b, -invm[b][:, None] * nvec * dl[:, None])
            xt[PIN] = PIN_POS
            v = (xt - xold) / dt
            v[PIN] = 0.0
            x = xt
            steps_done += 1
        if np.abs(v).max() < v_tol:
            break
    return x, np.abs(v).max(), steps_done


def rest_from(P):
    """按 COLORS 顺序取边长"""
    return [np.linalg.norm(P[a] - P[b], axis=1) for a, b in COLORS]


def sim_report(name, x0, rest, x_truth):
    x_eq, vmax, ns = xpbd(x0, rest)
    dev = np.linalg.norm(x_eq - x_truth, axis=1)
    zrms = np.sqrt((x_eq[:, 2] ** 2).mean())
    strain = [np.abs(np.linalg.norm(x_eq[a] - x_eq[b], axis=1) - rest[ci]) / rest[ci]
              for ci, (a, b) in enumerate(COLORS)]
    strain = np.concatenate(strain)
    print(f"      {name:26s} 偏差 mean {dev.mean():.2e} / max {dev.max():.2e} m, "
          f"z-RMS {zrms:.1e}, 底边 y {x_eq[BOTTOM, 1].mean():.4f}, "
          f"平衡应变 mean {strain.mean():.1e}"
          f"（max|v|={vmax:.1e}, {ns} 步）")
    return dev

x_truth, vmax_t, ns_t = xpbd(V_flat.copy(), rest_from(V_flat))
print(f"\n  真值: 平铺布条钉顶边受张悬挂 → 平面平衡（max|v|={vmax_t:.1e}, {ns_t} 步, "
      f"z-RMS {np.sqrt((x_truth[:, 2] ** 2).mean()):.1e}, 底边 y {x_truth[BOTTOM, 1].mean():.4f}）")

print(f"\n  可展扰动（手风琴, RMS z = {rms_z_acc:.4f}）:")
A_acc3 = np.concatenate([P2_acc_al, np.zeros((NU * NV, 1))], 1)   # 平展版型, 对齐后摆到 z=0
sim_report("管线 A（平展→版型→仿真）", A_acc3, rest_from(A_acc3), x_truth)
sim_report("管线 B（3D 边长当 rest）", V_acc, rest_from(V_acc), x_truth)

print(f"\n  非可展扰动（鼓包, RMS z 与手风琴对齐）:")
for a in [A0 / 2, A0, 2 * A0]:
    Vb = bump_strip(a)
    P2b, rb = flatten(Vb)
    P2b_al = procrustes2d(P2b, V_flat[:, :2])
    Ab3 = np.concatenate([P2b_al, np.zeros((NU * NV, 1))], 1)
    print(f"    a = {a:.4f}（RMS z = {a * rms_unit:.4f}, 平展残差 max {rb.max():.2e}）:")
    sim_report("  管线 A（平展→版型→仿真）", Ab3, rest_from(Ab3), x_truth)
    sim_report("  管线 B（3D 边长当 rest）", Vb, rest_from(Vb), x_truth)

Ey = A0 ** 2 * (np.pi / H) ** 2 / 4          # E[(∂w/∂y)²]
print(f"\n  B 管线解析锚: 底边下垂过剩 ≈ H·E[(∂w/∂y)²]/2 = {H * Ey / 2:.4f}"
      f"（对照 a={A0:.4f} 行底边 y − 真值底边 y）")

# B-accordion 的亚稳检验: 收敛后仍偏 = 皱纹亚稳态（等距嵌入零能量陷阱）, 非未收敛
x_eq, vmax, ns = xpbd(V_acc.copy(), rest_from(V_acc), v_tol=1e-5, max_steps=9600)
dev = np.linalg.norm(x_eq - x_truth, axis=1)
print(f"\n  B-accordion 亚稳检验: v_tol=1e-5 再跑 → {ns} 步, max|v|={vmax:.1e}, "
      f"偏差 mean {dev.mean():.2e} / max {dev.max():.2e} 不再下降"
      f" ← 收敛到皱纹亚稳态而非真值（rest 长度已断言 = 真值 6e-17, 度量无损伤）")

# ================= 实验 3：SDS —— 不落地的先验蒸馏 =================
print("\n=== 实验 3: SDS 梯度结构（GarmentDreamer §3.2 机理 toy: 先验只蒸馏到中间 3DGS） ===")
# 冻结的"2D 扩散先验" = 解析 GMM（K=3 模式, 各向同性 σ_c）; 最优去噪器由 Tweedie
# 公式解析给出（ε̂(x_t) = (x_t − E[y|x_t])/σ）——充当"冻结的 Stable Diffusion"。
# "3D 表示" θ ∈ R², render = 恒等（单视角; 24 视角 = 24 次独立噪声平均, 注记见 (c)）。
# conditioning 选定 μ₀ = 目标（"文本描述的那件衣服"）; 干扰模式 = 其他服装类型。
MU = np.array([[0.0, 0.0], [1.1, 0.5], [-0.9, 0.7]])
SC, PI = 0.5, np.ones(3) / 3

def _sigprep(sig):
    """标量 σ 与 (N,) 数组 σ 统一广播（(N,1) 展开供 (N,2) 批量求值）"""
    sig = np.asarray(sig, float)
    if np.ndim(sig):
        return sig[:, None], (sig ** 2)[:, None], (SC ** 2 + sig ** 2)[:, None]
    return sig, sig ** 2, SC ** 2 + sig ** 2

def post_mean_mix(x, sig):
    single = np.ndim(x) == 1
    X = np.atleast_2d(np.asarray(x, float))
    _, _, v = _sigprep(sig)
    d2 = ((X[:, None, :] - MU[None]) ** 2).sum(-1)
    logw = np.log(PI) - 0.5 * d2 / v
    logw -= logw.max(-1, keepdims=True)
    w = np.exp(logw); w /= w.sum(-1, keepdims=True)
    out = np.einsum('nk,kd->nd', w, MU)
    return out[0] if single else out

def eps_mix(x, sig):
    s, _, _ = _sigprep(sig)
    return (x - post_mean_mix(x, sig)) / s

def eps_cond(x, sig):
    """选中 μ₀ 的条件去噪器（完美 conditioning 的解析极限）"""
    s, s2, v = _sigprep(sig)
    m = (SC ** 2 * x + s2 * MU[0]) / v
    return (x - m) / s

def eps_cfg(x, sig, lam):
    return eps_mix(x, sig) + lam * (eps_cond(x, sig) - eps_mix(x, sig))

# Gauss–Hermite 求积: h(θ;λ) = E_{σ,ε}[ε_cfg(θ+σε)] 的确定性高精度求值
# （λ 大时 (c) 项被 λ 放大, MC 噪声会把"根"变成伪根——固定 256 样本集的根是噪声伪影）
GH_N = 41
_gh_u, _gh_w = np.polynomial.hermite.hermgauss(GH_N)
GH_U = np.stack(np.meshgrid(_gh_u, _gh_u, indexing='ij'), -1).reshape(-1, 2)
GH_W = (_gh_w[:, None] * _gh_w[None, :]).reshape(-1) / np.pi
SIG_GRID = np.linspace(0.2, 1.0, 9)   # 退火 σ ~ U(0.2, 1.0) 的离散化

def h_exact(th, lam):
    tot = np.zeros(2)
    for sg in SIG_GRID:
        xt = np.asarray(th)[None, :] + sg * np.sqrt(2) * GH_U
        tot += (eps_cfg(xt, sg, lam) * GH_W[:, None]).sum(0)
    return tot / len(SIG_GRID)

# ---- (a) 单步梯度统计: 精确分解 g = σ(θ−μ₀)/v − ε·σ_c²/v ----
# λ=1: E[g] = σ(θ−μ₀)/v（信号, 峰值在 σ=σ_c）; 噪声系数 σ_c²/v（σ=σ_c 时 = 1/2）
theta0 = np.array([1.1, 0.4])
r0 = np.linalg.norm(theta0 - MU[0])
M = 200000
print(f"\n  (a) SDS 单步梯度 g = ε̂_cond(θ+σε)−ε 的统计（θ 距目标 r₀={r0:.3f}, λ=1）:")
print(f"      精确理论: g = σ(θ−μ₀)/v − ε·σ_c²/v ⇒ |E g| = σr/v, SNR = σr/(√2σ_c²)")
for sig in [0.05, 0.10, 0.25, 0.50, 1.00]:
    e = rng.normal(size=(M, 2))
    xt = theta0[None] + sig * e
    g = eps_cond(xt, sig) - e
    m_abs = np.linalg.norm(g.mean(0))
    noise = np.sqrt((g.std(0) ** 2).sum())        # 2D 噪声 RMS
    v = SC ** 2 + sig ** 2
    print(f"      σ={sig:5.2f}: |E g|={m_abs:.4f}（理论 σr/v={sig * r0 / v:.4f}, "
          f"比值 {m_abs / (sig * r0 / v):.4f}）, SNR={m_abs / noise:.4f}"
          f"（理论 σr/(√2σ_c²)={sig * r0 / (np.sqrt(2) * SC ** 2):.4f}, "
          f"比值 {(m_abs / noise) / (sig * r0 / (np.sqrt(2) * SC ** 2)):.4f}）")
print(f"      → 信号 |E g| 在 σ=σ_c 处峰值（步长/曲率视角）; 每抽单步 SNR 却随 σ 线性升")
print(f"        （噪声系数 σ_c²/v 随 σ 降——同一 σ_c 让信号最大与噪声最小不可兼得）")
print(f"      → 每步 24 视角平均: 方差 ↓24, SNR ↑√24≈{np.sqrt(24):.1f}×")

# ---- (b) guidance λ 的代价: 有偏不动点 + 分支消亡 + 动力学逃逸 ----
# CFG 合成去噪器 ε_cfg = λ·ε_cond + (1−λ)·ε_mix; 退火更新的不动点 = h(θ;λ)=0 的根。
# λ=1: h = ā₁·θ 精确线性（唯一根 = μ₀）; λ>1: 混叠残差 m₀ ≡ E[ε_mix](μ₀) ≠ 0 进入。
m0 = h_exact(np.zeros(2), 0.0)                    # = E[ε_mix](μ₀): 先验混叠残差
a1 = np.mean([sg / (SC ** 2 + sg ** 2) for sg in SIG_GRID])   # = E[σ/v]
Bmat = np.zeros((2, 2))
for j in range(2):
    dvec = np.zeros(2); dvec[j] = 1e-6
    Bmat[:, j] = (h_exact(dvec, 0.0) - h_exact(-dvec, 0.0)) / 2e-6   # ∇E[ε_mix](0)
mh = m0 / np.linalg.norm(m0)
beta = mh @ Bmat @ mh
lam_star_lin = beta / (beta - a1)
print(f"\n  (b) guidance 固定点分析（GH 求积 h(θ;λ), 41²×9 节点, 确定性）:")
print(f"      先验混叠残差 m₀ = E[ε_mix](μ₀) = [{m0[0]:+.4f},{m0[1]:+.4f}], "
      f"|m₀|={np.linalg.norm(m0):.4f} ≈ {np.linalg.norm(m0) / SC:.2f}σ_c")
print(f"      （指向干扰模式质心的反方向——无条件先验在目标真值处仍被其他模式拉扯）")
print(f"      线性化: h ≈ λā₁θ + (1−λ)(m₀+Bθ) ⇒ x* = (λ−1)m₀/(λā₁−(λ−1)β), "
      f"ā₁={a1:.4f}, β=m̂ᵀBm̂={beta:.4f}, 解析临界 λ*={lam_star_lin:.3f}")
print(f"\n      {'λ':>6} {'精确根 ‖x*‖':>12} {'线性化 ‖x*‖':>12} {'线性化相对误差':>14}")
x_cont = np.zeros(2)
cont_rows = []
for lam in [1.0, 1.02, 1.05, 1.10, 1.20, 1.30, 1.50, 1.60, 1.70]:
    sol = least_squares(lambda z: h_exact(z, lam), x_cont,
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    hn = np.linalg.norm(h_exact(sol.x, lam))
    x_cont = sol.x
    xlin = (lam - 1) * m0 / (lam * a1 - (lam - 1) * beta)
    rel = (np.linalg.norm(x_cont) - np.linalg.norm(xlin)) / np.linalg.norm(x_cont) if lam > 1 else 0.0
    cont_rows.append((lam, np.linalg.norm(sol.x), hn))
    print(f"      {lam:6.2f} {np.linalg.norm(sol.x):12.4f} {np.linalg.norm(xlin):12.4f} "
          f"{rel:14.3f}   x*=[{sol.x[0]:+.3f},{sol.x[1]:+.3f}]")
print(f"      → 偏移近似 ∝(λ−1)（斜率 |m₀|/ā₁≈{np.linalg.norm(m0) / a1:.2f}）, 方向 = −m₀（远离干扰质量）")
print(f"      → λ→λ*: 根范数发散（主支跑向无穷远后消失）:")
x_cont = np.array([-0.3182, -3.0335])
for lam in [1.700, 1.705, 1.708, 1.710, 1.712]:
    sol = least_squares(lambda z: h_exact(z, lam), x_cont,
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=5000)
    x_cont = sol.x
    print(f"        λ={lam:.3f}: ‖x*‖ = {np.linalg.norm(x_cont):9.3f}")
print(f"      → 主支在 λ*≈1.71 消失（线性化预测 {lam_star_lin:.2f}, 非线性提前）; "
      f"λ≥λ* 只剩远处鞍点（如 λ=2: x*=(0.36,3.48), ∇h 特征值 +2.98/−0.33）")
# 远场: h(θ)/θ → ā(λ)（大距离解析系数; 反恢复 ⇒ 下降动力学逃逸）
th_far = 50 * np.array([1., 1.]) / np.sqrt(2)
print(f"\n      远场曲率（θ_far=50·(1,1)/√2; h(θ)≈ā(λ)θ, ā<0 ⇒ 反恢复 ⇒ 发散）:")
for lam in [1, 2, 5, 100]:
    a_num = h_exact(th_far, lam) @ th_far / (th_far @ th_far)
    a_ana = np.mean([-(lam - 1) / sg + lam * sg / (SC ** 2 + sg ** 2) for sg in SIG_GRID])
    print(f"        λ={lam:4d}: 数值 ā={a_num:9.3f}, 解析 ā={a_ana:9.3f}（比值 {a_num / a_ana:.3f}）")
print(f"      → ā(λ)>0 ⟺ λ < E[1/σ]/(E[1/σ]−ā₁) ≈ "
      f"{np.mean(1 / SIG_GRID) / (np.mean(1 / SIG_GRID) - a1):.2f}: 两个临界值都在 ~1.7–1.8,")
print(f"        论文 guidance=100 深入崩溃区")
# 动力学验证: λ=1 收敛到根; λ≥2 逃逸
def run_sds_b(lam, eta0, n_steps, seed, clip=3.0):
    r_ = np.random.default_rng(seed)
    th = np.array([0.9, 0.5])
    for t in range(n_steps):
        sig = r_.uniform(0.2, 1.0)
        e = r_.normal(size=2)
        g = eps_cfg(th + sig * e, sig, lam) - e
        nrm = np.linalg.norm(g)
        if nrm > clip: g = g * clip / nrm
        th = th - eta0 / (1 + 1e-3 * t) * g
    return th

print(f"\n      随机退火动力学（η₀=0.02, 梯度裁剪 ‖g‖≤3; 8 种子平均终点）:")
for lam in [1, 2, 5, 100]:
    ends = np.array([run_sds_b(lam, 0.02, 6000, s) for s in range(8)])
    print(f"        λ={lam:4d} (6000 步): ‖θ̄−μ₀‖ = {np.linalg.norm(ends.mean(0)):9.4f}"
              f"   （λ=1 落在机器零根上; λ≥2 沿 −m₀ 方向逃逸）")
th = np.array([0.9, 0.5]); r_ = np.random.default_rng(0)
for t in range(2001):
    sig = r_.uniform(0.2, 1.0); e = r_.normal(size=2)
    g = eps_cfg(th + sig * e, sig, 5) - e
    nrm = np.linalg.norm(g)
    if nrm > 3: g = g * 3 / nrm
    th = th - 0.02 / (1 + 1e-3 * t) * g
    if t in (20, 500, 1000, 2000):
        print(f"          λ=5 轨迹: t={t:4d} θ=[{th[0]:8.3f},{th[1]:8.3f}]")
print(f"      → 高 guidance 的 SDS 没有可收敛的目标: 'guidance 100 + 1200 步固定预算'")
print(f"        （GarmentDreamer 代码配置）是截停而非收敛; 前期沿 −m₀ 的漂移即'风格化传输'")

# ---- (c) 同冻结先验、同查询预算: SDS vs 重采样伪 GT vs 固定伪 GT ----
# 1 查询 = 1 次去噪器前向。伪 GT 样本 = 一条完整反扩散链 = 50 查询（DDPM-50）。
# SDS 步 = 24 次前向（24 视角平均）。方差最优步长 η_t = 1/(ā₁t)（Robbins–Monro,
# 该调度下 θ_T = −(1/(ā₁T))Σξ_t, RMS 误差 = √V/(ā₁√T) 精确）。
CHAIN = 50
Ev = np.mean([(SC ** 2 / (SC ** 2 + sg ** 2)) ** 2 for sg in SIG_GRID])
V_step = 2 * Ev / 24            # 每步 2D 噪声方差（24 视角平均后）
print(f"\n  (c) 同查询预算下三个估计器（λ=1; RMS 误差, 200 种子）:")
print(f"      理论: SDS = √V/(ā₁√T)（V={V_step:.5f}, T=B/24）; 重采样 PG = σ_c√(2·{CHAIN}/B);")
print(f"            固定 PG = σ_c√(2/K)（K=24, 预算无关地板 {SC * np.sqrt(2 / 24):.4f}）")
def sds_run_c(T, seed, nview=24):
    r = np.random.default_rng(seed)
    th = np.array([0.9, 0.5])
    for t in range(1, T + 1):
        sig = r.uniform(0.2, 1.0, size=nview)
        e = r.normal(size=(nview, 2))
        x = th[None] + sig[:, None] * e
        g = (eps_cond(x, sig) - e).mean(0)
        th = th - 1.0 / (a1 * t) * g
    return th

NSEED = 200
print(f"\n      {'预算 B':>8} {'SDS 实测':>10} {'SDS 理论':>10} {'重采样PG':>10} {'PG 理论':>10} {'固定PG':>8}")
for T in [100, 400, 1200, 1600, 6400, 25600]:
    B = 24 * T
    ns = B // CHAIN
    e_sds = np.array([np.linalg.norm(sds_run_c(T, s)) for s in range(NSEED)])
    rms_sds = np.sqrt(np.mean(e_sds ** 2))
    e_rs = np.array([np.linalg.norm(
        np.random.default_rng(10000 + s).normal(0, SC, (ns, 2)).mean(0))
        for s in range(NSEED)])
    rms_rs = np.sqrt(np.mean(e_rs ** 2))
    e_fx = np.array([np.linalg.norm(
        np.random.default_rng(20000 + s).normal(0, SC, (24, 2)).mean(0))
        for s in range(20000)])   # 预算无关 → 一次算足种子, 避免 200 种子的 ±1.6σ 波动
    rms_fx = np.sqrt(np.mean(e_fx ** 2))
    tag = "  ← 论文预算(24×1200)" if T == 1200 else ""
    print(f"      {B:8d} {rms_sds:10.4f} {np.sqrt(V_step) / (a1 * np.sqrt(T)):10.4f} "
          f"{rms_rs:10.4f} {SC * np.sqrt(2.0 / ns):10.4f} {rms_fx:8.4f}{tag}")
ratio_theory = (SC * np.sqrt(2 * CHAIN)) / (np.sqrt(V_step) * np.sqrt(24) / a1)
print(f"      → 同预算效率比 SDS:重采样PG = {ratio_theory:.1f}× "
      f"（√{CHAIN}≈{np.sqrt(CHAIN):.2f} 链成本优势 ÷ "
      f"{np.sqrt(24) * np.sqrt(V_step) / (a1 * np.sqrt(2) * SC):.2f} Tweedie 噪声代偿）")
print(f"      → 固定伪 GT（GG 10 MagicMan 式: 生成一次, 确定性回归）: 采样噪声地板")
print(f"        {SC * np.sqrt(2 / 24):.3f} 与预算无关——回归阶段不再付方差, 但继承偏移")
print(f"      → GarmentDreamer = 两者串联: 便宜含噪的 SDS 探索（3DGS 阶段, guidance 100）")
print(f"        + 干净的确定性回归出最终资产（mesh 阶段 L1 对 3DGS 渲染）——回归继承")
print(f"        SDS 的全部偏移(guidance 偏置 + 先验混叠), 只是不再付它的方差")
