"""GG 10 数值实验：单图 → 仿真就绪链路的机理验证（numpy/scipy，无学习框架）
1) pattern coordinates 的度量局部性：近不可伸长布条折成紧密手风琴后，
   3D 欧氏"感受野"里大半点是隔着几层布的远邻（布料距离可达半径数倍），
   而 UV/裁片坐标把布料的内蕴度量变成欧氏度量（可展等距折叠 → 测地 =
   参数距离，解析精确 + 图测地独立验证）—— DMap 用 pattern coordinates
   做变形学习载体的机理
2) 单视角深度零空间 vs 多视角伪 GT + 先验投影：单视角观测对深度方向盲
   （零空间），伪 GT 用幻觉填零空间（低频偏差 + 高频噪声）；弯曲先验是
   低通滤波器——滤掉高频（无论对错）、放行低频（无论对错）；只有真
   第二视角根治零空间 —— Dress-1-to-3 "初估 + 多视角伪 GT + 可微物理
   精修" 流程的误差结构
3) 逐帧独立/准静态映射 vs 时序-动力学建模：受驱动阻尼振子的频响
   （解析 + ODE 数值双验证），准静态误差随驱动频率单调增长；速度+
   加速度一阶修正由数据拟合自动恢复动力学系数（α1→−c/k, α2→−m/k）、
   低频段吃掉大部分缺口，共振段无解 —— DMap 静态版逐帧独立用于视频
   会抖动（TPAMI Table III: 顶点加速度 18.2 vs GT 3.7 mm/frame²）、
   动态版时序建模的动机与边界

实现注记（踩坑记录）:
- 手风琴折叠的段方向数组必须按【段号】奇偶取值; 初版误用 u 网格点索引
  （网格点 kk 的段号是 floor(kk/15) 而非 kk）, 折痕结构整体错乱——
  3D 总长仍 =1（等距性检查不报错）, 但"层"不存在了。教训: 等距性检查
  抓不住拓扑/折叠结构的错误。
- 图测地验证要求 v 方向分辨率与 u 相当（H·dv ≈ du）, 否则粗方向的
  阶梯效应 dominates（dv 粗时同行点对精确、跨行偏差 ~30%）。
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr
from scipy.spatial import cKDTree

rng = np.random.default_rng(10)

# ================= 实验 1：pattern coordinates 的度量局部性 =================
print("=== 实验 1: UV≈布料测地 vs 3D 欧氏跨层混淆 —— pattern coordinates 的机理 ===")

# 手风琴折叠布帘: 布长 1（u 按弧长参数化）, 折 16 折, 折角 α, v 方向拉伸 H
# 扇形折: 偶段方向 +α, 奇段方向 π−α —— 同奇偶段平行叠层, 层间隙 2·SEG·sinα·cosα
# 网格约定: NU−1 被 2·N_FOLD 整除（折点对齐网格点, 边不跨折面抄近路）,
#           H·dv ≈ du（两方向分辨率相当, 图测地才无粗方向阶梯偏差）
N_FOLD, ALPHA = 16, np.deg2rad(10.0)
N_SEG = 2 * N_FOLD
SEG = 1.0 / N_SEG
GAP = 2 * SEG * np.sin(ALPHA) * np.cos(ALPHA)
NU, NV, H = 481, 241, 0.5          # du = 1/480, H·dv = 0.5/240 → 两者相等

u = np.linspace(0, 1, NU)
v = np.linspace(0, 1, NV)
seg = np.minimum((u / SEG).astype(int), N_SEG - 1)     # 每个网格点所属段
tin = u - seg * SEG                                    # 段内弧长
seg_ang = np.where(np.arange(N_SEG) % 2 == 0, ALPHA, np.pi - ALPHA)
seg_d = np.stack([np.cos(seg_ang), np.sin(seg_ang)], 1)  # 段方向（按段号!）
bnd = np.zeros((N_SEG + 1, 2))
for kk in range(N_SEG):
    bnd[kk + 1] = bnd[kk] + SEG * seg_d[kk]
P2 = bnd[seg] + tin[:, None] * seg_d[seg]              # (NU,2) 折叠平面坐标

XYZ = np.zeros((NU, NV, 3))                            # 3D: (x, y=v·H, z)
XYZ[..., 0] = P2[:, 0][:, None]
XYZ[..., 1] = v[None, :] * H
XYZ[..., 2] = P2[:, 1][:, None]
flat = XYZ.reshape(-1, 3)
ii, jj = np.unravel_index(np.arange(NU * NV), (NU, NV))

# 内蕴（pattern）测地: 可展等距折叠 → 测地 = sqrt(Δu² + H²Δv²), 解析精确
def intrinsic(ui, vi, uj, vj):
    return np.sqrt((u[ui] - u[uj]) ** 2 + (H * (v[vi] - v[vj])) ** 2)

assert abs(np.linalg.norm(np.diff(P2, axis=0), axis=1).sum() - 1.0) < 1e-12  # 等距性

# ---- (a) 图测地独立验证解析公式 ----
rows, cols, ws = [], [], []
for i in range(NU):
    for j in range(NV):
        for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            i2, j2 = i + di, j + dj
            if 0 <= i2 < NU and 0 <= j2 < NV:
                rows.append(i * NV + j); cols.append(i2 * NV + j2)
                ws.append(float(np.linalg.norm(XYZ[i, j] - XYZ[i2, j2])))
G = coo_matrix((ws, (rows, cols)), shape=(NU * NV, NU * NV)).tocsr()
chk = rng.choice(NU * NV, 60, replace=False)
Dg = shortest_path(G, directed=False, indices=chk)
rel, negmin = [], 0.0
for r_, node in enumerate(chk):
    i1, j1 = np.unravel_index(node, (NU, NV))
    ana = intrinsic(np.full(NU * NV, i1), np.full(NU * NV, j1), ii, jj)
    m_ = ana > 1e-2
    dev = Dg[r_][m_] - ana[m_]
    rel.append(np.median(np.abs(dev) / ana[m_]))
    negmin = min(negmin, dev.min())
print(f"\n  布帘: 布长 1, {N_FOLD} 折（{N_SEG} 段）, 折角 {np.rad2deg(ALPHA):.0f}°, "
      f"层间隙 {GAP:.4f}（= 布条全长的 {GAP:.1%}）")
print(f"  (a) 图测地（3D 边长权 + Dijkstra, 60 个源点） vs 解析内蕴距离:")
print(f"      中位相对偏差 {np.median(rel):.1%}, 且全部偏差 ≥ {negmin:.0e}——系统性为正"
      f"（8 邻接方向的方向量化, 浅斜率路径只能绕远）, 无一例负偏差"
      f"（图路径没有穿越空气的捷径）→ 折叠是等距的, UV 距离 = 布料测地")

# ---- (b) 3D 感受野 vs pattern 感受野 ----
r = 0.035
tree = cKDTree(flat)
frac_r, frac_2r, max_geo, nfield = [], [], [], []
for _ in range(200):
    qi = rng.integers(60, NU - 60)
    qj = rng.integers(40, NV - 40)
    node = qi * NV + qj
    nbr = np.array(tree.query_ball_point(flat[node], r))
    nbr = nbr[nbr != node]
    geo = intrinsic(np.full(len(nbr), qi), np.full(len(nbr), qj), ii[nbr], jj[nbr])
    frac_r.append((geo > r).mean())
    frac_2r.append((geo > 2 * r).mean())
    max_geo.append(geo.max())
    nfield.append(len(nbr))
print(f"\n  (b) 3D 欧氏感受野半径 r={r}（≈ {r/GAP:.1f} 个层间隙）, 200 个随机查询点:")
print(f"      感受野内平均 {np.mean(nfield):.0f} 个点: 布料距离 > r 的占 {np.mean(frac_r):.1%},"
      f" > 2r 的占 {np.mean(frac_2r):.1%}, 最远 {np.max(max_geo):.3f}（= {np.max(max_geo)/r:.1f}×r）")
print(f"      pattern 感受野同半径: 布料距离 ≤ r（构造保证——参数距离就是布料距离）")

# ---- (c) 相关性: 3D 距离在褶皱带内不携带布料距离信息 ----
pp = rng.choice(NU * NV, 4000)
pp2 = rng.choice(NU * NV, 4000)
keep = pp != pp2
pp, pp2 = pp[keep], pp2[keep]
d3p = np.linalg.norm(flat[pp] - flat[pp2], axis=1)
geop = intrinsic(ii[pp], jj[pp], ii[pp2], jj[pp2])
rho_all = spearmanr(d3p, geop).statistic
loc = d3p <= 0.1
rho_loc = spearmanr(d3p[loc], geop[loc]).statistic
band = (d3p > GAP * 0.8) & (d3p < GAP * 2.2)
print(f"\n  (c) Spearman ρ(3D 距离, 布料测地): 全域 {rho_all:.2f} → 局部（d3≤0.1）{rho_loc:.2f}")
print(f"      3D 距离 ∈ [0.8g, 2.2g]（g=层间隙）的点对: 布料测地从 {geop[band].min():.3f} "
      f"到 {geop[band].max():.3f}（{geop[band].max()/geop[band].min():.0f}× 全谱, 中位 {np.median(geop[band]):.3f}）")
print(f"""  → 读数:
    (a) 折叠是等距映射: UV/裁片参数距离 = 布料测地距离（解析构造 + 图测地独立验证）;
    (b) 3D 感受野被褶皱污染: 半径 r 的邻域里 {np.mean(frac_r):.0%} 的点布料距离超出 r、
        {np.mean(frac_2r):.0%} 超出 2r（最远 {np.max(max_geo)/r:.1f}×r, 隔着好几层布）——
        在 3D 空间做卷积/注意力, 感受野混入"空间上贴着、布料上很远"的点;
    (c) 3D 欧氏距离对布料距离排序能力弱（局部 ρ={rho_loc:.2f}）, 且带内混淆是本质的:
        3D 距离同落 [0.8g, 2.2g] 的点对, 布料距离铺满 {geop[band].max()/geop[band].min():.0f}× 全谱
        ——知道 3D 距离也定不了布料距离。
    pattern coordinates 的价值: 把布料的【内蕴度量】变成参数空间的【欧氏度量】
    ——卷积的局部性 = 布料的局部性, 遮挡/自接触/褶皱不再破坏邻域结构。
    ISP 的 UV 空间（GG 04）与 DMap 的 pattern coordinates 是同一机理的两次落地。
    对 Work 3: 视频先验给出的是逐像素【表面】对应——天然生活在内蕴坐标里,
    这是'视频 → 版型'蒸馏比'视频 → 3D mesh'更少一道有损变换的原因。""")

# ================= 实验 2：单视角零空间 + 伪 GT 幻觉结构 + 先验低通 =================
print("\n=== 实验 2: 单视角深度零空间 + 多视角伪 GT 的误差结构 + 先验低通 ===")

# 服装横截面（2D 闭合曲线）: 单位圆 + Fourier 位移场; 真值只在前 3 阶
n, K = 64, 16
phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
Bm = np.cos(np.outer(phi, np.arange(1, K + 1)))        # (n,K), 列正交: BᵀB = n/2·I
thx_t = np.zeros(K); thy_t = np.zeros(K)
thx_t[0] = 0.20; thx_t[2] = -0.10
thy_t[1] = 0.15; thy_t[2] = 0.08

def pts(thx, thy):
    return np.stack([np.cos(phi) + Bm @ thx, np.sin(phi) + Bm @ thy], 1)

def rms_err(thy_hat):
    return np.linalg.norm(pts(thx_t, thy_hat) - pts(thx_t, thy_t)) / np.sqrt(n)

# 正视图观测 x（精确, 所有方法共享）; 深度方向 y 不观测 = 零空间
# 伪 GT（多视角扩散幻觉的模型）: 低频偏差（模 2, 平滑但错）+ 高频噪声（模 10+）
g2 = np.random.default_rng(21)
bias_lf = np.zeros(K); bias_lf[1] = 0.15
hf = g2.normal(0, 0.06, K) * (np.arange(K) >= 9)
thy_pg = thy_t + bias_lf + hf
bias_hf = np.zeros(K); bias_hf[11] = 0.15              # 对照: 同幅度偏差放在高频
thy_pg2 = thy_t + bias_hf + hf
thy_pg3 = thy_t + hf                                   # 对照: 无偏差、只有噪声的"完美伪 GT"

LAM = 1e-3                                             # 强平滑先验 λ_k = λ0·k⁴
kmod = np.arange(1, K + 1)
trans = 1.0 / (1.0 + LAM * kmod ** 4)                  # 先验对伪 GT 的逐模传递函数

# 双视角真观测（对照上界）: 侧视图 y 带观测噪声 σ=0.01
y_obs = np.sin(phi) + Bm @ thy_t + g2.normal(0, 0.01, n)
thy_dual = (Bm.T @ (y_obs - np.sin(phi))) / (n / 2)

print(f"\n  (a) 估计器 = 伪 GT + 强平滑先验（λ_k = {LAM:g}·k⁴）, 逐模传递函数 1/(1+λk⁴):")
for k in [2, 4, 10, 16]:
    print(f"      mode {k:2d}: {trans[k-1]:.3f}   （{'低频放行' if trans[k-1] > 0.5 else '高频压制'}）")

rows2 = [
    ("单视角（深度零空间, 先验填 0）",   np.zeros(K),       "误差 = 真深度本身（一无所知）"),
    ("+ 伪 GT 直接信（无先验）",         thy_pg,           "低频偏差 + 高频噪声全进——幻觉比不知道更糟"),
    ("+ 伪 GT + 平滑先验（低通）",       thy_pg * trans,   "高频全压, 低频偏差 98% 原样放行"),
    ("对照: 偏差在高频（mode 12）",      thy_pg2 * trans,  "同幅度偏差被先验压到 ~5%"),
    ("对照: 完美伪 GT + 先验",           thy_pg3 * trans,  "先验自身代价 = 低频信号收缩, 很小"),
    ("双视角真观测（噪声 σ=0.01）",      thy_dual,         "零空间根治, 只剩观测噪声"),
]
print(f"\n  (b) 深度方向（y）RMS 误差（x 由正视图精确观测, 各方法相同）:")
print(f"  {'方法':32s} {'RMS':>7s}   误差构成")
for name, thy_hat, note in rows2:
    print(f"  {name:34s} {rms_err(thy_hat):7.4f}   {note}")
print(f"""  → 读数:
    单视角观测对深度方向是【盲】的（零空间）——先验只能填 0, 误差 = 真深度;
    伪 GT 用幻觉填零空间, 误差结构 = 低频有偏（平滑、plausible、但错）
    + 高频噪声。平滑先验/物理正则是【低通滤波器】: 滤的是频率,
    不是对错——高频噪声和高频偏差都被压制, 低频偏差 {trans[1]:.0%} 原样放行。
    伪 GT 的价值必须靠先验兑现（第三行 < 第一行）, 且只到低频偏差为止;
    真第二视角才根治零空间（末行）。
    Dress-1-to-3 式流程 = 伪 GT（多视角扩散 MagicMan, 低频有偏）+ 可微物理
    精修（仿真 + 平滑正则）——本实验给出其误差结构: 高频幻觉物理投影能治
    （论文自己的 Limitation 3"表面过度平滑"就是低通的实证——真高频褶皱
    也一起被压）, 低频系统性偏差只有真观测或多视角一致性约束能治。
    对 Work 3: 视频先验 = 时序多视角真观测——恰好压的是低频偏差这项。""")

# ================= 实验 3：准静态映射 vs 动力学 =================
print("\n=== 实验 3: 准静态 pose→形状映射 vs 动力学 —— 频响与修正边界 ===")

# 质量-弹簧-阻尼: 锚点 a(t)=A·sin(ωt)（= 身体/pose 驱动）, 服装 = 惯性质量 m
# 准静态映射（DrapeNet/DMap-静态式 pose→形状）: y_qs = a
# 约定: a(t) = Im[A e^{iωt}] → 稳态 y 复振幅 = H(ω)·A, H = k/(k−mω²+icω)
m_, kk_, cc = 1.0, 100.0, 2.0                          # ω0 = 10 rad/s, ζ = 0.1
def H(omega):
    return kk_ / (kk_ - m_ * omega ** 2 + 1j * cc * omega)

def steady(omega, A=1.0):
    """ODE 数值稳态 → 复振幅（y = Im[Ĥ·A·e^{iωt}], Ĥ = (p+iq)/A）"""
    T = 2 * np.pi / omega
    f = lambda t, s: [s[1], -kk_ * (s[0] - A * np.sin(omega * t)) - cc * s[1]]
    sol = solve_ivp(f, [0, 40 * T], [0.0, 0.0], t_eval=np.linspace(30 * T, 40 * T, 4001),
                    rtol=1e-9, atol=1e-11)
    t, y = sol.t, sol.y[0]
    Amat = np.stack([np.sin(omega * t), np.cos(omega * t)], 1)
    p, q = np.linalg.lstsq(Amat, y, rcond=None)[0]
    return (p + 1j * q) / A

# (a) 数值 vs 解析频响验证
print(f"\n  (a) ODE 数值稳态 vs 解析频响 H(ω) = k/(k−mω²+icω):")
om_chk = [1.0, 3.0, 5.0, 7.0]
err_h = [abs(steady(om) - H(om)) / abs(H(om)) for om in om_chk]
print(f"      ω = {om_chk}: 相对误差最大 {max(err_h):.1e}（解析与数值一致）")

# (b) 速度/加速度一阶修正: 数据拟合恢复动力学系数
# y ≈ a + α1·ȧ + α2·ä → 复振幅 (Ĥ−1)·A = iωα1·A − ω²α2·A
om_tr = [1.0, 1.5, 2.0]
rowsA, yb = [], []
for om in om_tr:
    dh = steady(om) - 1.0
    rowsA.append([om, 0.0]);  yb.append(dh.imag)      # ω·α1 = Im(Ĥ−1)
    rowsA.append([0.0, -om ** 2]); yb.append(dh.real) # −ω²·α2 = Re(Ĥ−1)
a1, a2 = np.linalg.lstsq(np.array(rowsA), np.array(yb), rcond=None)[0]
print(f"\n  (b) 从训练频率 {om_tr} 的数据拟合 y ≈ a + α1·ȧ + α2·ä:")
print(f"      α1 = {a1:+.4f}（解析 −c/k = {-cc/kk_:+.4f}）, "
      f"α2 = {a2:+.4f}（解析 −m/k = {-m_/kk_:+.4f}）")

# (c) 误差 vs 频率: 准静态 / +速度 / +速度+加速度
w0 = np.sqrt(kk_ / m_)
print(f"\n  (c) 稳态 RMS 误差 / A（归一化; ω0 = √(k/m) = {w0:.0f} rad/s, ζ = 0.1）:")
print(f"  {'ω':>4s} {'ω/ω0':>5s} {'准静态 |H−1|':>12s} {'+速度':>8s} {'+速度+加速度':>12s}")
for om in [2.0, 4.0, 6.0, 8.0, 10.0]:
    hh = H(om)
    print(f"  {om:4.1f} {om/w0:5.2f} {abs(hh-1):12.3f} "
          f"{abs(hh-(1+1j*om*a1)):8.3f} {abs(hh-(1+1j*om*a1-om**2*a2)):12.3f}")
print(f"""  → 读数:
    准静态映射 y=a 的稳态误差 = |H(ω)−1|, 随驱动频率单调增长（ω=0.6ω0 时
    误差已达激励幅值的 ~58%）——忽略时间维的两种代价都是这个形状:
    GG 01/04 的准静态 draping 丢惯性项; DMap 静态版逐帧独立用于视频
    → 抖动（TPAMI Table III: 顶点加速度 18.2 vs GT 3.7 mm/frame²）;
    速度+加速度一阶修正在低频段吃掉大部分缺口（ω=0.4ω0: 0.21→0.03）,
    且系数由数据拟合自动恢复动力学参数（α1→−c/k, α2→−m/k, 训练频率
    ≤0.2ω0 时二阶污染 <5%）——时序条件化（速度/历史帧）在原理上够覆盖
    日常运动的频段, 这就是 DMap 动态版时序模块在学的东西;
    共振段（ω→ω0）三列一起爆, 修正项反而放大误差（泰勒展开发散）——
    共振必须真动力学。这正是 'simulation-ready'（本篇主题）存在的理由:
    学习映射终究有不覆盖的频段, 生成的版型要能进仿真器兜底。""")
