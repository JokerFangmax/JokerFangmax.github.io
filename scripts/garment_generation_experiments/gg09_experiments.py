"""GG 09 数值实验：版型表示三条新路线的机理验证（numpy/scipy，无学习框架）
1) 栅格 vs 序列的组合泛化：pairwise 交互（缝合）在栅格空间是【局部共享规则】
   （可从任意 partner 对学到、对 unseen 组合精确外推），在序列空间是【全局配对表】
   （unseen 组合 = 表上缺行 → 交互项不可预测）——GarmentImage unseen IoU 0.81
   vs 序列 0.31/0.43 的机理还原
2) 隐式场的拓扑平滑插值 + 可微网格化：SDF 插值路径上拓扑经鞍点连续变化
   （IoU(t) 平滑）vs 序列 padding 的离散跳变；零水平集点对参数的导数用
   隐函数定理解析给出，有限差分验证（Latent FM 的 iso-surface gradient
   ∂v/∂z = −∇d_c·∂d_c/∂z 的标量版机理）
3) DSL 的 valid-by-construction：随机程序 100% 有效（闭合由构造保证）+
   程序空间组合爆炸计数（学分布永远追不上写规则——GarmentoPIA 的动机）
"""
import numpy as np

rng = np.random.default_rng(9)

# ================= 实验 1：栅格 vs 序列的组合泛化 =================
print("=== 实验 1: pairwise 缝合交互 —— 局部规则（栅格） vs 配对表（序列） ===")

# 玩具世界: 6 种 panel，服装 = 左右相邻两 panel 的组合（16 种有序对, 15 种无序对）
# 每 panel 有 8 行的边缘轮廓: presence bit + 朝向类 (0..3)
P = 6
ROWS = 8
DIR_ANG = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])   # 朝向 → 角度

g0 = np.random.default_rng(11)
# 每 panel 的左/右边缘轮廓（随机但固定）
pres_L = (g0.random((P, ROWS)) > 0.25).astype(float)
pres_R = (g0.random((P, ROWS)) > 0.25).astype(float)
dir_L = g0.integers(0, 4, (P, ROWS))
dir_R = g0.integers(0, 4, (P, ROWS))

# ---- 栅格空间的缝合通道（GarmentImage 式: 交互在空间上局部化）----
# 服装 (p,q) 的边界列上, 行 y 的 stitch 通道值 = φ(dir_R[p,y], dir_L[q,y])
# φ 是【与 panel 身份无关的局部规则】—— 任何 pair 共享同一个 φ
def phi(a_cls, b_cls):
    return np.cos(DIR_ANG[a_cls] - DIR_ANG[b_cls])

def grid_seam(p, q):
    """边界列 8 行的 stitch 通道: 双方边缘都 present 才有缝合"""
    both = pres_R[p] * pres_L[q]
    return both * phi(dir_R[p], dir_L[q])

# ---- 序列空间的 stitch tag（NeuralTailor/DressCode 式: 全局 3D 标签）----
# tag 挂在 panel 的 token 上, 值 = f(两侧 panel 的 3D 放置) —— 含 pair 专属交互项
g1 = np.random.default_rng(12)
ALPHA = g1.normal(0, 0.3, P)          # panel p 的自身贡献
BETA = g1.normal(0, 0.3, P)
GAMMA = {}                             # pair 交互项（配对表）
for p in range(P):
    for q in range(p, P):
        GAMMA[(p, q)] = g1.normal(0, 0.5)

def seq_tag(p, q):
    """panel p 在 pair (p,q) 中的 stitch tag（3D 缝合中点位置的标量化）"""
    k = (min(p, q), max(p, q))
    return ALPHA[p] + BETA[q] + GAMMA[k]

# ---- 训练/测试划分: 留出 pair (0,5) 作为 unseen 组合 ----
TEST_PAIR = (0, 5)
all_pairs = [(p, q) for p in range(P) for q in range(p, P)]
train_pairs = [pr for pr in all_pairs if pr != TEST_PAIR]

# ---- 栅格侧学习器: 从 seen pairs 学 φ（4×4 朝向交互表）----
# φ 本质是【朝向对】的表: 特征 = onehot(a) ⊗ onehot(b) 16 维。
# 关键: 表的参数空间是朝向（4×4=16 格）, 不是 panel 身份（6×6=21 对）——
# 每个 seen pair 为同一张表提供样本 → unseen pair 复用
X_g, y_g = [], []
for p, q in train_pairs:
    val = grid_seam(p, q)
    for y in range(ROWS):
        if pres_R[p, y] and pres_L[q, y]:
            f = np.zeros(16)
            f[dir_R[p, y] * 4 + dir_L[q, y]] = 1.0
            X_g.append(f)
            y_g.append(val[y])
X_g = np.array(X_g); y_g = np.array(y_g)
W_g = np.linalg.lstsq(X_g, y_g, rcond=None)[0]

def grid_predict_seam(p, q):
    out = np.zeros(ROWS)
    for y in range(ROWS):
        if pres_R[p, y] and pres_L[q, y]:
            f = np.zeros(16)
            f[dir_R[p, y] * 4 + dir_L[q, y]] = 1.0
            out[y] = f @ W_g
    return out

# ---- 序列侧学习器: 从 seen pairs 学 tag 的加性近似（panel 身份 one-hot）----
# t(p,q) ≈ a_p + b_q; 交互项 γ 是【配对表】, unseen pair 的 γ 不可学
X_s, y_s = [], []
for p, q in train_pairs:
    X_s.append(np.concatenate([np.eye(P)[p], np.eye(P)[q]]))
    y_s.append(seq_tag(p, q))
X_s = np.array(X_s); y_s = np.array(y_s)
W_s = np.linalg.lstsq(X_s, y_s, rcond=None)[0]

def seq_predict_tag(p, q):
    return np.concatenate([np.eye(P)[p], np.eye(P)[q]]) @ W_s

# ---- 评测: seen pairs（两者都应准） vs unseen pair (0,5) ----
def err_pct(pred, true, mask=None):
    m = mask if mask is not None else np.ones_like(true, dtype=bool)
    if m.sum() == 0:
        return 0.0, 0.0
    e = np.abs(pred[m] - true[m])
    scale = np.abs(true[m]).mean() + 1e-9
    return e.mean(), e.mean() / scale

print(f"\n  {P} 种 panel, {len(all_pairs)} 个组合, 留出 ({TEST_PAIR[0]},{TEST_PAIR[1]}) 作 unseen")
print(f"  栅格 φ 共享规则训练样本 {len(y_g)} 行, 序列 tag 配对表训练样本 {len(y_s)} 对\n")

# seen 泛化误差（所有训练 pair 的平均）
g_err_seen = []
s_err_seen = []
for p, q in train_pairs:
    m = (pres_R[p] * pres_L[q]) > 0
    e, _ = err_pct(grid_predict_seam(p, q), grid_seam(p, q), m)
    g_err_seen.append(e)
    e, _ = err_pct(np.full(ROWS, seq_predict_tag(p, q)), np.full(ROWS, seq_tag(p, q)))
    s_err_seen.append(abs(seq_predict_tag(p, q) - seq_tag(p, q)))
print(f"  {'':12s} {'栅格（局部规则 φ）':>18s} {'序列（配对表 tag）':>18s}")
print(f"  {'seen 组合':12s} {np.mean(g_err_seen):16.4f} {np.mean(s_err_seen):16.4f}   （两者都能拟合训练分布）")

# unseen pair
p, q = TEST_PAIR
m = (pres_R[p] * pres_L[q]) > 0
g_pred = grid_predict_seam(p, q)
g_true = grid_seam(p, q)
g_e_un, g_rel = err_pct(g_pred, g_true, m)
s_pred, s_true = seq_predict_tag(p, q), seq_tag(p, q)
s_e_un = abs(s_pred - s_true)
gamma_scale = abs(GAMMA[TEST_PAIR])
print(f"  {'unseen 组合':12s} {g_e_un:16.4f} {s_e_un:16.4f}   （序列误差 ≈ 交互项 γ={gamma_scale:.3f} 的量级）")

# 端到端 IoU 式汇总: 服装 = 两 panel 占用 ∪ 缝合通道
# 占用部分两种表示都精确（panel 各自内容与 partner 无关）
# 只有缝合交互不同 → unseen 组合的整版误差被缝合交互主导
occ_frac = 0.9   # 版型张量里占用类通道占比（缝合通道是少数但决定可制造性）
grid_acc = 1 - (1 - occ_frac) * g_rel if g_rel < 1 else 1 - (1 - occ_frac)
seq_rel = s_e_un / (abs(s_true) + 1e-9)
seq_acc = max(0.0, 1 - (1 - occ_frac) * seq_rel)
print(f"\n  端到端（占用 90% 精确 + 缝合通道 10% 由交互决定）:")
print(f"    栅格 unseen 组合准确率 ~{grid_acc:.3f}   序列 unseen 组合准确率 ~{seq_acc:.3f}")
print(f"""  → 机理读数:
    同一个 pairwise 缝合交互, 在两种表示里的【可学结构】不同:
    · 栅格: 缝合通道落在两 panel 的空间边界上, 值 = φ(两侧边缘朝向) ——
      φ 与 panel 身份无关, 任何 seen pair 都提供 φ 的样本 → unseen 组合
      (0,5) 的缝合通道被【精确外推】(误差 {g_e_un:.4f} ≈ 0);
    · 序列: stitch tag 挂在 panel token 上, 值含 pair 专属交互项 γ_pq ——
      16 对各一个 γ, 是配对表不是共享规则; unseen pair 的 γ 从训练数据
      【原理上不可预测】, 加性最优预测的误差 = |γ| = {gamma_scale:.3f} 量级
      (实测 {s_e_un:.3f})。
    '组合方式'这个自由度: 栅格把它空间化为局部邻接规则（O(P) 个模式）,
    序列把它摊在全局 token 相关性上（O(P²) 个模式）——unseen 组合数
    占总组合数的比例随 P 增长, 这就是 GarmentImage unseen IoU 0.81
    vs NeuralTailor 0.31 / SewFormer 0.43 的机理（GG 02 §5.3 的数字）。""")

# ================= 实验 2：隐式场拓扑平滑插值 + 可微网格化 =================
print("\n=== 实验 2: SDF 隐式场的拓扑平滑性 + 可微网格化（隐函数定理） ===")

# ---- (a) 两个版型之间的插值: T 恤前片（1 大 panel）→ 双 panel（左右襟） ----
# SDF 表示: 版型 = Ω 上的符号距离场（panel 边界 = 零水平集）
def sd_rect(px, py, hw, hh):
    """2D box SDF（标准精确距离）"""
    dx = np.abs(px) - hw
    dy = np.abs(py) - hh
    outside = np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2)
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside

def sdf_shirt(x, y, s=1.0):
    """单连通 T 恤前片: 矩形 ∪ 两侧肩袖（SDF 的光滑并）"""
    body = sd_rect(x, y, 0.35 * s, 0.5 * s)
    lsleeve = sd_rect(x + 0.55 * s, y - 0.35 * s, 0.18 * s, 0.12 * s)
    rsleeve = sd_rect(x - 0.55 * s, y - 0.35 * s, 0.18 * s, 0.12 * s)
    # smooth union（取小者; b≫a 时 h→1 → 结果→a）—— iq 的 smin: h=clamp(.5+.5(b-a)/k), mix(b,a,h)
    def sunion(a, b, k=0.08):
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
        return (1 - h) * b + h * a
    return sunion(sunion(body, lsleeve), rsleeve)

def sdf_doublet(x, y, s=1.0):
    """双 panel（左右襟, 中缝分开）: 两矩形, 中间有缝"""
    left = sd_rect(x + 0.18, y, 0.16 * s, 0.5 * s)
    right = sd_rect(x - 0.18, y, 0.16 * s, 0.5 * s)
    return np.minimum(left, right)

# 序列表示的插值（对照）: panel 数 1 vs 2 —— presence flag / padding 槽
# 中间态 t∈(0,1): "半个 panel"? padding 值线性插值 = 无效版型
def seq_validity(t):
    """序列空间插值的有效性: presence flag p(t) = t 阈值化才有意义,
    线性插值 t∈(0,1) 给出非 0 非 1 的 flag → 序列化器无法解码"""
    return 1.0 if t in (0.0, 1.0) else 0.0

N = 200
xs = np.linspace(-1, 1, N)
XX, YY = np.meshgrid(xs, xs)

def sdf_mix(t, s=1.0):
    return (1 - t) * sdf_shirt(XX, YY, s) + t * sdf_doublet(XX, YY, s)

def marching_cells(F):
    """零水平集所在 cell 的占比（拓扑体积的离散代理）"""
    return (F[:-1, :-1] * F[1:, 1:] < 0) | (F[:-1, 1:] * F[1:, :-1] < 0)

def iou(F1, F2):
    m1, m2 = F1 < 0, F2 < 0
    return (m1 & m2).sum() / max((m1 | m2).sum(), 1)

print(f"\n  (a) T 恤前片（1 panel, 单连通） → 左右襟（2 panel, 中缝拓扑分裂）:")
print(f"  {'t':>4s} {'SDF 内部占比':>10s} {'零集 cell 数':>10s} {'与 t=0 的 IoU':>11s}")
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    F = sdf_mix(t)
    print(f"  {t:4.2f} {(F < 0).mean():10.3f} {marching_cells(F).sum():10d} {iou(F, sdf_mix(0.0)):11.3f}")
F_prev = sdf_mix(0.0)
min_iou = 1.0
for t in np.linspace(0, 1, 101)[1:]:
    F = sdf_mix(t)
    min_iou = min(min_iou, iou(F, F_prev))
    F_prev = F
print(f"  相邻插值步（Δt=0.01）的 IoU 最小值 = {min_iou:.3f}（连续: 拓扑分裂经鞍点渐变）")

# ---- (b) 可微网格化: 零点位置对参数 θ 的导数（隐函数定理）----
# 1D 切片: s(x; θ) = SDF 在 y=y0 行的值, θ 控制版型尺寸
# 零点 x*(θ): s(x*; θ)=0 → dx*/dθ = −(∂s/∂θ)/(∂s/∂x)
print(f"\n  (b) 可微网格化: 零点 x*(θ) 对参数 θ 的梯度")
y0 = 0.0

def sdf_slice(x, theta):
    """参数化切片: s(x; θ) = 混合场在 (x, y0) 处, θ = (插值参数 t, 尺寸 s)"""
    t, s = theta
    x = np.asarray(x, dtype=float)
    return (1 - t) * sdf_shirt(x, np.full_like(x, y0), s) + t * sdf_doublet(x, np.full_like(x, y0), s)

theta0 = np.array([0.3, 1.0])
from scipy.optimize import brentq
x_star = brentq(lambda x: sdf_slice(x, theta0), 0.2, 0.45)

def zero_crossing(theta):
    return brentq(lambda x: sdf_slice(x, theta), 0.2, 0.45)

# 解析梯度（隐函数定理, 中心差分求偏导）
h = 1e-5
dsdtheta = np.array([
    (sdf_slice(x_star, theta0 + np.array([h, 0])) - sdf_slice(x_star, theta0 - np.array([h, 0]))) / (2 * h),
    (sdf_slice(x_star, theta0 + np.array([0, h])) - sdf_slice(x_star, theta0 - np.array([0, h]))) / (2 * h),
])
dsdx = (sdf_slice(x_star + h, theta0) - sdf_slice(x_star - h, theta0)) / (2 * h)
grad_analytic = -dsdtheta / dsdx

# 有限差分梯度（"数值梯度 check"）
grad_fd = np.zeros(2)
for i in range(2):
    d = np.zeros(2); d[i] = h
    grad_fd[i] = (zero_crossing(theta0 + d) - zero_crossing(theta0 - d)) / (2 * h)

print(f"    零点 x* = {x_star:.6f}（右襟外边界）")
print(f"    隐函数定理 dx*/dθ = [{grad_analytic[0]:+.6f}, {grad_analytic[1]:+.6f}]")
print(f"    有限差分           = [{grad_fd[0]:+.6f}, {grad_fd[1]:+.6f}]")
print(f"    相对误差           = {np.abs(grad_analytic - grad_fd).max() / np.abs(grad_fd).max():.2e}")
print(f"""  → 读数:
    (a) SDF 插值路径上拓扑变化经【鞍点】连续发生（相邻步 IoU ≥ {min_iou:.2f}）:
        1 panel → 2 panel 的中缝是场从鞍点分裂出来的, 不需要任何离散决策;
        序列表示里同样的 morph 需要 presence flag 跳变（中间态不可解码）——
        隐式场把'拓扑'变成了场的连续几何。
    (b) 零水平集点的位置对参数可微, 梯度由隐函数定理解析给出
        （数值验证 {np.abs(grad_analytic - grad_fd).max() / np.abs(grad_fd).max():.0e} 量级吻合）——
        即 LFM 论文的 iso-surface gradient ∂v/∂z 机理:
        网格化不是梯度的死胡同, 这是 latent flow matching 能把'生成 → 网格 →
        损失'整条链端到端可微的基础（GG 04 ISP 的损失家族同理）。""")

# ================= 实验 3：DSL 的 valid-by-construction + 组合爆炸 =================
print("\n=== 实验 3: 程序合成路线 —— 有效性由构造保证, 组合空间指数爆炸 ===")

# ---- (a) 玩具 GarmentCode 式 DSL: 闭合由构造保证 ----
# 组件库: body/sleeve/skirt/collar/cuff/pocket 六类, 每类内 4 个变体
# 程序 = 组件树的嵌套（depth ≤ 3）, 求值 = 边环拼接
COMPONENTS = ['body', 'sleeve', 'skirt', 'collar', 'cuff', 'pocket']
VARIANTS = 4
g3 = np.random.default_rng(31)

def eval_program(prog):
    """求值玩具 DSL: 每 component 贡献一个闭合边环（顶点数 8-20, 随机但由
    程序+种子决定）。返回全部边环——闭合性由'每环由顶点列表+回卷'构造保证"""
    loops = []
    for comp, variant in prog:
        r = np.random.default_rng(hash((comp, variant)) % (2**32))
        n = r.integers(8, 21)
        ang = np.sort(r.uniform(0, 2 * np.pi, n))
        rad = 1 + 0.4 * r.random(n)
        loops.append(np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1))
    return loops

def closure_residuals(loops):
    """显式边序列的闭合残差（对照 GG 07 实验 2 的量化口径）"""
    res = []
    for V in loops:
        E = np.roll(V, -1, axis=0) - V
        res.append(np.linalg.norm(E.sum(0)))
    return np.array(res)

def random_program(g, max_depth=3):
    n_comp = g.integers(1, max_depth + 1)
    return [(COMPONENTS[g.integers(0, len(COMPONENTS))], g.integers(0, VARIANTS))
            for _ in range(n_comp)]

n_prog = 2000
progs = [random_program(g3) for _ in range(n_prog)]
res_all = np.concatenate([closure_residuals(eval_program(p)) for p in progs])
print(f"\n  (a) {n_prog} 个随机程序求值 → {len(res_all)} 个边环:")
print(f"      闭合残差: max = {res_all.max():.2e}（IEEE 双精度级别, 构造即闭合）")
print(f"      对照 GG 07 实验 2: 量化 token 表示 h=20mm 时 19/20 版型闭合超差")
print(f"      → DSL 求值【永不产生无效版型】: 有效性是类型系统性质, 不是学习目标")

# ---- (b) 程序空间计数: 组件 k 种 × 变体 v × 嵌套深度 d 的组合数 ----
print(f"\n  (b) 程序空间大小（{len(COMPONENTS)} 组件 × {VARIANTS} 变体, 长度 1..d 的程序数）:")
for d in [2, 3, 4, 5]:
    n_space = sum((len(COMPONENTS) * VARIANTS) ** L for L in range(1, d + 1))
    print(f"      depth ≤ {d}: {n_space:>12,} 个程序   （GarmentCodeData 全量 115,195 件"
          f"{' << 覆盖率 {:.3f}%'.format(115195 / n_space * 100) if n_space > 115195 else ' 覆盖全部'}）")
print(f"""  → 读数:
    学习路线（GG 07/08）学的是【程序空间的投影】——115k 样本对 depth≤4 的
    程序空间覆盖率 ~{115195 / sum((len(COMPONENTS)*VARIANTS)**L for L in range(1,5)) * 100:.2f}%
    （真实 GarmentCode 组件库远大于本玩具, 覆盖率更低）;
    程序求值器对【任意】程序输出有效版型——泛化性来自求值规则而不是数据分布。
    GarmentoPIA 的分工 = LLM 提议程序（语义先验: 懂'a hoodie dress'指什么）
    + DSL 求值保证有效性（语法先验）。学习路线与程序路线不是竞争:
    前者有分布内的平滑插值（GG 08 FSQ）, 后者有分布外的组合泛化
    （本实验 (a) 的 100% 有效 + 组合计数）——GG 12 的谱系总结会回到这一点。""")
