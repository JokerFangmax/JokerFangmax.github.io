"""GG 02 数值实验：版型曲线表达力 / 缝合匹配歧义 / 坐标量化误差"""
import numpy as np

rng = np.random.default_rng(0)

# ============ 合成 T 恤版型（前片+后片+双袖，直线边+曲边混合） ============
def bezier(p0, p1, p2, p3, n=64):
    t = np.linspace(0, 1, n)[:, None]
    return ((1-t)**3)*p0 + 3*((1-t)**2)*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3

def edge_len(pts):
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

# 前片（单位: 米，直裁片的典型画法）：衣身轮廓 12 条边
# 关键点: 肩点→领口→肩点→袖窿→侧缝下摆→下摆→...
front_outline = {
    'vertices': np.array([
        [0.20, 0.72],   # 左肩点 (领口左)
        [0.30, 0.78],   # 左肩外端
        [0.48, 0.70],   # 左袖窿上
        [0.50, 0.45],   # 左袖窿底/侧缝上
        [0.48, 0.00],   # 左下摆
        [-0.48, 0.00],  # 右下摆
        [-0.50, 0.45],  # 右侧缝
        [-0.48, 0.70],  # 右袖窿上
        [-0.30, 0.78],  # 右肩外端
        [-0.20, 0.72],  # 右肩点 (领口右)
    ]),
    'curved': {0: 'neckline_half', 2: 'armhole_l', 7: 'armhole_r'},  # 顶点序号 -> 曲线段
}
# 领口（前）：从 (-0.20,0.72) 经 (-0.30,0.74)? 不对——领口在前中凹陷
# 简化处理: 领口曲线从左肩点 (0.20,0.72) 到右肩点 (-0.20,0.72)，中间下垂到 (0, 0.62)
neckline_front = bezier(np.array([0.20,0.72]), np.array([0.12,0.60]),
                        np.array([-0.12,0.60]), np.array([-0.20,0.72]))
# 后片领口浅一些
neckline_back = bezier(np.array([0.20,0.72]), np.array([0.10,0.66]),
                       np.array([-0.10,0.66]), np.array([-0.20,0.72]))
# 袖窿（前片左）：从 (0.48,0.70) 内凹到 (0.50,0.45)
armhole = bezier(np.array([0.48,0.70]), np.array([0.56,0.58]),
                 np.array([0.56,0.50]), np.array([0.50,0.45]))

# 统计合成版型的规模
n_panels = 4   # 前后身(2) + 双袖(2)
n_edges_body = 10  # 前片轮廓边数（12 顶点简化为 10 边环）
n_edges_sleeve = 4
total_edges = 2*n_edges_body + 2*n_edges_sleeve
print("=== 合成 T 恤版型规模 ===")
print(f"  panels: {n_panels}, 边总数: {total_edges} (身片 {2*n_edges_body} + 袖片 {2*n_edges_sleeve})")
print(f"  领口前弧长: {edge_len(neckline_front):.3f} m, 后弧长: {edge_len(neckline_back):.3f} m")
print(f"  袖窿弧长: {edge_len(armhole):.3f} m")

# ============ 实验 1：曲线表达力（每参数 expressiveness） ============
# 任务: 用不同参数化逼近前领口曲线（不对称测试用真实"前中深、一侧略偏"的曲线）
# 目标曲线: 幂函数领口 x^(2/3)（前中下垂型）+ 轻微不对称
t = np.linspace(0, 1, 200)
target = np.stack([0.2 - 0.4*t, 0.72 - 0.12*np.sin(np.pi*t)*(0.9 + 0.2*t)], axis=1)
print(f"\n=== 实验 1: 曲线表达力（目标=不对称领口, 弧长 {edge_len(target):.3f} m）===")

def fit_error(pred):
    # 双向 chamfer 距离（采样对齐）
    d1 = np.min(np.linalg.norm(target[:,None,:] - pred[None,:,:], axis=2), axis=1)
    d2 = np.min(np.linalg.norm(pred[:,None,:] - target[None,:,:], axis=2), axis=1)
    return float(np.mean(np.concatenate([d1, d2]))), float(np.max(np.concatenate([d1, d2])))

# (a) 单条直线段（2 参数: 端点已由邻边给定 → 实际 0 自由参数）
pred = np.stack([np.linspace(0.2, -0.2, 50), np.full(50, 0.72)], axis=1)
e = fit_error(pred); print(f"  (a) 直线段        [0 自由参数]: mean {e[0]*1000:6.2f} mm, max {e[1]*1000:6.2f} mm")

# (b) 对称二次 Bezier：NeuralTailor 式（1 自由参数: 对称鼓起量 h，端点切线固定水平）
from scipy.optimize import minimize_scalar
def sym_bezier_err(h):
    p0, p2 = target[0], target[-1]
    p1 = (p0+p2)/2 + np.array([0, -h])
    tt = np.linspace(0,1,100)
    pred = ((1-tt)**2)[:,None]*p0 + 2*((1-tt)*tt)[:,None]*p1 + (tt**2)[:,None]*p2
    return fit_error(pred)[0]
res = minimize_scalar(sym_bezier_err, bounds=(0, 0.5), method='bounded')
h = res.x
p0, p2 = target[0], target[-1]
p1 = (p0+p2)/2 + np.array([0, -h])
tt = np.linspace(0,1,100)
pred = ((1-tt)**2)[:,None]*p0 + 2*((1-tt)*tt)[:,None]*p1 + (tt**2)[:,None]*p2
e = fit_error(pred); print(f"  (b) 对称二次 Bezier [1 自由参数 h={h:.3f}]: mean {e[0]*1000:6.2f} mm, max {e[1]*1000:6.2f} mm")

# (c) 一般二次 Bezier（2 自由参数: 控制点 2 坐标，端点固定）
def bezier2(cp):
    p0, p2 = target[0], target[-1]
    tt = np.linspace(0,1,100)
    return ((1-tt)**2)[:,None]*p0 + 2*((1-tt)*tt)[:,None]*cp + (tt**2)[:,None]*p2
def err2(cp):
    return fit_error(bezier2(cp))[0]
from scipy.optimize import minimize
res = minimize(err2, np.array([0.0, 0.60]), method='Nelder-Mead')
pred = bezier2(res.x)
e = fit_error(pred); print(f"  (c) 一般二次 Bezier [2 自由参数 cp=({res.x[0]:.3f},{res.x[1]:.3f})]: mean {e[0]*1000:6.2f} mm, max {e[1]*1000:6.2f} mm")

# (d) 三次 Bezier（4 自由参数: 两控制点，端点固定）—— SewFormer 式
def bezier3(cps):
    p0, p3 = target[0], target[-1]
    tt = np.linspace(0,1,100)
    return ((1-tt)**3)[:,None]*p0 + 3*(((1-tt)**2)*tt)[:,None]*cps[:2] + 3*((1-tt)*(tt**2))[:,None]*cps[2:4] + (tt**3)[:,None]*p3
def err3(cps):
    return fit_error(bezier3(cps))[0]
res = minimize(err3, np.array([0.12,0.62, -0.12,0.60]), method='Nelder-Mead',
               options={'xatol':1e-6,'fatol':1e-9,'maxiter':2000})
pred = bezier3(res.x)
e = fit_error(pred); print(f"  (d) 三次 Bezier    [4 自由参数]: mean {e[0]*1000:6.2f} mm, max {e[1]*1000:6.2f} mm")

# (e) 折线逼近（2N 参数: N 段的中点偏移）
print("  (e) 折线逼近（每段中点 1 法向偏移参数）:")
for nseg in [2, 4, 8]:
    ts = np.linspace(0, 1, nseg+1)
    anchors = np.stack([np.interp(ts, t, target[:,0]), np.interp(ts, t, target[:,1])], axis=1)
    # 中点放采样点上（上界误差估计）
    mids_t = (ts[:-1]+ts[1:])/2
    mids = np.stack([np.interp(mids_t, t, target[:,0]), np.interp(mids_t, t, target[:,1])], axis=1)
    poly = np.empty((2*nseg+1, 2)); poly[0::2] = anchors; poly[1::2] = mids
    e = fit_error(poly)
    print(f"      N={nseg} 段 [{nseg} 参数]: mean {e[0]*1000:6.2f} mm, max {e[1]*1000:6.2f} mm")

# ============ 实验 2：缝合匹配（贪心 vs 匈牙利 + 对称歧义） ============
print("\n=== 实验 2: 缝合匹配 ===")
# 把版型"缝起来"：对每对被缝合的边，把两边都弯到同一空间圆弧上取 3D 位置
def wrap_to_arc(pts2d, R=0.15):
    """把 2D 边弯成半径 R 的圆弧（模拟缝合后的空间形态）"""
    L = edge_len(pts2d)
    phi = np.linspace(-L/(2*R), L/(2*R), len(pts2d))
    x = R*np.sin(phi); z = R*(np.cos(phi)-1)
    y = pts2d[:, 1]  # 沿边走向保持
    return np.stack([x, y, z], axis=1)

# 构造 5 对真缝合边: (前片边, 后片边)
true_pairs = []
for i in range(5):
    a = bezier(np.array([0,0.1*i]), np.array([0.05,0.1*i+0.05]),
               np.array([-0.05,0.1*i+0.05]), np.array([0,0.1*i+0.1]))
    b = a + rng.normal(0, 0.002, a.shape)  # 轻噪声
    true_pairs.append((wrap_to_arc(a), wrap_to_arc(b)))

edgesA = [p[0] for p in true_pairs]
edgesB = [p[1] for p in true_pairs]
# 加一对"假边"（不该配对）：前片一条离群边
edgesA.append(wrap_to_arc(bezier(np.array([0.3,0.5]), np.array([0.35,0.55]),
                                  np.array([0.25,0.55]), np.array([0.3,0.6]))))

# 边特征: 中点位置 + 法向（平面拟合）
def edge_feat(pts):
    mid = pts.mean(axis=0)
    # 边的切向
    tan = pts[-1] - pts[0]; tan /= np.linalg.norm(tan)
    return mid, tan

def greedy_match(edgesA, edgesB, tau):
    feasA = [edge_feat(e) for e in edgesA]
    feasB = [edge_feat(e) for e in edgesB]
    matches = []
    used = set()
    for i in range(len(edgesA)):
        cands = []
        for j in range(len(edgesB)):
            if j in used: continue
            d = np.linalg.norm(feasA[i][0] - feasB[j][0])
            if d < tau:
                cands.append((d, j))
        if cands:
            d, j = min(cands)
            # 互最近检查
            back = min((np.linalg.norm(feasA[k][0]-feasB[j][0]), k) for k in range(len(edgesA)) if k not in [m[0] for m in matches])
            if back[1] == i:
                matches.append((i, j)); used.add(j)
    return matches

matches = greedy_match(edgesA, edgesB, tau=0.05)
correct = sum(1 for i,j in matches if i == j and i < 5)
print(f"  5 真对 + 1 离群假边, 贪心互最近匹配: {len(matches)} 对, 正确 {correct}/5, "
      f"假边误配 {'有' if any(i==5 for i,j in matches) else '无'}")

# 对称歧义案例: 镜像对称服装里, 左/右袖边到身体袖窿边的距离完全相同 → 纯几何匹配失效
armhole_q = wrap_to_arc(bezier(np.array([0,0]), np.array([0.04,0.03]), np.array([-0.04,0.03]), np.array([0,0.06])))
sleeve_L = armhole_q + np.array([ 0.01, 0, 0.005])   # 左袖候选边
sleeve_R = armhole_q + np.array([-0.01, 0, 0.005])   # 右袖候选边（镜像）
dL = np.linalg.norm(edge_feat(armhole_q)[0] - edge_feat(sleeve_L)[0])
dR = np.linalg.norm(edge_feat(armhole_q)[0] - edge_feat(sleeve_R)[0])
print(f"  对称歧义: 身体袖窿边到左/右袖边的距离 = {dL:.6f} vs {dR:.6f} (|Δd| = {abs(dL-dR):.2e})")
print(f"  → 纯几何特征完全无法区分该缝哪只袖子，需要语义级缝合信息（stitch tags / DSL Interface）")

# ============ 实验 3：坐标量化误差（FSQ 预览） ============
print("\n=== 实验 3: 版型坐标直接量化 vs 潜空间量化 ===")
# 版型坐标归一化到 [-1, 1]（1.6m 版型范围）后 L 级均匀量化
# 理论: 均匀量化 RMS 误差 = step/sqrt(12)
scale = 0.8  # 1.6m 范围 → [-1,1]
for L in [5, 8, 16, 32]:
    step = 2.0 / (L - 1)
    xn = neckline_front / scale
    qn = np.clip(np.round((xn + 1) / step) * step - 1, -1, 1)
    rec = qn * scale
    err = np.linalg.norm(rec - neckline_front, axis=1)
    rms_theory = step * scale / np.sqrt(12)
    print(f"  L={L:3d} 级 (step {step*scale*1000:5.1f} mm): 实测 mean {err.mean()*1000:6.2f} mm / "
          f"RMS {np.sqrt((err**2).mean())*1000:6.2f} mm (理论 RMS {rms_theory*1000:6.2f} mm)")
print("  → 5 级直接量化坐标: RMS ~11.5cm, 32 级也还有 ~1.5cm —— 版型坐标的信息密度太高,")
print("    直接量化不可行。SewingLDM 的 5 级 FSQ 之所以可行: 量化的是训练学到的平滑潜变量,")
print("    且自编码器与量化器联合训练（量化在训练回路里, 编码器学会把信息放进量化网格能保住的方向）。")
