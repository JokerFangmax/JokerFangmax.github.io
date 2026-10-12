"""GG 05 数值实验：made-to-measure 公式依赖传播（Jacobian 结构）/ Interface 弧长展平与 ruffle / 采样规则的数据覆盖度"""
import numpy as np

rng = np.random.default_rng(11)

# ============ 实验 1：made-to-measure 的公式依赖——Jacobian 结构与约束重映射 ============
print("=== 实验 1: made-to-measure 公式传播（∂版型/∂身体测量）===")
# 玩具 bodice 前片: 8 项身体测量 → 派生量（真实裁剪公式风格, Armstrong 制版法量级）
#   前片宽 = 胸围/4 + 1cm;  袖窿深 = 胸围/8 + 10.5cm（经典公式）
#   腰线高 = 身高 − 头长(身高/8) − 20cm;  领口宽/深 = 颈宽/5 ∓ ...
#   collar 宽度用 GarmentCode 的真实机制: 归一化 style 参数 t∈[-0.5,1] 经线性插值
#   重映射到 [颈宽, 肩宽−4]（t≥0）/ [0, 颈宽)（t<0）——不是 clamp（对照 bodice.py 234–240 行）
# 注: arm_len/back_len 影响袖片/后片, 本玩具只画前片——它们对前片的 Jacobian 列应为 0

M_NAMES = ['height', 'chest', 'waist', 'hip', 'shoulder_w', 'neck_w', 'arm_len', 'back_len']
M_BASE = np.array([170.0, 92.0, 76.0, 98.0, 40.0, 15.0, 58.0, 41.0])  # cm

STYLE = dict(collar_t=0.6, ease=4.0)   # collar_t: 归一化领宽参数

def lin_interp(lo, hi, t):
    return lo + t * (hi - lo)

def draft_front(m, style):
    """8 测量 → 前片 12 顶点 (x,y)。公式模仿真实制版法的依赖结构。"""
    height, chest, waist, hip, sh_w, neck_w, arm, back = m
    front_w   = chest / 4 + 1 + style['ease'] / 4      # 前片宽
    armhole_d = chest / 8 + 10.5                        # 袖窿深
    waist_lvl = height - height / 8 - 20                # 腰线高
    neck_wd   = neck_w / 5 - 0.5                        # 领口宽
    neck_dp   = neck_w / 5 + 1                          # 领口深
    sh_pos    = sh_w / 2                                # 肩点 x
    # GarmentCode 式约束重映射: t∈[-0.5,1] → collar_w ∈ [0, 颈宽) ∪ [颈宽, 肩宽−4]
    t = style['collar_t']
    min_w, max_w = neck_w, sh_w - 4     # bodice.py: max_w = _base_sleeve_balance − 2 = shoulder_w − 4
    collar_w = lin_interp(min_w, max_w, t) if t >= 0 else lin_interp(0, min_w, 1 + t)
    hem_w     = hip / 4 + 0.5                           # 下摆宽
    waist_x   = waist / 4 + 2.5                         # 腰部收省后的侧缝 x
    shoulder_y = armhole_d + 12
    pts = np.array([
        [0, 0],                     # 0  左下摆
        [hem_w, 0],                 # 1  右下摆
        [front_w, 0],               # 2  右侧缝底
        [waist_x, waist_lvl],       # 3  右腰（收省）
        [front_w, armhole_d + (waist_lvl - armhole_d) / 2],  # 4  侧缝中点
        [front_w, armhole_d],       # 5  袖窿底
        [sh_pos, armhole_d + 4],    # 6  肩点
        [sh_pos, shoulder_y],       # 7  肩上
        [collar_w, shoulder_y],     # 8  领肩点 ← collar 重映射在此进入
        [neck_wd, shoulder_y - neck_dp * 0.3],  # 9 领口曲线点
        [neck_wd, shoulder_y],      # 10 领口中
        [0, shoulder_y],            # 11 左肩上
    ])
    return pts

V_BASE = draft_front(M_BASE, STYLE)
print(f"  基准身体: " + ", ".join(f"{n}={v:.0f}" for n, v in zip(M_NAMES, M_BASE)))
print(f"  前片 12 顶点, 宽 {V_BASE[:,0].max():.1f}cm × 高 {V_BASE[:,1].max():.1f}cm")

# 数值 Jacobian ∂(24 顶点坐标)/∂(8 测量)
def pattern_vec(m):
    return draft_front(m, STYLE).ravel()

J = np.zeros((24, 8))
h = 0.01
for k in range(8):
    mp, mm_ = M_BASE.copy(), M_BASE.copy()
    mp[k] += h; mm_[k] -= h
    J[:, k] = (pattern_vec(mp) - pattern_vec(mm_)) / (2 * h)

print(f"\n  Jacobian (24×8) 每项测量影响的顶点坐标数:")
for k, name in enumerate(M_NAMES):
    touched = np.where(np.abs(J[:, k]) > 1e-10)[0]
    print(f"    {name:11s}: {len(touched):2d}/24 个坐标")

# 解析对照（挑 4 个可手算的系数）
checks = [
    ("∂(右侧缝底 x = chest/4+1+ease/4)/∂chest", 2 * 2 + 0, 1, 0.25),
    ("∂(袖窿底 y = chest/8+10.5)/∂chest",        5 * 2 + 1, 1, 0.125),
    ("∂(腰线 y = h−h/8−20)/∂height",             3 * 2 + 1, 0, 0.875),
    ("∂(右腰 x = waist/4+2.5)/∂waist",           3 * 2 + 0, 2, 0.25),
]
print(f"\n  解析检验（数值 vs 理论）:")
ok = True
for label, row, col, theo in checks:
    num = J[row, col]
    good = abs(num - theo) < 1e-6
    ok &= good
    print(f"    {label:45s} = {num:.4f}  （理论 {theo:.4f} {'✓' if good else '✗'}）")
assert ok

# 约束重映射的分段行为: 归一化参数 t 从 -0.5 扫到 1, 看领肩点 x
t_grid = np.linspace(-0.5, 1.0, 31)
xs = np.array([draft_front(M_BASE, dict(STYLE, collar_t=t))[8, 0] for t in t_grid])
pos = t_grid >= 0
slope_pos = np.polyfit(t_grid[pos], xs[pos], 1)[0]
slope_neg = np.polyfit(t_grid[~pos], xs[~pos], 1)[0]
print(f"\n  约束重映射（GarmentCode 式, collar_t∈[-0.5,1] → [0, 颈宽) ∪ [颈宽, 肩宽−4]）:")
print(f"    领肩点 x: t<0 段斜率 {slope_neg:.1f}（收窄支, 映到 [0, {M_BASE[5]:.0f})）,")
print(f"    t≥0 段斜率 {slope_pos:.1f}（常规支, 映到 [{M_BASE[5]:.0f}, {M_BASE[4]-4:.0f}]）——处处连续无跳变")
# 换身体: 颈宽/肩宽变化时, 同一 t 自动重映射到新区间——版型永不失效
for nw, sw in [(13.0, 36.0), (18.0, 46.0)]:
    m2 = M_BASE.copy(); m2[5] = nw; m2[4] = sw
    x2 = draft_front(m2, STYLE)[8, 0]
    valid = 0 <= x2 <= sw - 4
    print(f"    身体 neck_w={nw:.0f}, shoulder_w={sw:.0f}: 同一 collar_t={STYLE['collar_t']} → 领宽 {x2:.1f}cm "
          f"（新区间 [{nw:.0f},{sw-4:.0f}] 内 {'✓' if valid else '✗'}）")
print(f"    → 约束不靠 clamp 不靠异常: 归一化参数本身定义在合法区间上——")
print(f"      '换身体仍有效' = 公式（连续传播）+ 重映射（参数空间即合法空间）")

# ============ 实验 2：Interface 展平——m≠n 边缝合与均匀褶量分布 ============
print("\n=== 实验 2: Interface 弧长展平（边数不等 + ruffle 褶皱系数）===")

def flatten_interface(edges_A, edges_B):
    """两边按累计弧长比例互相投影, 在并集分割点插入顶点, 返回配对段列表。
    ruffle 语义 = B/A 总长比（B 边缝合时打折裥/拉伸的量, 由两边自然长度决定）。"""
    LA = np.sum(edges_A); LB = np.sum(edges_B)
    brk_A = np.concatenate([[0], np.cumsum(edges_A)]) / LA   # A 的分割点(弧长比例)
    brk_B = np.concatenate([[0], np.cumsum(edges_B)]) / LB   # B 的分割点(弧长比例)
    union = np.unique(np.concatenate([brk_A, brk_B]))        # 映射到公共参数域取并集
    pairs = [(union[i+1]*LA - union[i]*LA, union[i+1]*LB - union[i]*LB)
             for i in range(len(union) - 1)]
    return np.array(pairs), LA, LB

def show(name, eA, eB):
    pairs, LA, LB = flatten_interface(eA, eB)
    ratios = pairs[:, 1] / pairs[:, 0]
    print(f"  [{name}] A(总长 {LA:.1f}cm, {len(eA)} 边) ↔ B(总长 {LB:.1f}cm, {len(eB)} 边)")
    print(f"    展平成 {len(pairs)} 个配对段; B/A 比值 mean {ratios.mean():.4f}, "
          f"std {ratios.std():.2e}（均匀）")
    # 朴素对照: 逐边顺序配对（不展平）, 前 min(m,n) 段硬配
    n = min(len(eA), len(eB))
    naive = eB[:n] / eA[:n]
    err = np.abs(np.concatenate([eB[:n], [np.sum(eB[n:])]]) -
                 np.concatenate([eA[:n], [np.sum(eA[n:])]]))
    print(f"    对照·朴素逐边配对: 比值 {naive.round(2)}（参差）, 段长误差和 {err.sum():.2f}cm")
    # 守恒
    assert abs(pairs[:, 0].sum() - LA) < 1e-9 and abs(pairs[:, 1].sum() - LB) < 1e-9
    print(f"    守恒: A 侧 {pairs[:,0].sum():.4f} = {LA:.4f} ✓, B 侧 {pairs[:,1].sum():.4f} = {LB:.4f} ✓")
    return ratios

# (a) 袖窿(3 边) ↔ 袖山(5 边): 袖山比袖窿长 30% 是真实缝纫的"袖山吃势"(cap ease)
show("袖窿↔袖山", np.array([10.0, 5.0, 8.0]), np.array([6.0, 7.0, 6.0, 5.0, 6.0]))

# (b) 腰(2 边) ↔ 裙上缘(6 边): 3.5 倍褶量
r = show("腰↔裙上缘(ruffle)", np.array([12.0, 8.0]), np.array([10.0, 12.5, 10.0, 15.0, 12.5, 10.0]))
print(f"\n  → Interface 的语义: ruffle = B/A 总长比({r.mean():.2f}), 弧长展平把它均摊到每一段——")
print(f"    褶量按弧长均匀分布, 不在接缝处堆积。这正是 GG 02 习题 3 的算法实现。")

# ============ 实验 3：采样规则 → 数据覆盖度（组合空间的占据问题） ============
print("\n=== 实验 3: 程序空间的采样覆盖度（占据问题）===")
# 玩具组件库（模仿 GarmentCode 的离散选择结构）:
#   bodice: 领口(3) × 门襟(2) × 袖型(3) × 衣长(2) = 36
#   skirt:  型(3) × 腰带(2) = 6;   pants: 型(2) = 2
#   组合语法: 单件 bodice / 单件 skirt / 单件 pants / dress = bodice×skirt
C_bodice, C_skirt, C_pants = 36, 6, 2
C = C_bodice + C_skirt + C_pants + C_bodice * C_skirt
print(f"  离散设计空间 |D| = {C_bodice}+{C_skirt}+{C_pants}+{C_bodice}×{C_skirt} = {C} 个拓扑")

# 模拟 GarmentCodeData 式采样: N 件独立均匀采样
for N in [100, 1000, 10000, 115000]:
    sample = rng.integers(0, C, N)
    seen = len(np.unique(sample))
    expect = C * (1 - (1 - 1 / C) ** N)   # 占据问题解析解
    print(f"  N = {N:6d}: 采到 {seen:3d}/{C} 个设计 (解析期望 {expect:6.1f}, "
          f"偏差 {abs(seen-expect)/expect*100:.1f}%), 平均每设计 {N/C:.1f} 件")

print(f"\n  → 玩具空间 115k 采样把每个拓扑覆盖 ~{115000//C} 次——多样性瓶颈不是离散组合数,")
print(f"    而是连续 style 参数 + made-to-measure 身体变化（每个'设计'是一个参数族, 不是单点）。")
print(f"    换算: 组件库再大 50 倍(|D|={C*50}), 115k 采样每设计仅 ~{115000/(C*50):.1f} 件——")
print(f"    'seen/unseen 划分'必须在连续参数或组合层级上才有意义（GG 07–08 的 SewingLDM 实测:")
print(f"    同 prompt 结构多样性低, 正是组合空间覆盖不均的症状）")
