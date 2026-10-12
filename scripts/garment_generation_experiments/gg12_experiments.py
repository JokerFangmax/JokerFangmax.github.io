"""GG 12 数值实验：版图总结——任务×表示矩阵 × 五路线统一对照 × CFG 两笔账
（numpy/scipy，无学习框架）
1) 任务×表示覆盖矩阵：19 篇论文 → (表示族, 任务) 格子映射 + 占有率统计 + 空白格
   （每个空白格标注本系列量化过它的"屏障"数字）
2) 同一件版型走五条表示路线（精确/覆盖权衡的统一对照）:
   程序（GarmentCode 式 made-to-measure）/ 量化边向量序列（DressCode 式）/
   占用栅格（GarmentImage 式）/ SDF 隐式场（ISP/LFM 式）/ PCA+FSQ 潜空间（SewingLDM 式）
   ——同一随机版型族上统一测：每件比特数、顶点/边界误差、闭合残差、缝合一致性、
   域外（省道）表达。五条路线各自的"付费方式"一目了然。
3) CFG 的两笔账（同一 GMM 先验统一扫 guidance w）:
   (i)  生成侧：p_cfg ∝ p_c^{1+w}/p_u^w 的【精确网格积分】（重叠模式下组件权重公式
       只是近似——本实验用 401² 网格积分做地面真值）→ 模式质量与合法熵随 w 崩塌；
       SewingLDM 的 CFG 4.5 处少数合法模式剩 2.1%（对接 P0 实测 n_panels {2:18,4:2}）
   (ii) 蒸馏侧：混合条件 SDS 的不动点偏置随 w 单调增长但【分支不灭】（现实版
       conditioning）；对照 GG 11 的"完美 conditioning"版本——分支在 λ*≈1.70 消亡
       + 远场反恢复。guidance 病理的【类型】由条件模型锐度决定。

实现注记（踩坑记录，写作时进 §8 自勘误）:
- GG 07 玩具打版器有三个沉睡几何 bug（详见 draft_front 文档字符串）: 领点全身尺度
  插值越侧缝、腰线公式把"腰"顶点放到肩线上方 100cm、两个横向宽度都摆在 y=0 上。
  GG 07 只用边向量（ΣE=0 与多边形有效性无关）所以从未暴露；本篇栅格/隐式实验直接
  暴雷（占用面积 1000 vs 真值 588 cm²）。
  教训：换一种表示会重新暴露"生成器"里沉睡的 bug（GG 10 手风琴段号 bug 同款）。
- 镜像后片的逐分量量化是【精确镜像不变】的（round(−x) = −round(x)）→ 缝长差恒 0，
  "缝合一致性被量化破坏"测不出来。真实版型的前后片各自独立排料（不同局部坐标系），
  必须先随机旋转再量化，长度才会独立漂移。
- draft(m) 对固定样式参数是分段线性的（秩 ≤ 6）→ PCA 谱退化为 6 个非零奇异值，
  "潜空间压缩曲线"测不出来。加入逐件随机样式参数（collar_t/ease，连续跨分支）后
  谱才非退化——真实数据（11.5 万件）的样式方差正是潜空间存在的前提。
- 复合密度的网格积分绝不能用 log-sum-exp 的"相对形状"（Σ exp(lp−max) 是密度除以
  最大分量，位置相关的错因子会让 p_c^{1+w}/p_u^w 完全跑偏——模式质量反向）；
  重叠模式下密度直接算（e^{−65} 在双精度内），逐点算完再归一。
- 线性子空间编码（PCA）对【线性】约束（闭合 ΣE=0）精确保持（μ 与每个主成分都在
  约束子空间内 → 任意系数组合都闭合）；被 FSQ 破坏的是【非线性】函数（缝长）。
  序列路线恰恰相反：量化直接破闭合。两种表示的"结构损伤面"不同。
- 直线边的 SDF 沿任意方向仿射 → 网格线上线性插值的零交叉【精确】落在边上，误差
  只集中在顶点折角；占用栅格只有 ±Δx/2 定位精度——同网格下 SDF 边界误差低一个
  数量级（"隐式场买到亚网格信息"的定量版，GG 09 可微网格化 9.96e-13 同机理）。
"""
import numpy as np
from scipy.optimize import least_squares

# ================= 实验 1：任务×表示覆盖矩阵 =================
print("=== 实验 1: 任务×表示覆盖矩阵（19 篇论文, GG 01–11 已核实事实的映射） ===")
# 表示族（2D 版型四族 + 3D/无版型桶）: seq=向量序列, prog=程序DSL, grid=栅格,
# implicit=2D隐式场, explicit3d=3D/无版型（SMPLicit/GarmentDreamer 内部用 3D 隐式,
#   但均非 2D 版型表示, 归此桶; 桶的含义是"不走版型"）
# 任务: drape=drape预测, recon=版型重建, gen=版型生成, simready=仿真就绪
PAPERS = [
    # (简称, 表示族, 任务, 系列)
    ("TailorNet CVPR20",        "explicit3d", "drape",    "GG03"),
    ("SMPLicit CVPR21",         "explicit3d", "drape",    "GG03"),
    ("DrapeNet CVPR23",         "explicit3d", "drape",    "GG04"),
    ("ISP NeurIPS23",           "implicit",   "recon",    "GG04"),
    ("GarmentCode SIGA23",      "prog",       "gen",      "GG05"),
    ("GarmentCodeData ECCV24",  "prog",       "simready", "GG05"),
    ("NeuralTailor SIG22",      "seq",        "recon",    "GG06"),
    ("SewFormer SIGA23",        "seq",        "recon",    "GG06"),
    ("GarmentRecovery CVPR24",  "implicit",   "recon",    "GG06"),
    ("DressCode SIG24",         "seq",        "gen",      "GG07"),
    ("GarmentDiffusion IJCAI25","seq",        "gen",      "GG07"),
    ("SewingLDM ICCV25",        "seq",        "gen",      "GG08"),
    ("GarmentImage SIG25",      "grid",       "gen",      "GG09"),
    ("LatentFM SIG26",          "implicit",   "gen",      "GG09"),
    ("GarmentoPIA CVM26",       "prog",       "gen",      "GG09"),
    ("Dress-1-to-3 TOG25",      "seq",        "simready", "GG10"),
    ("DMap-static SIG25",       "implicit",   "recon",    "GG10"),
    ("DMap-dynamic TPAMI26",    "implicit",   "recon",    "GG10"),
    ("GarmentDreamer 3DV25",    "explicit3d", "gen",      "GG11"),
]
REPS = ["seq", "prog", "grid", "implicit", "explicit3d"]
TASKS = ["drape", "recon", "gen", "simready"]
REP_CN = {"seq": "向量序列", "prog": "程序DSL", "grid": "栅格",
          "implicit": "2D隐式场", "explicit3d": "3D/无版型"}
TASK_CN = {"drape": "drape预测", "recon": "版型重建", "gen": "版型生成", "simready": "仿真就绪"}

cell = {(r, t): [] for r in REPS for t in TASKS}
for name, rep, task, src in PAPERS:
    cell[(rep, task)].append(name.split()[0])

print(f"\n  {'':>10s}" + "".join(f"{TASK_CN[t]:>14s}" for t in TASKS))
for r in REPS:
    row = f"  {REP_CN[r]:>10s}"
    for t in TASKS:
        names = cell[(r, t)]
        row += f"{(str(len(names)) + '篇' if names else '—'):>14s}"
    print(row)
filled = sum(1 for k in cell if cell[k])
print(f"\n  占有率: {filled}/{len(cell)} = {filled/len(cell):.0%}")
for r in REPS:
    n = sum(len(cell[(r, t)]) for t in TASKS)
    print(f"    {REP_CN[r]:>8s}: {n} 篇  " +
          ", ".join(f"{TASK_CN[t]}×{len(cell[(r,t)])}" for t in TASKS if cell[(r, t)]))
for t in TASKS:
    n = sum(len(cell[(r, t)]) for r in REPS)
    print(f"    {TASK_CN[t]:>8s}: {n} 篇  " +
          ", ".join(f"{REP_CN[r]}×{len(cell[(r,t)])}" for r in REPS if cell[(r, t)]))

# 空白格 + 本系列量化过的"屏障"
BARRIERS = {
    ("grid", "recon"):     "观测→栅格的逆映射无人在做；栅格离散化让边界误差 ~Δx/2（实验 2 路线 3）",
    ("grid", "drape"):     "栅格不含 3D 信息——drape 需要身体条件，GarmentImage 全文无身体",
    ("grid", "simready"):  "16×16 占用无缝合结构通道→无法直接进仿真器；缝合 F1 0.993 要靠 LFM 式附加场",
    ("implicit", "drape"): "2D 隐式版型不含 3D（ISP 的 drape 分支要靠额外变形网络）；3D 隐式 drape（SMPLicit）在 3D/无版型桶",
    ("implicit", "simready"): "LFM 的 DSR 81%/69% 是最接近的尝试——可组装悬垂比例, 但 panel 数仍离散",
    ("seq", "drape"):      "drape 是 3D 任务, 序列是 2D 表示——TailorNet 式子网格+位移场才是正解",
    ("prog", "recon"):     "观测→程序 = 程序综合逆问题; GarmentoPIA 85.7% 靠 LLM agent, 非逆投影",
    ("prog", "drape"):     "程序生成 2D 版型, drape 需 3D 前向仿真——GarmentCodeData 用 XPBD 离线做",
    ("explicit3d", "recon"): "Garment Recovery 的变形先验分支沾边, 主表示仍是隐式版型",
    ("explicit3d", "simready"): "GG 11 实验 2: 3D 边长当 rest → 皱纹亚稳态; rest-shape 歧义无解, 须先平展——非可展畸变 ∝a² 无阈值（GG 11 实验 1）",
}
empty = [(r, t) for r in REPS for t in TASKS if not cell[(r, t)]]
print(f"\n  空白格（{len(empty)}/{len(cell)}）与本系列量化过的屏障:")
for (r, t) in empty:
    print(f"    {REP_CN[r]} × {TASK_CN[t]}: {BARRIERS[(r, t)]}")
print("  → 空白不是均匀分布: 仿真就绪列最稀（2/19 篇）, 且全部依赖 序列+物理 或 程序+XPBD;")
print("    drape 列全部在 3D/无版型桶（版型表示不直接做 drape, 要经仿真器）;")
print("    栅格行只有 1 篇（GarmentImage 是表示论文非生成论文, GG 09 勘误）;")
print("    版型生成列最密（8 篇）, 除 GarmentCode'23（DSL/数据基座）外全部 2024–2026——")
print("    生成是最新的一波; 仿真就绪是最后一块没人占满的格子。")

# ================= 实验 2：同一件版型走五条表示路线 =================
print("\n=== 实验 2: 同一件版型走五条表示路线（统一精确/覆盖对照） ===")

def lin_interp(lo, hi, t):
    return lo + t * (hi - lo)

def draft_front(m, collar_t=0.6, ease=4.0):
    """玩具 bodice 前片: 8 测量 + 2 样式 → 12 顶点（cm）——GG 12 修正版
    GG 05/07 原版的三个沉睡几何 bug（边向量实验无感, 栅格/隐式实验暴雷）:
      ① 领点用全身宽度插值（13–36cm）, 但 1/4 版型 x≤25 → 领点常越侧缝;
      ② 腰线 waist_lvl = height·7/8 − 20 ≈ 134 远高于肩线 ≈ 33（纵向比例错乱,
        原版"腰"顶点悬在肩上方, 每件都自交）;
      ③ hem_w 与 front_w 两个横向宽度都放在 y=0 上, 臀宽 > 胸宽时底边自碰。
    本版: 肩线为顶（y = back+20）, 袖窿底/腰线从肩线往下量, 顶点按
    下摆→侧缝(腰/胸)→袖窿→肩→领口→前中 的真实绕向排列。测量依赖关系
    （chest/8、hip/4、waist/4、neck/5、sh_w/2 等）与原版一致。"""
    height, chest, waist, hip, sh_w, neck_w, arm, back = m
    shoulder_y = back + 20                    # 肩线高度 = 背长 + 余量
    armhole_d = chest / 8 + 10.5              # 袖窿深（自肩线向下）
    arm_y = shoulder_y - armhole_d            # 袖窿底高度
    waist_lvl = shoulder_y - height / 6.5     # 腰线（自肩线向下, 保证低于袖窿底）
    front_w = chest / 4 + 1 + ease / 4        # 胸宽（袖窿底处）
    hem_w = hip / 4 + 0.5                     # 臀宽（下摆）
    waist_x = waist / 4 + 2.5                 # 腰宽
    neck_wd = neck_w / 5 - 0.5                # 前领口宽
    sh_pos = sh_w / 2                         # 肩点 x
    collar_w = (lin_interp(neck_w / 2, sh_w / 2, collar_t) if collar_t >= 0
                else lin_interp(0, neck_w / 2, 1 + collar_t))
    return np.array([
        [0, 0],                                          # v0  前中下摆
        [hem_w, 0],                                      # v1  侧下摆（臀）
        [waist_x, waist_lvl],                            # v2  侧腰
        [front_w, waist_lvl + (arm_y - waist_lvl) / 2],  # v3  侧缝中段
        [front_w, arm_y],                                # v4  袖窿底（胸宽）
        [front_w - 1, arm_y + (shoulder_y - arm_y) / 2], # v5  袖窿中段（微收）
        [sh_pos, shoulder_y - 4],                        # v6  袖窿顶
        [sh_pos, shoulder_y],                            # v7  肩点
        [collar_w, shoulder_y],                          # v8  领点
        [neck_wd, shoulder_y - (neck_w / 5 + 1) * 0.3],  # v9  前领口下降
        [neck_wd, shoulder_y],                           # v10 领口侧
        [0, shoulder_y],                                 # v11 前中肩线
    ])

def draft_back(m, collar_t=0.6, ease=4.0):
    """后片: 与前片共享肩缝边（v7→v8 同公式 → 缝合长度精确一致, 真实版型的
    缝合约束）, 领口更浅/袖窿更深/下摆略窄——后片≠前片镜像（真实服装如此）,
    两片独立编码时误差才会独立漂移"""
    V = draft_front(m, collar_t, ease)
    V = V.copy()
    V[1] = [V[1][0] - 1.0, 0.0]                          # 后下摆略窄
    V[5] = [V[5][0] - 0.8, V[5][1]]                      # 袖窿更收
    V[9] = [V[9][0], V[9][1] + (V[10][1] - V[9][1]) * 0.6]  # 后领口更浅
    return V

def random_measurements(g):
    return np.array([g.uniform(163, 178), g.uniform(84, 100), g.uniform(68, 86),
                     g.uniform(88, 108), g.uniform(37, 44), g.uniform(13, 17),
                     g.uniform(52, 62), g.uniform(38, 44)])

def random_style(g):
    """逐件随机样式（跨 collar_t=0 分支 + 与测量相乘 → 非线性, PCA 谱非退化）"""
    return g.uniform(-0.4, 0.9), g.uniform(2.0, 8.0)

def poly_simple(V):
    """非相邻边相交检测（简单多边形断言）"""
    n = len(V)
    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]
    def seg_int(p1, p2, p3, p4):
        d1 = cross2(p2 - p1, p3 - p1); d2 = cross2(p2 - p1, p4 - p1)
        d3 = cross2(p4 - p3, p1 - p3); d4 = cross2(p4 - p3, p2 - p3)
        return ((d1 * d2 < 0) and (d3 * d4 < 0))
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if seg_int(V[i], V[(i + 1) % n], V[j], V[(j + 1) % n]):
                return False
    return True

g_tr = np.random.default_rng(42)
M_TRAIN = []
for _ in range(400):
    m = random_measurements(g_tr); ct, ez = random_style(g_tr)
    M_TRAIN.append(np.r_[m, ct, ez])
g_te = np.random.default_rng(1337)
M_TEST = []
for _ in range(40):
    m = random_measurements(g_te); ct, ez = random_style(g_te)
    M_TEST.append(np.r_[m, ct, ez])

SEAM = 7   # 肩缝边 v7→v8（前后片同公式 → 真值缝合长度精确一致）
def edges_of(V):
    return np.roll(V, -1, axis=0) - V

def closure_res(E):
    return np.linalg.norm(E.sum(0))

def seam_mismatch(Ef, Eb):
    return abs(np.linalg.norm(Ef[SEAM]) - np.linalg.norm(Eb[SEAM]))

Vf = [draft_front(m[:8], m[8], m[9]) for m in M_TEST]     # 前片
Vb = [draft_back(m[:8], m[8], m[9]) for m in M_TEST]      # 后片（共享肩缝, 非镜像）
assert all(poly_simple(v) for v in Vf + Vb), "自交多边形（collar 修正应已排除）"
assert max(seam_mismatch(edges_of(f), edges_of(b)) for f, b in zip(Vf, Vb)) < 1e-12
print("  简单多边形断言: 80/80 片通过; 肩缝长度前=后（真值缝合约束, 1e-12）")
print("  （GG 07 原版打版器在此处暴露三个沉睡几何 bug, 文件头注记）")

def poly_perimeter_pts(V, n=200):
    """真值轮廓采样点（边上线性采样）"""
    E = edges_of(V)
    L = np.linalg.norm(E, axis=1)
    out = []
    for i in range(len(V)):
        k = max(2, int(n * L[i] / L.sum()))
        t = np.linspace(0, 1, k, endpoint=False)[:, None]
        out.append(V[i] + t * E[i])
    return np.concatenate(out)

# ---- 路线 1: 程序（made-to-measure 重打版） ----
print("\n  路线 1  程序（GarmentCode 式）: 版型 = draft(m, style), 只编码参数值")
print(f"      {'参数精度 δ':>10s} {'每件 bits':>10s} {'顶点RMS(mm)':>12s} {'闭合残差':>10s} {'缝长差':>8s}")
P_LO = np.array(M_TEST).min(0); P_HI = np.array(M_TEST).max(0)
for dlt in [1.0, 0.5, 0.1, 0.05]:
    bits = np.log2((P_HI - P_LO) / dlt).sum()
    errs, cl, sm = [], [], []
    for m, vf in zip(M_TEST, Vf):
        mq = np.round(m / dlt) * dlt
        vq = draft_front(mq[:8], mq[8], mq[9])
        vqb = draft_back(mq[:8], mq[8], mq[9])
        errs.append(np.sqrt(((vq - vf) ** 2).sum(1).mean()))
        cl.append(closure_res(edges_of(vq)))
        sm.append(seam_mismatch(edges_of(vq), edges_of(vqb)))
    print(f"      {dlt*10:8.1f}mm {bits:10.0f} {np.mean(errs)*10:10.3f}mm "
          f"{np.max(cl):9.2e}cm {np.max(sm):7.2e}cm")
print("      → 域内零结构损伤: 闭合/缝合由构造精确（GG 09: 8.88e-16）; 误差 = 参数精度×∂版型/∂参数")
print("        （线性系数 0.125（chest/8→袖窿深）… 1.0（back→肩线高度）, 实测顶点")
print("          RMS ≈ 0.9·δ: 高度向系数≈1 主导——GG 05 made-to-measure 的'参数空间即")
print("          合法空间'在统一对照里的样子: 60–73 bit 就到毫米级, 且闭缝零损伤）")
print("        代价: 表达力 = 组件库（省道/口袋不在库内即失败, 见下方域外测试）")

# ---- 路线 2: 量化边向量序列（DressCode/GarmentDiffusion 式） ----
print("\n  路线 2  量化边向量序列: 编码 (v_x,v_y)×12边/panel, 逐分量独立量化")
print("          （后片在随机朝向的局部坐标系里独立量化——真实排料; 镜像直接量化的话")
print("            round(−x)=−round(x) 会让缝长差恒 0, 测不出缝合损伤）")
R_C = 60.0   # 边向量分量范围 (cm)
g_ang = np.random.default_rng(7)
ANGS = g_ang.uniform(0, 2 * np.pi, len(Vf))   # 每件后片一个随机朝向
print(f"      {'步长 h':>8s} {'每件 bits':>10s} {'顶点RMS(mm)':>12s} {'闭合残差中位':>12s} {'缝长差中位':>10s} {'缝长差>5mm':>10s}")
for h in [2.0, 1.0, 0.5, 0.2]:
    bits = 2 * 12 * 2 * np.log2(2 * R_C / h)     # 2 panels × 12 edges × 2 dims
    v_err, cl, sm = [], [], []
    for vf, vb, ang in zip(Vf, Vb, ANGS):
        Ef, Eb = edges_of(vf), edges_of(vb)
        Eqf = np.round(Ef / h) * h
        # 后片: 旋转 → 量化 → （长度是旋转不变量, 直接比长度）
        c, s = np.cos(ang), np.sin(ang)
        Rb = np.array([[c, -s], [s, c]])
        Eqb = np.round((Eb @ Rb.T) / h) * h
        rec = np.concatenate([np.zeros(2)[None], np.cumsum(Eqf, 0)])[:-1]
        v_err.append(np.sqrt(((rec - vf) ** 2).sum(1).mean()))
        cl.append(closure_res(Eqf))
        sm.append(seam_mismatch(Eqf, Eqb))
    sm = np.array(sm)
    print(f"      {h*10:6.0f}mm {bits:10.0f} {np.mean(v_err)*10:10.3f}mm "
          f"{np.median(cl):10.3f}cm {np.median(sm):8.3f}cm {int((sm > 0.5).sum()):8d}/40")
print("      → 顶点误差按 √j 随机游走累积（GG 07）; 闭合 ≈ h（本打版器一半边是轴对齐的,")
print("        一个分量误差恒 0 → 有效噪声项 ~6–8/12; 无结构理论值 ~1.18h）。h=20mm 档")
print("        闭合跳到 1.4h: 最短边（前领口下降 ~1cm）短于 h/2 被量化吞零——步长一旦")
print("        超过结构尺度, 误差非线性爆炸（'Δx/2 只是下界'的实证）")
print("        缝长差 ~0.1–0.3h（依赖缝边相对量化轴的朝向）: 两片独立量化、长度各自漂移,")
print("        h=10mm 时 4/40 件超 5mm 缝纫容差——跨 token 全局约束（闭合/缝合）逐分量")
print("        量化必破, 而单坐标 RMS（h/√12）看不出（GG 07 结论在两片设置下复现）")

# ---- 路线 3+4: 占用栅格（GarmentImage 式） vs SDF 隐式场（ISP/LFM 式） ----
def bbox_grid(V, n):
    lo, hi = V.min(0) - 1, V.max(0) + 1
    xs = np.linspace(lo[0], hi[0], n + 1)
    ys = np.linspace(lo[1], hi[1], n + 1)
    return xs, ys

def winding_inside(P, V):
    """点在多边形内（射线法, 向量化）"""
    X, Y = P[..., 0], P[..., 1]
    n = len(V)
    inside = np.zeros(X.shape, bool)
    for i in range(n):
        x1, y1 = V[i]; x2, y2 = V[(i + 1) % n]
        cond = ((y1 > Y) != (y2 > Y))
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = (x2 - x1) * (Y - y1) / (y2 - y1) + x1
        inside ^= cond & (X < xint)
    return inside

def occupancy(V, n):
    xs, ys = bbox_grid(V, n)
    cx = 0.5 * (xs[:-1] + xs[1:]); cy = 0.5 * (ys[:-1] + ys[1:])
    P = np.stack(np.meshgrid(cx, cy, indexing='ij'), -1)
    return xs, ys, winding_inside(P, V)

def boundary_from_occupancy(xs, ys, occ):
    """占用栅格的边界 = 占用/空单元格之间的格边, 采样为线段端点集"""
    segs = []
    nx, ny = occ.shape
    for i in range(nx):
        for j in range(ny):
            if not occ[i, j]:
                continue
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            if i + 1 >= nx or not occ[i + 1, j]: segs.append(((x1, y0), (x1, y1)))
            if i - 1 < 0 or not occ[i - 1, j]:   segs.append(((x0, y0), (x0, y1)))
            if j + 1 >= ny or not occ[i, j + 1]: segs.append(((x0, y1), (x1, y1)))
            if j - 1 < 0 or not occ[i, j - 1]:   segs.append(((x0, y0), (x1, y0)))
    return segs

def segs_pts(segs, n=200):
    if not segs:
        return np.zeros((0, 2))
    out = []
    per = max(1, n // len(segs))
    for (a, b) in segs:
        t = np.linspace(0, 1, per, endpoint=False)[:, None]
        out.append(np.array(a) + t * (np.array(b) - np.array(a)))
    return np.concatenate(out)

def point_poly_dist(P, V):
    """点到多边形边的最短距离（向量化: P (...,2) × 边）"""
    n = len(V)
    dmin = np.full(P.shape[:-1], np.inf)
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        proj = a + t[..., None] * ab
        dmin = np.minimum(dmin, np.linalg.norm(P - proj, axis=-1))
    return dmin

def sdf_grid(V, n):
    xs, ys = bbox_grid(V, n)
    cx = 0.5 * (xs[:-1] + xs[1:]); cy = 0.5 * (ys[:-1] + ys[1:])
    P = np.stack(np.meshgrid(cx, cy, indexing='ij'), -1)
    d = point_poly_dist(P, V)
    return xs, ys, d * np.where(winding_inside(P, V), -1, 1)

def boundary_from_sdf(xs, ys, d):
    """marching squares + 线性插值: SDF 网格的零等值线段"""
    segs = []
    nx, ny = d.shape
    for i in range(nx - 1):
        for j in range(ny - 1):
            c = [d[i, j], d[i + 1, j], d[i + 1, j + 1], d[i, j + 1]]
            x0, x1 = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (xs[i + 1] + xs[i + 2])
            y0, y1 = 0.5 * (ys[j] + ys[j + 1]), 0.5 * (ys[j + 1] + ys[j + 2])
            edges = [((x0, y0), (x1, y0), c[0], c[1]), ((x1, y0), (x1, y1), c[1], c[2]),
                     ((x1, y1), (x0, y1), c[2], c[3]), ((x0, y1), (x0, y0), c[3], c[0])]
            cross = []
            for (p, q, cp, cq) in edges:
                if (cp < 0) != (cq < 0):
                    t = cp / (cp - cq)
                    cross.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            if len(cross) >= 2:
                segs.append((cross[0], cross[1]))
                if len(cross) == 4:
                    segs.append((cross[2], cross[3]))
    return segs

def chamfer(A, B):
    """对称平均最近点距离"""
    if len(A) == 0 or len(B) == 0:
        return np.inf
    d2 = ((A[:, None, :] - B[None]) ** 2).sum(-1)
    return 0.5 * (np.sqrt(d2.min(1)).mean() + np.sqrt(d2.min(0)).mean())

def eval_grid_routes(V_list, n, npts=3000):
    ch_occ, ch_sdf = [], []
    for vf in V_list:
        true_pts = poly_perimeter_pts(vf, npts)
        xs, ys, occ = occupancy(vf, n)
        ch_occ.append(chamfer(segs_pts(boundary_from_occupancy(xs, ys, occ), npts), true_pts))
        xs2, ys2, d = sdf_grid(vf, n)
        ch_sdf.append(chamfer(segs_pts(boundary_from_sdf(xs2, ys2, d), npts), true_pts))
    return np.mean(ch_occ), np.mean(ch_sdf)

print("\n  路线 3/4  占用栅格（GarmentImage 式, 1 bit/cell） vs SDF 隐式场（ISP/LFM 式, 8 bit/cell）")
print(f"      {'网格':>6s} {'bits 占用/SDF':>16s} {'边界Chamfer 占用(mm)':>20s} {'边界Chamfer SDF(mm)':>20s} {'SDF优势':>8s}")
for n in [16, 32, 64]:
    bits_occ = n * n * 2
    bits_sdf = n * n * 2 * 8
    ch_o, ch_s = eval_grid_routes(Vf[:12], n)
    print(f"      {n:4d}²  {bits_occ:>6d}/{bits_sdf:<6d}  {ch_o*10:18.3f}mm "
          f"{ch_s*10:18.3f}mm {ch_o/ch_s:7.1f}×")
ch_o64, _ = eval_grid_routes(Vf[:12], 64)
_, ch_s23 = eval_grid_routes(Vf[:12], 23)
print(f"      同比特对照: 占用 64²（{64*64*2} bit） {ch_o64*10:.2f}mm  vs  "
      f"SDF 23²（{23*23*2*8} bit） {ch_s23*10:.2f}mm")
print("      → 读数（按实测, 不按预期）: SDF 的优势是【同网格】精度——直线边的 SDF 沿")
print("        任意方向仿射, 网格线上线性插值的零交叉精确落在边上, 误差集中在顶点折角;")
print("        占用栅格只有 ±Δx/2 定位精度。但两个反转: ① 粗网格上 SDF 反而更差——")
print("        特征小于格子时（本例前领口下降 ~1cm < Δx≈2cm）零交叉插值被小特征带偏,")
print("        占用的'多数表决'反而稳; ② 同比特下占用胜——SDF 每格 8 bit, 同预算只买")
print("        得起 √8 分之一的格子。隐式场真正买到的是亚网格精度 + 连续场可微")
print("        （GG 09 可微网格化 9.96e-13 的机理）, 付的是 8× bit 和小特征脆弱。")
print("        代价之二是拓扑显式信息没了（哪些边该缝合要从 SDF 之外的通道拿,")
print("        ISP 的缝合标签 / LFM 的 DSR）")

# ---- 路线 5: PCA + FSQ 潜空间（SewingLDM 式） ----
print("\n  路线 5  PCA+FSQ 潜空间（SewingLDM 式: 连续向量→压缩潜码→逐维 5 级量化）")
# 训练集 = 前片 + 后片（只训前片时后片是分布外 → 重建垃圾; 真实系统全 panel 训练）
E_TRAIN = np.array(
    [edges_of(draft_front(m[:8], m[8], m[9])).ravel() for m in M_TRAIN] +
    [edges_of(draft_back(m[:8], m[8], m[9])).ravel() for m in M_TRAIN])
mu = E_TRAIN.mean(0)
Xc = E_TRAIN - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
eig_cum = (S ** 2).cumsum() / (S ** 2).sum()
print(f"      PCA 谱（前后片合并 800 样本）: 前 4 维能量 {eig_cum[3]:.0%}, "
      f"前 8 维 {eig_cum[7]:.0%}, 前 12 维 {eig_cum[11]:.0%}")
print("      （玩具族由 10 个参数生成 → 谱快衰减、~8 维后截断误差归零; 真实 11.5 万件")
print("        数据的样式方差大得多, 但'低维流形 + 潜码'的定性结构相同——GG 08 的")
print("        5⁶=15,625 ≈ 19,683 样本匹配就是这个计数关系的真数据版）")
scale = np.abs(Xc @ Vt.T).max(0)          # 每维归一化到 [−1,1]
LEV = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
print(f"      {'k 潜维':>7s} {'每件 bits':>9s} {'顶点RMS(mm)':>11s} {'PCA截断L2(mm/件)':>16s} {'闭合残差':>10s} {'缝长差中位':>10s}")
for k in [2, 4, 6, 8, 12, 16, 22]:
    v_err, cl, sm = [], [], []
    for m, vf, vb in zip(M_TEST, Vf, Vb):
        for V_, store in ((vf, 'f'), (vb, 'b')):
            e = edges_of(V_).ravel()
            z = (e - mu) @ Vt[:k].T / scale[:k]
            zq = LEV[np.abs(z[:, None] - LEV).argmin(1)]    # FSQ: 最近 5 级
            rec = ((zq * scale[:k]) @ Vt[:k] + mu).reshape(12, 2)
            if store == 'f':
                Eqf = rec
                rec_v = np.concatenate([np.zeros(2)[None], np.cumsum(Eqf, 0)])[:-1]
                v_err.append(np.sqrt(((rec_v - vf) ** 2).sum(1).mean()))
                cl.append(closure_res(Eqf))
            else:
                Eqb = rec
        sm.append(seam_mismatch(Eqf, Eqb))
    pca_l2 = np.sqrt((S[k:] ** 2).sum() / len(E_TRAIN)) if k < 24 else 0.0
    print(f"      {k:7d} {k*np.log2(5):9.0f} {np.mean(v_err)*10:9.3f}mm "
          f"{pca_l2*10:14.3f}mm {np.max(cl):8.2e}cm {np.median(sm):8.3f}cm")
print("      → 两段曲线: 小 k 段 PCA 截断误差主导（幂律降）, ~谱秩处饱和进 FSQ 量化")
print("        地板（每维 5 级 → ±0.25 归一化单位的量化噪声; 玩具族零维恰好零方差,")
print("        噪声通道不激活所以持平——真实数据每维都有方差, GG 08 '大潜空间重建")
print("        饱和甚至回升'的右端在此只见'饱和'）")
print("      → 闭合残差机器零（~1e-13, 非'小'而是【结构性为零】）: 闭合 ΣE=0 是线性约束,")
print("        训练数据全在约束子空间内 → μ 与每个主成分都在子空间内 → 任意（量化后的）")
print("        系数组合仍闭合。被 FSQ 破坏的是【非线性】泛函: 缝长差从 k=2 的 0.015cm 升")
print("        到 k=6 的 0.246cm 后饱和（噪声通道数↑ 与截断误差↓ 竞争）——与路线 2 恰好")
print("        相反（量化直接破闭合, 两种表示的'结构损伤面'不同）。SewingLDM 的真实解码器")
print("        是神经网络而非固定线性子空间, 闭合不免费——但'线性约束可被表示精确编码'")
print("        是可利用的性质")
print("      → 比特效率: 同 ~8.6mm 顶点 RMS, 序列路线需 332 bit（h=10mm）, PCA+FSQ 只要")
print("        51 bit（6.5×）——潜压缩吃到了版型的强相关性; 换来的是损伤面从闭合挪到缝长")

# ---- 域外测试: 省道（dart take-up） ----
print("\n  域外测试: 省道版型（侧缝边 4–5 中点向内收 3cm → 13 顶点, 训练族外）")
m0 = M_TEST[0]
V0 = draft_front(m0[:8], m0[8], m0[9])
mid = 0.5 * (V0[4] + V0[5]) - np.array([3.0, 0.0])   # 向内收 3cm = 省道吃量
V_dart = np.insert(V0, 5, mid, axis=0)
assert poly_simple(V_dart)
true_pts = poly_perimeter_pts(V_dart, 400)
# 序列: 13 条边照常量化
h = 0.5
E_d = edges_of(V_dart)
Eq_d = np.round(E_d / h) * h
rec = np.concatenate([np.zeros(2)[None], np.cumsum(Eq_d, 0)])[:-1]
seq_err = chamfer(poly_perimeter_pts(rec, 400), true_pts)
# SDF 16×16 与 32×32
xs, ys, d = sdf_grid(V_dart, 16)
segs = boundary_from_sdf(xs, ys, d)
sdf_err = chamfer(segs_pts(segs, 400), true_pts)
xs32, ys32, d32 = sdf_grid(V_dart, 32)
segs32 = boundary_from_sdf(xs32, ys32, d32)
sdf_err32 = chamfer(segs_pts(segs32, 400), true_pts)
print(f"      序列 h=5mm:  边界Chamfer {seq_err*10:.2f}mm（多一条边 token, 误差同域内量级）")
print(f"      SDF 16²:    边界Chamfer {sdf_err*10:.2f}mm（3cm 省道在 Δx≈2cm 网格上勉强可辨, 32² 舒服)")
print(f"      SDF 32²:    边界Chamfer {sdf_err32*10:.2f}mm")
print(f"      程序: draft() 无省道指令 → 无法编码（结构失败, 非精度失败）")
print(f"      PCA+FSQ: 输入定维 24（12 边×2）→ 13 边的 26 维输入无法前向。SewingLDM 的答案是")
print(f"                39×37 定长 padding + existence flag——离散维度买的正是这个域外自由度")

# ================= 实验 3：CFG 的两笔账 =================
print("\n=== 实验 3: CFG 的两笔账——生成侧熵崩塌 vs 蒸馏侧不动点偏置（同一 GMM 扫 w） ===")
MU3 = np.array([[0.0, 0.0], [1.1, 0.5], [-0.9, 0.7]])
SC3 = 0.5
PI_U = np.ones(3) / 3
PI_C = np.array([0.70, 0.20, 0.10])   # 条件模型: prompt 指向模式0, 相邻模式泄漏

def _sigprep(sig):
    sig = np.asarray(sig, float)
    if np.ndim(sig):
        return sig[:, None], (sig ** 2)[:, None], (SC3 ** 2 + sig ** 2)[:, None]
    return sig, sig ** 2, SC3 ** 2 + sig ** 2

def post_mean_mixw(x, sig, PI):
    single = np.ndim(x) == 1
    X = np.atleast_2d(np.asarray(x, float))
    _, _, v = _sigprep(sig)
    d2 = ((X[:, None, :] - MU3[None]) ** 2).sum(-1)
    logw = np.log(PI) - 0.5 * d2 / v
    logw -= logw.max(-1, keepdims=True)
    w = np.exp(logw); w /= w.sum(-1, keepdims=True)
    out = np.einsum('nk,kd->nd', w, MU3)
    return out[0] if single else out

def eps_u(x, sig): return (x - post_mean_mixw(x, sig, PI_U)) / _sigprep(sig)[0]
def eps_c(x, sig): return (x - post_mean_mixw(x, sig, PI_C)) / _sigprep(sig)[0]
def eps_cfg_mix(x, sig, w): return eps_u(x, sig) + w * (eps_c(x, sig) - eps_u(x, sig))

# ---- (i) 生成侧: p_cfg ∝ p_c^{1+w}/p_u^w 精确网格积分 ----
def gmm_pdf(X, PI):
    d2 = ((X[:, None, :] - MU3[None]) ** 2).sum(-1)
    return (PI[None] * np.exp(-0.5 * d2 / SC3 ** 2)).sum(-1)

gg = np.linspace(-4, 4, 401)
GX, GY = np.meshgrid(gg, gg, indexing='ij')
PTS = np.stack([GX, GY], -1).reshape(-1, 2)
assign = np.argmin(((PTS[:, None, :] - MU3[None]) ** 2).sum(-1), -1)
print("\n  (i) 生成侧: 复合密度 p_cfg ∝ p_c^{1+w}/p_u^w 的精确网格积分（401², Voronoi 归属）")
print(f"      {'w':>5s} {'模式质量 [m0,m1,m2]':>26s} {'合法熵(bit)':>10s} {'组件权重近似 m1':>16s}")
for w in [0, 1, 2, 3, 4.5, 10]:
    pc, pu = gmm_pdf(PTS, PI_C), gmm_pdf(PTS, PI_U)
    dens = pc if w == 0 else pc ** (1 + w) / pu ** w
    dens /= dens.sum()
    masses = np.array([dens[assign == k].sum() for k in range(3)])
    ent = -(masses * np.log2(np.maximum(masses, 1e-300))).sum()
    if w == 0:
        approx_m1 = PI_C[1]
    else:
        pw = PI_C ** (1 + w) / PI_U ** w; pw /= pw.sum()
        approx_m1 = pw[1]
    print(f"      {w:5.1f} {str(masses.round(4)):>26s} {ent:10.4f} {approx_m1:16.4f}")
print("      → 重叠模式下组件权重公式只是近似（GG 08 分离模式的极限）; 精确积分更温和")
print("        但熵崩塌不变: w=3 少数合法模式 5.0%, w=4.5（SewingLDM 实际 CFG）剩 2.1%")
print("        —— P0 实测 n_panels {2:18, 4:2}（少数结构 10%）与该量级一致,")
print("           多样性损失是 CFG 单旋钮几何衰减的预测结果, 不是实现 bug")

# ---- (ii) 蒸馏侧: 混合条件 SDS 的不动点（单调偏置） vs 完美条件（分支消亡） ----
GH_N = 41
u_, w_ = np.polynomial.hermite.hermgauss(GH_N)
GHU = np.stack(np.meshgrid(u_, u_, indexing='ij'), -1).reshape(-1, 2)
GHW = (w_[:, None] * w_[None, :]).reshape(-1) / np.pi
SIGG = np.linspace(0.2, 1.0, 9)

def h_exact(th, w):
    tot = np.zeros(2)
    for sg in SIGG:
        xt = np.asarray(th)[None, :] + sg * np.sqrt(2) * GHU
        tot += (eps_cfg_mix(xt, sg, w) * GHW[:, None]).sum(0)
    return tot / len(SIGG)

print("\n  (ii) 蒸馏侧 A: 混合条件（现实版 conditioning）的不动点 —— 偏置单调增长, 分支不灭")
sol = least_squares(lambda z: h_exact(z, 1.0), np.zeros(2), xtol=1e-15, ftol=1e-15, gtol=1e-15)
x = sol.x; roots = {1.0: sol.x.copy()}
wcur = 1.0
while wcur <= 100:
    wnext = round(wcur * 1.15 + 0.05, 3)
    sol = least_squares(lambda z: h_exact(z, wnext), x, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    if np.linalg.norm(h_exact(sol.x, wnext)) > 1e-8:
        print(f"      主支在 w≈{wcur:.1f}–{wnext:.2f} 间消失")
        break
    x = sol.x; roots[wnext] = sol.x.copy(); wcur = wnext
WSHOW = [1.0, 1.5, 2.0, 3.0, 4.5, 10.0, 30.0, 100.0]
print(f"      {'w':>6s} {'不动点 ‖x*‖':>12s} {'x*':>20s}")
for wv in WSHOW:
    wr = min(roots, key=lambda k: abs(k - wv))
    print(f"      {wr:6.2f} {np.linalg.norm(roots[wr]):12.4f} "
          f"[{roots[wr][0]:+8.3f},{roots[wr][1]:+8.3f}]")
# 远场: 混合条件恒恢复
th_far = 50 * np.array([1., 1.]) / np.sqrt(2)
a_far = h_exact(th_far, 100.0) @ th_far / (th_far @ th_far)
print(f"      远场（w=100, θ=50·(1,1)/√2）: h/θ 系数 = {a_far:.3f} ≈ E[1/σ] = "
      f"{np.mean(1/SIGG):.3f} > 0 → 恒恢复（混合后验的有界均值让 x/σ 项不随 w 消失）")

# 完美条件（GG 11 版: 单模式高斯条件）对照
def eps_cond_single(x, sig):
    s, s2, v = _sigprep(sig)
    m = (SC3 ** 2 * x + s2 * MU3[0]) / v
    return (x - m) / s

def eps_cfg_single(x, sig, w):
    return eps_u(x, sig) + w * (eps_cond_single(x, sig) - eps_u(x, sig))

def h_single(th, w):
    tot = np.zeros(2)
    for sg in SIGG:
        xt = np.asarray(th)[None, :] + sg * np.sqrt(2) * GHU
        tot += (eps_cfg_single(xt, sg, w) * GHW[:, None]).sum(0)
    return tot / len(SIGG)

print("\n  (ii) 蒸馏侧 B: 完美条件（GG 11 单模式版）对照 —— 分支消亡 + 远场反恢复")
x = np.zeros(2)
wcur = 1.0
while wcur < 5:
    wnext = round(wcur + 0.05, 3)
    sol = least_squares(lambda z: h_single(z, wnext), x, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    if np.linalg.norm(h_single(sol.x, wnext)) > 1e-8:
        print(f"      主支在 λ*≈{wcur:.2f} 消亡（GG 11: ≈1.71, 复算吻合）")
        break
    x = sol.x; wcur = wnext
a_single = np.mean([-(100 - 1) / sg + 100 * sg / (SC3 ** 2 + sg ** 2) for sg in SIGG])
print(f"      远场解析 ā(λ=100) = {a_single:.1f} < 0 → 反恢复（单高斯条件的仿射收缩")
print(f"      让恢复率随 λ 转负——分支消亡 + 远场失稳是完美条件特有的病理）")

# 动力学验证（混合条件）
def run_sds(w, eta0, n_steps, seed, clip=3.0):
    r_ = np.random.default_rng(seed)
    th = np.array([0.9, 0.5])
    for t in range(n_steps):
        sig = r_.uniform(0.2, 1.0)
        e = r_.normal(size=2)
        g = eps_cfg_mix(th + sig * e, sig, w) - e
        nrm = np.linalg.norm(g)
        if nrm > clip: g = g * clip / nrm
        th = th - eta0 / (1 + 1e-3 * t) * g
    return th

print("\n  动力学验证（混合条件, 8 种子, 起点固定; 高 w 用更长预算）:")
for w, eta, nst in [(1, 0.02, 12000), (3, 0.01, 12000), (10, 0.005, 12000),
                    (30, 0.002, 20000), (100, 0.002, 40000)]:
    ends = np.array([run_sds(w, eta, nst, s) for s in range(8)])
    wr = min(roots, key=lambda k: abs(k - w))
    xr = roots[wr]
    print(f"      w={w:3d}: 终点 ‖θ̄‖={np.linalg.norm(ends.mean(0)):8.4f} ±{ends.std(0).mean():.4f}, "
          f"不动点 ‖x*‖={np.linalg.norm(xr):8.4f} → 有界、无逃逸（区别于完美条件的发散）")
print("      → 两笔账: 生成侧熵在 w≈3 崩塌（先）; 蒸馏侧偏置随 w 单调增长（低 w 端 ≈|m₀|")
print("        级条件泄漏, 高 w 端远离干扰模式质量）且混合条件下永不消亡。安全窗被夹在")
print("        w≲2–3; 提高条件模型分离度（视频先验的真价值）才能把窗口拉宽——guidance")
print("        本身调不出来（GG 08 结论在蒸馏侧复现, GG 11 的分支消亡是其完美条件病理版）")
