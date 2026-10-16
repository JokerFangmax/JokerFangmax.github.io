"""WM_18 toy: 3D 缓存 vs 2D 帧缓存——构造几何记忆 vs 视角依存的滑动窗口

机制内核(GEN3C, arXiv:2503.03751):
  记忆 = 显式 3D 点云缓存(世界坐标, 深度反投影 + 跨帧累积);
  条件 = 缓存沿用户相机轨迹的**解析投影**渲染(RGB + disocclusion mask);
  相机控制是构造出来的(投影公式), 不是学出来的.

对照(相机参数条件路线, 如 CameraCtrl):
  记忆 = 最近 K 帧的视角依存内容(滑动窗口);
  生成新视角必须由网络从 2D 参考帧隐式推断几何并执行相机指令——
  单帧深度误差未标定 + 相机执行有偏差(Plücker 嵌入只是输入, 映射要学).

协议(第一部分):
  40×40 m 世界(400 地面点 + 6 个盒子簇 60 点), 相机高 1.6 m, FOV 90°(半角 45°).
  探索相: 前进 25 步(每步 1.5 m, 航向抖动 ±3°), 每步观测写入两种记忆.
  探针机动(不写入, 纯召回): 小幅前进 1.5 m / 横移 8 m / 原地转 90° / 回到起点(重访).
  指标: 目标视角下可见点的 覆盖率(记忆中存在拷贝的比例)
        与 像素误差(预测投影位置 − 真值投影位置, 512px 画幅).

四种记忆:
  Cache3D   全历史累积点云; 每拷贝带 2% 相对深度噪声(反投影); 预测 = 拷贝投影均值;
            渲染用**精确用户位姿**(构造相机控制)——GEN3C 的路线
  Cache2D   最近 K=4 帧可见内容; 从位姿最接近的参考帧重合成: 单帧深度噪声 2%
            (与 Cache3D 相同 → 隔离相机执行项) + 相机执行误差(位置 σ=0.15m,
            航向 σ=1°)——Plücker 条件路线
  Cache2D+  同上但单帧深度噪声 5%(未标定的隐式几何推断)
  NoMem     无记忆(生成先验只能幻觉) → 覆盖 0

第二部分(多视图融合, 对应论文 Table 6 显式融合 vs 逐视图+网络融合):
  一面 8m 外的点墙, 参考视图 A(干净, 2% 深度噪声) / B(残余失配: 3% 尺度
  + 0.2m 径向偏移, 模拟 Eq.4 对齐残差 / DUSt3R 式跨视图失配), 查询居中.
  显式融合 → 两份拷贝同时可见(鬼影 = 两拷贝投影间距);
  逐视图 + 网络融合 → 单假设(误差 = 所选视图自身误差, 好坏两种情况都报).
"""

import numpy as np

rng = np.random.default_rng(18)

W_IMG = 512.0               # 图像宽(像素); 归一化坐标 ∈ [−1,1] → 每单位 256 px
FOV_HALF = np.pi / 4        # 90° FOV
F = 1.0 / np.tan(FOV_HALF)  # 焦距(归一化)
CAM_H = 1.6
SIGMA_CACHE = 0.02          # 缓存反投影: 2% 相对深度噪声(对齐后, DAV2 级)
SIGMA_2D = 0.02             # 2D 路线单帧深度噪声(与缓存同 → 隔离相机项)
SIGMA_2D_LOOSE = 0.05       # 2D 路线单帧深度噪声(未标定版)
SIGMA_POS = 0.15            # 2D 路线相机执行误差(米/每水平轴)
SIGMA_YAW = np.deg2rad(1.0)
K_WINDOW = 4
N_EPISODES = 40


# ---------- 相机几何 ----------

def cam_axes(yaw):
    d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    r = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    return d, r


def visible_mask(P, c, yaw):
    """水平 FOV 内可见(无遮挡——点稀疏, 如实简化)."""
    v = P - c
    ang = np.arctan2(v[:, 1], v[:, 0])
    diff = np.abs(np.mod(ang - yaw + np.pi, 2 * np.pi) - np.pi)
    return diff < FOV_HALF


def project(P, c, yaw):
    """针孔投影 → 像素坐标(图像中心为原点)."""
    d, r = cam_axes(yaw)
    v = P - c
    z = v @ d
    x = v @ r
    y = v[:, 2]
    return np.column_stack([F * x / z, F * y / z]) * (W_IMG / 2)


def unproject_noise(P, c, yaw, sigma):
    """沿视线方向的相对深度噪声反投影: p̂ = c + (1+ε)(p−c)."""
    v = P - c
    scale = 1.0 + rng.normal(0, sigma, len(P))
    return c + v * scale[:, None]


# ---------- 世界 ----------

def make_world():
    ground = np.column_stack([rng.uniform(-20, 20, 400),
                              rng.uniform(-20, 20, 400),
                              np.zeros(400)])
    pts = [ground]
    for _ in range(6):
        ctr = rng.uniform(-15, 15, 3)
        ctr[2] = rng.uniform(1.0, 3.0)
        off = rng.uniform(-1, 1, (10, 3))
        off[:, 2] *= 0.8
        pts.append(ctr + off)
    return np.vstack(pts)


# ---------- 第一部分: 缓存 vs 滑动窗口 ----------

def run_episode():
    P = make_world()
    yaw = rng.uniform(0, 2 * np.pi)
    c = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), CAM_H])
    start_c, start_yaw = c.copy(), yaw

    cache = {}      # Cache3D: 点索引 → 拷贝列表(世界坐标)
    window = []     # Cache2D: 最近 K 帧的 (位姿, 可见点索引)

    for _ in range(25):                      # 探索相: 边走边写
        m = visible_mask(P, c, yaw)
        idx = np.where(m)[0]
        noisy = unproject_noise(P[idx], c, yaw, SIGMA_CACHE)
        for j, i in enumerate(idx):
            cache.setdefault(i, []).append(noisy[j])
        window.append((c.copy(), yaw, set(idx.tolist())))
        if len(window) > K_WINDOW:
            window.pop(0)
        d, _ = cam_axes(yaw)
        nxt = c + 1.5 * d
        if np.linalg.norm(nxt[:2]) > 14:     # 快出界时朝世界中心转向(保证轨迹留在场内)
            to_c = np.arctan2(-c[1], -c[0])
            yaw = to_c + rng.uniform(-0.2, 0.2)
            d, _ = cam_axes(yaw)
            nxt = c + 1.5 * d
        c = nxt
        yaw += rng.uniform(-np.deg2rad(3), np.deg2rad(3))
    window = window[-K_WINDOW:]

    d, _ = cam_axes(yaw)
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    probes = {
        "小幅前进 1.5m":  (c + 1.5 * d, yaw),
        "横移 8m":        (c + 8.0 * right, yaw),
        "原地转 90°":     (c.copy(), yaw + np.pi / 2),
        "回起点(重访)":   (start_c.copy(), start_yaw),
    }

    out = {}
    for name, (qc, qyaw) in probes.items():
        vis = np.where(visible_mask(P, qc, qyaw))[0]
        n = len(vis)
        if n == 0:                # 探针朝向场外空域(罕见): 跳过该探针
            continue
        gt_pix = project(P[vis], qc, qyaw)

        # --- Cache3D: 全历史拷贝, 精确位姿 ---
        cov3, err3 = 0.0, []
        for j, i in enumerate(vis):
            if i in cache:
                cov3 += 1
                pred = project(np.array(cache[i]).mean(0)[None], qc, qyaw)[0]
                err3.append(np.linalg.norm(pred - gt_pix[j]))
        cov3 /= n

        # --- Cache2D / Cache2D+: 最近 K 帧 + 相机执行误差(整个探针抽一次) ---
        qc_exec = qc[:2] + rng.normal(0, SIGMA_POS, 2)
        qc_exec = np.array([qc_exec[0], qc_exec[1], CAM_H])
        qyaw_exec = qyaw + rng.normal(0, SIGMA_YAW)
        # 参考帧 = 窗口中位姿(位置+航向)最接近查询者
        def pose_dist(fr):
            dc = np.linalg.norm(fr[0][:2] - qc[:2])
            da = abs(np.mod(fr[1] - qyaw + np.pi, 2 * np.pi) - np.pi)
            return dc + 2.0 * da
        ref = min(window, key=pose_dist)

        res2 = {}
        for tag, sig in [("Cache2D", SIGMA_2D), ("Cache2D+", SIGMA_2D_LOOSE)]:
            cov, err = 0.0, []
            for j, i in enumerate(vis):
                if i in ref[2]:
                    cov += 1
                    p_hat = unproject_noise(P[i][None], ref[0], ref[1], sig)[0]
                    pred = project(p_hat[None], qc_exec, qyaw_exec)[0]
                    err.append(np.linalg.norm(pred - gt_pix[j]))
            res2[tag] = (cov / n, err)
        out[name] = {"n": n, "Cache3D": (cov3, err3), **res2}
    return out


def part1():
    schemes = ["Cache3D", "Cache2D", "Cache2D+"]
    agg = {}       # (probe, scheme) -> ([cov], [err 逐点])
    for _ in range(N_EPISODES):
        for name, res in run_episode().items():
            for s in schemes:
                agg.setdefault((name, s), ([], []))
                cov, err = res[s]
                agg[(name, s)][0].append(cov)
                agg[(name, s)][1].extend(err)

    print("===== (1) 3D 缓存 vs 2D 滑动窗口: 覆盖率(%) / 像素误差中位数(px, 512px 画幅) =====")
    print(f"{'探针机动':<12} | " + " | ".join(f"{s:>18}" for s in schemes) + " | NoMem 覆盖")
    for name in ["小幅前进 1.5m", "横移 8m", "原地转 90°", "回起点(重访)"]:
        cells = []
        for s in schemes:
            covs, errs = agg[(name, s)]
            cells.append(f"{np.mean(covs) * 100:5.1f}% / {np.median(errs):6.2f}px")
        print(f"{name:<12} | " + " | ".join(f"{c:>18}" for c in cells) + " |   0.0%")
    print("\n读法: 覆盖率差异来自记忆范围(全历史 vs 最近 K 帧)——重访时窗口在别处;")
    print(" Cache2D 与 Cache3D 深度噪声相同(2%), 误差差 = 相机执行项;")
    print(" Cache2D+ 与 Cache2D 相机误差相同, 误差差 = 未标定深度项;")
    print(" 原地转 90° 的覆盖缺口 = disocclusion——记忆只装观测过的区域, 缺口靠生成先验补;")
    print(" 转 90° 下 2D 路线的条件点是两近垂直光锥的边缘交区, 视差病态(深度投影除以")
    print(" 趋零的 z), 误差发散——大幅旋转下「从 2D 帧重合成」在几何上就是坏条件的.")


# ---------- 第二部分: 显式融合 vs 逐视图 + 网络融合 ----------

def part2():
    # 室内尺度几何(RE10K 式): 5m 外一面点墙, 基线 2–3.5m, 查询居中
    xs = np.arange(-3, 3.01, 0.2)
    zs = np.arange(0.3, 2.31, 0.2)
    wall = np.array([[x, 5.0, z] for x in xs for z in zs])
    yaw = np.pi / 2                      # 看向 +y
    A = np.array([-2.0, 0.0, CAM_H])
    B = np.array([1.5, 0.0, CAM_H])
    Q = np.array([0.0, 0.0, CAM_H])

    copies_A = unproject_noise(wall, A, yaw, SIGMA_CACHE)
    v = wall - B                          # B: 残余失配 = 4% 尺度 + 0.25m 径向偏移 + 2% 噪声
    dist = np.linalg.norm(v, axis=1)
    unit = v / dist[:, None]
    copies_B = B + 1.04 * v + 0.25 * unit \
        + unit * (rng.normal(0, SIGMA_CACHE, len(wall)) * dist)[:, None]

    pix = project(wall, Q, yaw)
    pixA = project(copies_A, Q, yaw)
    pixB = project(copies_B, Q, yaw)
    errA = np.linalg.norm(pixA - pix, axis=1)
    errB = np.linalg.norm(pixB - pix, axis=1)
    ghost = np.linalg.norm(pixA - pixB, axis=1)

    print("\n===== (2) 多视图融合: 显式融合 vs 逐视图 + 网络融合 (对应论文 Table 6) =====")
    print(f"参考视图 A(对齐良好, 2% 深度噪声) 自身误差:        {errA.mean():5.2f} px (p95 {np.percentile(errA, 95):5.2f})")
    print(f"参考视图 B(残余失配 4% 尺度 + 0.25m 偏移) 自身误差:  {errB.mean():5.2f} px (p95 {np.percentile(errB, 95):5.2f})")
    print(f"显式融合鬼影(两份拷贝投影间距, 必然同时可见):      {ghost.mean():5.2f} px (p95 {np.percentile(ghost, 95):5.2f})")
    print(f"逐视图 + 网络融合: 选对(A) = {errA.mean():.2f} px, 选错(B) = {errB.mean():.2f} px")
    print("\n读法: 显式融合把跨视图失配直接焊进点云——两份拷贝都在渲染结果里,")
    print("鬼影幅度 = 失配的投影量, 与查询视角无关地必然出现; 逐视图路线把失配")
    print("变成二选一——上限是好视图自身误差, 下限(选错)也不超过鬼影幅度,")
    print("且网络可按区域选择(论文实测 +2.3 dB PSNR, Table 6).")


if __name__ == "__main__":
    part1()
    part2()
