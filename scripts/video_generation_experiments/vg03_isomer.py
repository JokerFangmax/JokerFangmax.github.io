"""
VG 03 toy 实验 2：ISOMER 两个机制的定量检验（v3）
==================================================

按论文+代码核对后的事实（arXiv:2405.20343 v3 + AiuniAI/Unique3D），ISOMER 的真实机制：

  (a) 初始化（scripts/mesh_init.py + 论文 Eq.1）：对前/后伪法线图做积分得深度
      （d(i,j) = Σ_t n_x(t,j)），逐像素射线 → 前后 join → screened Poisson
      (depth 6) → 简化 2000 面。伪法线来自扩散模型、**非无旋**（不是任何
      高度场的梯度场），积分路径依赖 → 论文的 trick：积分前随机旋转图像、
      重复多次取均值（代码 target_iteration_count=96）。
  (b) ExplicitTarget 精修（论文核心创新，mesh_reconstruction/opt.py）：每个
      顶点的显式优化目标 = 各可见视角监督信号的加权和，权重 W ∝ cos²(顶点
      法线, 视线方向)——论文论证：投影面积 ∝ cos、预测精度也 ∝ cos（附录 D
      实测 Pearson −0.304）。纯几何的 per-vertex 置信度加权。
      （另注：代码里 mesh 优化器是 Adam 变体 lr=0.3 betas=(0.8,0.8)、带
      Laplacian 梯度平滑——论文写 "SGD" 不准确，见勘误节。）

实验 2a —— 法线积分的误差结构与旋转平均
  高度场 h(x,y)（整频率三角多项式，严格周期），周期前向差分梯度 + iid 噪声
  （伪法线非无旋的抽象）。三种重建共享同一套离散梯度与同一份噪声实现：
    1. 朴素列积分（沿 axis0 逐行累加，首行锚定真值——N 个数，信息量可忽略，
       焦点是噪声累积）；
    2. 论文的随机方向积分平均：沿随机方向 u 的射线积分（从边界真值锚点
       到每个像素累加 g·u）——数学上等价于"旋转图像再列积分"，避免了
       旋转-插值-转回的数值损失（v1 教训：三次双线性插值的误差 ≈ 高度场
       本身；v2 教训：非周期场 + 周期差分在接缝产生 O(1) 垃圾）。同一份
       噪声场重复 R 次随机方向取均值（与论文同一张法线图多次旋转一致）；
    3. 周期 DFT 泛函投影（Frankot–Chellappa 的离散精确版）：噪声梯度到
      可积场的 L2 投影，闭式（离散差分算子的符号除子），线性最优估计的
      理论锚（全局均值锚定，1 个数）。
  预测（跑前写下）：
    1 的误差 = 纯随机游走，RMS = σ√(E[i]) = σ√((N−1)/2)，行 RMS ∝ √i；
    2 单次方向积分误差与 1 同量级或更大（斜方向路径更长）；
      R 次平均按 ~1/√R 下降（不同方向的路径投影部分去相关），但**同一份
      噪声场**意味着存在相关残差 → 可能存在 R→∞ 的地板，地板是否等于
      3 的投影（旋转平均 = Poisson 投影的蒙特卡洛估计？）——待实测；
    3 = 线性最优地板。

实验 2b —— ExplicitTarget 的 cos² 加权 = 逆方差加权
  闭合曲线（2D"表面"）上 M 个点，4 个正交视图各给可见点一个带噪目标位置；
  噪声模型 σ_view ∝ 1/cos（掠射视角更不准，论文论证的抽象）。四种加权
  重建：均匀 / cos / cos² / oracle 逆方差（1/σ²）。
  预测：cos² ≈ oracle < cos < 均匀——cos² 恰是 σ∝1/cos 下的逆方差最优权，
  论文的几何论证等价于一个隐含的异方差噪声模型。
  场景 2（某一视图目标整体偏移 δ，模拟多视图扩散的生成性不一致）：
  预测——正对该错误视图的顶点 w≈1 吃满 δ（纯几何权重无法发现"自信地
  错"），其余顶点被多视图权重稀释；整体 RMS 增量 ~ E[w_wrong²]^½·δ。
  即：cos² 对噪声最优、对语义错误无害但也救不了正面区域。
"""
import numpy as np
from scipy import ndimage

# ================= 实验 2a：法线积分 + 旋转平均 =================

def make_height(N=64):
    """整频率三角多项式：严格周期，周期前向差分无接缝跳变。"""
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return (0.8 * np.sin(2 * np.pi * X + 0.7) * np.cos(2 * np.pi * Y)
            + 0.5 * np.sin(2 * np.pi * 2 * Y + 1.9)
            + 0.3 * np.cos(2 * np.pi * (X + Y)))

def noisy_grad_per(h, sigma, seed=123):
    """周期前向差分梯度 + iid 噪声。"""
    rng = np.random.default_rng(seed)
    gy = np.roll(h, -1, axis=0) - h
    gx = np.roll(h, -1, axis=1) - h
    return gy + rng.normal(0, sigma, h.shape), gx + rng.normal(0, sigma, h.shape)

def integrate_columns(gy, h_anchor):
    """沿 axis0 逐行累加（不含当前项——差分是 [i,i+1) 上的增量），首行锚定真值。
    噪声-free 时精确还原 h；有噪声时误差 = 纯随机游走。"""
    d = np.zeros_like(gy)
    d[1:] = np.cumsum(gy[:-1], axis=0)
    return d + h_anchor[0][None, :]

def depth_along_dir(gy, gx, h, ang_deg, delta=0.25):
    """沿方向 u 的射线积分：从每个像素沿 −u 回溯到网格边界，累加 g·u·Δ，
    锚 = 边界处的真值 h。等价于"旋转 ang 再列积分"。
    前向差分 g[i] 住在区间中点：gy 在 (i+0.5, j)、gx 在 (i, j+0.5)——
    必须在中点网格上采样，否则错位半像素的系统性偏置 ~0.09（v3 教训）。"""
    N = h.shape[0]
    th = np.deg2rad(ang_deg)
    uy, ux = np.cos(th), np.sin(th)
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    pi_ = I.astype(float).copy()
    pj = J.astype(float).copy()
    inside = np.ones_like(I, bool)
    acc = np.zeros_like(pi_)
    anchor = np.zeros_like(pi_)
    for _ in range(int(4 * N / delta) + 2):
        fy = ndimage.map_coordinates(gy, [pi_ - 0.5, pj], order=1, mode="nearest")
        fx = ndimage.map_coordinates(gx, [pi_, pj - 0.5], order=1, mode="nearest")
        acc += np.where(inside, uy * fy + ux * fx, 0.0) * delta
        pi_ -= uy * delta
        pj -= ux * delta
        out = inside & ((pi_ < 0) | (pi_ > N - 1) | (pj < 0) | (pj > N - 1))
        if out.any():
            hi = ndimage.map_coordinates(h, [pi_, pj], order=1, mode="nearest")
            anchor[out] = hi[out]
            inside &= ~out
        if not inside.any():
            break
    return anchor + acc

def depth_by_rotation_avg(gy, gx, h, R, seed=77):
    rng = np.random.default_rng(seed)
    acc = np.zeros_like(h)
    for _ in range(R):
        acc += depth_along_dir(gy, gx, h, rng.uniform(0, 360))
    return acc / R

def depth_by_dft_projection(gy, gx, h):
    """周期 DFT L2 投影：min_e ||D e − g||²，D = 周期前向差分算子。
    离散算子符号 a(k)=e^{−iθ}−1 精确已知 → 闭式解，无边界延拓争议。"""
    N = h.shape[0]
    GY = np.fft.fft2(gy)
    GX = np.fft.fft2(gx)
    ky = 2 * np.pi * np.fft.fftfreq(N)
    kx = 2 * np.pi * np.fft.fftfreq(N)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    a = np.exp(1j * KY) - 1       # roll(h,-1) 的 DFT 符号是 e^{+iθ}（v3 曾写反）
    b = np.exp(1j * KX) - 1
    den = np.abs(a) ** 2 + np.abs(b) ** 2
    den[0, 0] = 1
    H = (np.conj(a) * GY + np.conj(b) * GX) / den
    H[0, 0] = 0
    d = np.fft.ifft2(H).real
    return d + (h.mean() - d.mean())       # 常数自由度用真值均值固定（1 个数）

def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))

def exp2a():
    N, sigma = 64, 0.15
    h = make_height(N)
    gy, gx = noisy_grad_per(h, sigma)
    print("=" * 72)
    print("实验 2a：法线积分的误差结构（σ=%.2f, N=%d, 高度场 RMS=%.3f）" % (sigma, N, rms(h)))
    print("=" * 72)

    # 无噪声自检：三种算子在 σ=0 时都应精确还原（射线积分允许插值级小误差）
    gy0, gx0 = noisy_grad_per(h, 0.0)
    print(f"无噪声自检: 列积分 {rms(integrate_columns(gy0,h)-h):.2e}  "
          f"射线积分 {rms(depth_along_dir(gy0,gx0,h,37.0)-h):.2e}  "
          f"DFT投影 {rms(depth_by_dft_projection(gy0,gx0,h)-h):.2e}")

    # 1. 朴素列积分（单噪声场的列间涨落 ~5%，多种子平均后才可比对解析预测）
    d_naive = integrate_columns(gy, h)
    e_naive = rms(d_naive - h)
    e_naive_ms = np.mean([rms(integrate_columns(noisy_grad_per(h, sigma, seed=500+s)[0], h) - h)
                          for s in range(20)])
    print(f"朴素列积分:     RMS = {e_naive:.4f}（本噪声场）  20 种子均值 = {e_naive_ms:.4f}"
          f"   随机游走预测 σ√((N−1)/2) = {sigma*np.sqrt((N-1)/2):.4f}")
    row_rms = np.sqrt(np.mean((d_naive - h) ** 2, axis=1))
    print("  行 RMS √i 检验: ", end="")
    for i in [8, 16, 32]:
        print(f"i={i}: {row_rms[i]:.3f} vs σ√i={sigma*np.sqrt(i):.3f}", end="  ")
    print()

    # 2. 随机方向积分平均（同一份噪声场，R 个随机方向）
    print("随机方向积分平均（= 论文的随机旋转积分，同一噪声场）:")
    for R in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        d_rot = depth_by_rotation_avg(gy, gx, h, R)
        print(f"  R={R:3d}: RMS = {rms(d_rot-h):.4f}")

    # 3. DFT 投影（L2 最优锚；同样做 20 种子平均给地板一个稳定值）
    d_fc = depth_by_dft_projection(gy, gx, h)
    e_fc_ms = np.mean([rms(depth_by_dft_projection(*noisy_grad_per(h, sigma, seed=500+s), h) - h)
                       for s in range(20)])
    print(f"DFT-L2投影:     RMS = {rms(d_fc-h):.4f}（本噪声场）  20 种子均值 = {e_fc_ms:.4f}"
          f"   —— 线性最优估计的地板")


# ================= 实验 2b：ExplicitTarget cos² 加权 =================

def make_curve(M=400):
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    r = 1.0 + 0.25 * np.sin(3 * th) + 0.1 * np.cos(5 * th + 1)
    p = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    t = np.gradient(p, axis=0)
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    if np.mean(np.sum(n * p, axis=1)) < 0:
        n = -n
    return p, n

def exp2b():
    M, sigma0, delta = 400, 0.05, 0.15
    p, n = make_curve(M)
    views = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])   # 0/90/180/270
    cos_all = np.abs(n @ views.T).T                        # (4, M) |cos(法线,视线)|
    vis_all = (n @ -views.T).T > 0

    def run(shift_view=None, n_seeds=50):
        errs = {k: [] for k in ["uniform", "cos", "cos2", "oracle"]}
        bin_err = {b: [] for b in ["0.9-1.0", "0.6-0.9", "0.3-0.6", "<0.3"]}
        for s in range(n_seeds):
            r = np.random.default_rng(9000 + s)
            tgt = np.full((4, M, 2), np.nan)
            for v in range(4):
                sig_v = sigma0 / np.clip(cos_all[v], 0.15, 1.0)
                t_v = p + r.normal(0, 1, (M, 2)) * sig_v[:, None]
                if shift_view is not None and v == shift_view:
                    t_v = t_v + views[shift_view] * delta
                tgt[v] = np.where(vis_all[v][:, None], t_v, np.nan)
            sig = sigma0 / np.clip(cos_all, 0.15, 1.0)
            for name, w in [("uniform", np.ones((4, M))),
                            ("cos", cos_all.copy()),
                            ("cos2", cos_all.copy() ** 2),
                            ("oracle", 1.0 / np.clip(sig, 1e-3, None) ** 2)]:
                w = np.where(np.isnan(tgt[:, :, 0]), 0.0, w)
                est = np.nansum(tgt * w[:, :, None], axis=0) / np.maximum(
                    w.sum(0)[:, None], 1e-12)
                visible = w.sum(0) > 0
                e_full = np.sum((est - p) ** 2, axis=1)   # (M,) 逐点误差平方
                errs[name].append(np.sqrt(e_full[visible].mean()))
                if name == "cos2" and shift_view is not None:
                    # 分箱只在错误视图的可见点内（|cos| 会把背对的不可见点
                    # 也算成 cos≈1，稀释正面 bin——v3 修正）
                    vis_wrong = vis_all[shift_view]
                    cw = cos_all[shift_view][vis_wrong]   # 对错误视图的正面度
                    e_b = e_full[vis_wrong]
                    for lo, hi, key in [(0.9, 1.01, "0.9-1.0"), (0.6, 0.9, "0.6-0.9"),
                                        (0.3, 0.6, "0.3-0.6"), (-0.01, 0.3, "<0.3")]:
                        m = (cw >= lo) & (cw < hi)
                        if m.any():
                            bin_err[key].append(np.sqrt(e_b[m].mean()))
        out = {k: (np.mean(v), np.std(v)) for k, v in errs.items()}
        bins = {k: (np.mean(v), np.std(v)) for k, v in bin_err.items() if v}
        return out, bins

    print()
    print("=" * 72)
    print("实验 2b：ExplicitTarget 加权（σ_view ∝ 1/cos，M=%d 点，4 正交视图）" % M)
    print("=" * 72)
    res, _ = run()
    print("场景 1（只有异方差噪声）:")
    for k, (m, s) in res.items():
        print(f"  {k:8s}: RMS = {m:.4f} ± {s:.4f}")
    print("  预测：cos² ≈ oracle（逆方差最优）< cos < uniform")

    res2, bins = run(shift_view=1)
    print(f"场景 2（视图 90° 目标整体偏移 δ={delta} —— 多视图生成性不一致）:")
    for k, (m, s) in res2.items():
        print(f"  {k:8s}: RMS = {m:.4f} ± {s:.4f}")
    print(f"  cos² 逐点误差按对错误视图的正面度分箱（δ={delta}）:")
    for k, (m, s) in bins.items():
        print(f"    cos∈{k:8s}: RMS = {m:.4f} ± {s:.4f}")
    print("  预测：正面 bin 吃满 δ（几何权重发现不了'自信地错'），侧面 bin 回到场景 1 水平")


if __name__ == "__main__":
    exp2a()
    exp2b()
