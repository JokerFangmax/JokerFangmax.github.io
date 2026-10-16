"""WM_03 toy: teacher forcing 为何低估 drift —— 训练/推理的条件位姿误差分布错配

Memory Forcing 的两个论断做成最小定量实验:
  (A) teacher-forced 训练里, 条件帧全是 GT, 条件位姿误差恒 0; 而推理时
      AR rollout 的位姿估计误差是随机游走, 条件窗口的位姿误差随 t 无界增长
      —— 训练分布完全没有覆盖推理时实际遇到的 drift。
  (B) Chained Forward Training (CFT) 把模型自己的预测链进后续窗口当条件,
      训练分布里就有了 drift 样本; 链越长, 覆盖的 drift 范围越大。
  (C) 为什么 drift 大会「把模型推向 spatial memory」: temporal 条件的误差
      随 rollout 无界累积; spatial 记忆经增量 3D 重建 + cross-window scale
      alignment 后误差有界(对齐到全局坐标系, 残差不随序列长度增长)。
      drift 越大, temporal 越不可靠而 spatial 不变 —— CFT 在大 drift 区间
      训练 = 教模型在 temporal 不可靠时去信 spatial。

抽象(剥离生成器, 只留位姿链):
  - 真实位姿 p_t 由动作驱动; 「模型」输出帧的位姿估计带每步误差
    e_step ~ N(0, σ²) —— rollout 误差 e_t = Σ e_step(随机游走)。
  - temporal 条件窗口 L 帧: 条件位姿误差 = 窗口内帧的 |e|。
  - CFT 链: 依论文 Algorithm 1, 窗口 j 的末帧由窗口 j 自身去噪生成(无梯度),
    成为后续窗口的条件 —— 预测帧的误差 ≈ 其条件窗口误差 + 新增 σ_gen,
    链式传播。
  - spatial 对齐: 每窗口经重叠帧最小二乘对齐到全局系, 残差建模为有界
    N(0, σ_align²)(不随 t 累积)。

输出:
  1) 推理 rollout 各时刻的 temporal 条件误差分位数(无界增长)
  2) 三种训练协议(TF / CFT-链4 / CFT-链16)的训练误差分布对推理分布的
     覆盖率: P(推理条件误差 ≤ 训练分布 95% 分位)
  3) temporal vs spatial 条件误差随 t 的对比(无界 vs 有界)
"""
import numpy as np

rng = np.random.default_rng(23)

SIGMA_STEP = 0.15   # 每步位姿估计误差 (格)
SIGMA_GEN = 0.15    # CFT 无梯度去噪生成一帧的新增误差
SIGMA_ALIGN = 0.30  # spatial 对齐后的有界残差
L = 16              # 条件窗口长度 (论文同款)
T = 1500            # rollout 长度 (WorldMem 数据集同款 1500 帧)
CHAINS = [4, 16]    # CFT 链长(窗口数)
N_ROLLOUTS = 200


def rollout_errors():
    """AR rollout: 返回各帧位姿估计误差 |e_t|。"""
    e = np.abs(np.cumsum(rng.normal(0, SIGMA_STEP, T)))
    return e


def temporal_cond_err(e, t):
    """t 时刻 temporal 条件窗口(前 L 帧)的平均位姿误差。"""
    lo = max(0, t - L)
    return e[lo:t].mean() if t > lo else 0.0


def cft_train_distribution(chain):
    """模拟论文 Algorithm 1: 链长 chain 个窗口的 CFT 训练条件误差样本。

    窗口 j = 0..chain-1, 每窗 L 帧。窗口 j 的末帧由该窗自身去噪生成:
    其误差 = 窗口内条件误差均值 + σ_gen(简化: 链式累积一帧/窗)。
    训练条件误差 = 各窗口里所有条件帧(含链入的预测帧)的误差样本。
    """
    e = np.abs(np.cumsum(rng.normal(0, SIGMA_STEP, chain * L + 1)))
    samples = []
    for j in range(chain):
        w0 = j * L
        # 窗口 j 的条件帧: 窗口开头是 GT(误差 e), 末尾链入的预测帧误差
        # 由窗口 j-1 的生成引入, 简化为窗内 GT 误差 + σ_gen 的累积
        pred_err = e[w0:w0 + L] + SIGMA_GEN * np.sqrt(np.arange(1, L + 1))
        samples.append(e[w0:w0 + L].tolist())      # GT 条件帧
        samples.append(pred_err.tolist())          # 链入预测帧
    return np.array([s for chunk in samples for s in chunk])


def main():
    # ---- (A) 推理 rollout 的 temporal 条件误差(无界增长) ----
    all_err = []
    for _ in range(N_ROLLOUTS):
        all_err.append(rollout_errors())
    all_err = np.array(all_err)                     # (N, T)
    cond = np.array([[temporal_cond_err(all_err[r], t) for t in range(1, T)]
                     for r in range(N_ROLLOUTS)])    # (N, T-1)

    print("== (A) 推理 AR rollout: temporal 条件误差分位数 (无界增长) ==")
    print(f"{'t':>6} | {'P50':>8} | {'P95':>8} | {'P99':>8}")
    for t in [16, 64, 128, 256, 512, 1024, 1499]:
        col = cond[:, t - 1]
        print(f"{t:>6} | {np.percentile(col, 50):8.3f} | "
              f"{np.percentile(col, 95):8.3f} | {np.percentile(col, 99):8.3f}")

    # ---- (B) 训练分布覆盖率 ----
    print("\n== (B) 训练分布对推理 drift 的覆盖率 ==")
    print("     覆盖率 = P_t( 推理条件误差 ≤ 训练分布 95% 分位 )")
    # Teacher forcing: 条件全 GT → 训练误差恒 0
    q_tf = 0.0
    cov_tf = (cond <= q_tf).mean()
    print(f"{'TeacherForcing':>14} | q95={q_tf:6.3f} | 覆盖率 {cov_tf:6.1%}"
          f"  (t=1 之外全部超出)")
    for chain in CHAINS:
        pool = np.concatenate([cft_train_distribution(chain) for _ in range(50)])
        q95 = np.percentile(pool, 95)
        cov = (cond <= q95).mean()
        # 分段覆盖率: 前段(链覆盖范围内) vs 后段
        in_range = cond[:, :chain * L]
        out_range = cond[:, chain * L:]
        print(f"{f'CFT-chain{chain}':>14} | q95={q95:6.3f} | 覆盖率 {cov:6.1%}"
              f"  (前 {chain*L} 步 {np.mean(in_range <= q95):5.1%},"
              f" 之后 {np.mean(out_range <= q95) if out_range.size else 0:5.1%})")

    # ---- (C) temporal vs spatial 条件误差 ----
    print("\n== (C) temporal(无界) vs spatial(对齐后有界) 条件误差 P95 ==")
    print(f"{'t':>6} | {'temporal P95':>12} | {'spatial P95':>12} | {'temporal/spatial':>15}")
    for t in [16, 64, 256, 1024, 1499]:
        tp95 = np.percentile(cond[:, t - 1], 95)
        sp95 = np.percentile(np.abs(rng.normal(0, SIGMA_ALIGN, 5000)), 95)
        print(f"{t:>6} | {tp95:12.3f} | {sp95:12.3f} | {tp95 / sp95:15.1f}×")


if __name__ == "__main__":
    main()
