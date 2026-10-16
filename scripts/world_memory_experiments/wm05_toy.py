"""WM_05 toy: 记忆衰减与 temporal reframing——注意力饥饿的定量标本

机制(WorldPlay §3.3 的最小化模型):
  对当前 chunk 的预测 = 记忆条目的 softmax 加权平均 + 生成噪声:
    logit_i = affinity_i - alpha * penalty_dist(d_i),  w = softmax(logit)
    a_hat_t = w_sp * v_sp + sum_k w_k * v_k + eps_t
  记忆条目(对齐论文配置: 时间记忆 3 chunk + 空间记忆 1 chunk):
    空间记忆 v_sp: 当前位置的已存储值(affinity 优势 beta)——一致性之锚
    时间记忆 v_1..v_3: 最近 3 个 chunk 的生成值(affinity 0, 距离 1/2/3)
  两种位置索引:
    absolute: d_sp(t) = t - t_store, 无界增长
              (超过训练视界 D_max 后叠加二次外推惩罚——RoPE 外推失真的代理)
    reframed: d_sp = 4 固定(恰好排在时间窗之外), 与真实时间差无关

三个测量:
  (1) w_sp 随距离的衰减曲线(记忆衰减): absolute 指数饿死, reframed 恒定
  (2) 长驻漂移: 一直停在锚点 600 chunk, 生成值误差曲线
      absolute -> 误差随机游走(锚失效后只能抄近邻的漂移值)
      reframed -> AR(1) 被锚拉住, 误差进稳态
  (3) 重访误差 vs 离开间隔 Delta: 重访时时间记忆装的是途中别的场景
      (iid N(0, sigma_scene)), 空间记忆是锚的存储值
      absolute -> w_sp 饿死后误差 -> 场景散布; reframed -> 常数小误差
"""

import numpy as np

rng = np.random.default_rng(5)

# ---- 机制参数 ----
D_MAX = 21        # 训练视界: 5 s = 81 帧 / 4 = 21 latents (Wan 级底座的典型值)
ALPHA = 0.15      # RoPE 距离惩罚(每单位距离的 logit 折减)
KAPPA = 0.05      # 超过训练视界后的二次外推惩罚
BETA = 3.0        # 空间记忆的 affinity 优势(几何相关 -> 内容匹配)
SIGMA = 0.08      # 每 chunk 生成噪声
SIGMA_OBS = 0.10  # 存储时的观测噪声
SIGMA_SCENE = 1.0 # 途中其他场景的内容散布(重访时时间记忆里的东西)
D_REFRAME = 4     # reframed 方案下空间记忆的固定距离
RUNS = 200


def penalty(d):
    d = np.asarray(d, dtype=float)
    lin = ALPHA * np.minimum(d, D_MAX)
    quad = KAPPA * np.maximum(d - D_MAX, 0.0) ** 2
    return lin + quad


def spatial_weight(d_sp):
    """空间记忆的 softmax 权重(对手: 3 条时间记忆, 距离 1/2/3, affinity 0)"""
    logits = np.concatenate([BETA - penalty(np.atleast_1d(d_sp)),
                             -penalty(np.array([1.0, 2.0, 3.0]))], axis=0)
    m = logits.max(axis=0, keepdims=True)
    e = np.exp(logits - m)
    return (e[0] / e.sum(axis=0))


def exp1_decay():
    print("=" * 70)
    print("实验 1: 空间记忆注意力权重 w_sp 随真实时间差的衰减")
    print("=" * 70)
    print(f"{'Δ (chunk)':>10} | {'absolute w_sp':>14} | {'reframed w_sp':>14}")
    for d in [1, 4, 8, 16, 21, 30, 40, 60, 100, 160]:
        wa = spatial_weight(d)
        wr = spatial_weight(D_REFRAME)
        print(f"{d:>10} | {wa:>14.4f} | {wr:>14.4f}")
    print(f"(D_MAX={D_MAX}: 超过训练视界后惩罚切换为二次项; "
          f"reframed 恒取 d={D_REFRAME})")


def exp2_drift():
    print()
    print("=" * 70)
    print("实验 2: 长驻漂移——停在锚点 600 chunk, 生成值 RMS 误差")
    print("=" * 70)
    T = 600
    checkpoints = [50, 100, 200, 400, 600]
    sq = {"absolute": np.zeros(T), "reframed": np.zeros(T)}
    for _ in range(RUNS):
        v_sp = rng.normal(0, SIGMA_OBS)          # 锚的存储值(带观测噪声)
        # 公共随机数: 两种方案用同一生成噪声序列(前 3 步外方案轨迹分叉, 无妨)
        eps_seq = rng.normal(0, SIGMA, T)
        for scheme in sq:
            temporal = [0.0, 0.0, 0.0]           # 近邻 chunk 生成值(初始为真值)
            for t in range(1, T + 1):
                d_sp = t if scheme == "absolute" else D_REFRAME
                w = spatial_weight(d_sp)
                wt = np.exp(-penalty(np.array([1.0, 2.0, 3.0])))
                wt = wt / (wt.sum() + np.exp(BETA - penalty(d_sp)))
                a_hat = w * v_sp + np.dot(wt, temporal) + eps_seq[t - 1]
                sq[scheme][t - 1] += a_hat ** 2
                temporal = [a_hat, temporal[0], temporal[1]]
    print(f"{'chunk':>8} | {'absolute RMS':>13} | {'reframed RMS':>13}")
    for tc in checkpoints:
        a = np.sqrt(sq["absolute"][tc - 1] / RUNS)
        r = np.sqrt(sq["reframed"][tc - 1] / RUNS)
        print(f"{tc:>8} | {a:>13.3f} | {r:>13.3f}")


def exp3_revisit():
    print()
    print("=" * 70)
    print("实验 3: 重访误差 vs 离开间隔 Δ (时间记忆装的是途中场景 ~N(0,1))")
    print("=" * 70)
    deltas = [4, 8, 16, 32, 64, 128, 256]
    print(f"{'Δ':>6} | {'absolute MAE':>13} | {'reframed MAE':>13}")
    for d in deltas:
        errs = {"absolute": [], "reframed": []}
        for _ in range(RUNS):
            # 公共随机数: 两种方案用同一组噪声, 消除抽样方差
            v_sp = rng.normal(0, SIGMA_OBS)
            others = rng.normal(0, SIGMA_SCENE, 3)
            eps = rng.normal(0, SIGMA)
            for scheme in errs:
                d_sp = d if scheme == "absolute" else D_REFRAME
                w = spatial_weight(d_sp)
                wt = np.exp(-penalty(np.array([1.0, 2.0, 3.0])))
                wt = wt / (wt.sum() + np.exp(BETA - penalty(d_sp)))
                a_hat = w * v_sp + np.dot(wt, others) + eps
                errs[scheme].append(abs(a_hat))
        print(f"{d:>6} | {np.mean(errs['absolute']):>13.3f} | "
              f"{np.mean(errs['reframed']):>13.3f}")


if __name__ == "__main__":
    exp1_decay()
    exp2_drift()
    exp3_revisit()
