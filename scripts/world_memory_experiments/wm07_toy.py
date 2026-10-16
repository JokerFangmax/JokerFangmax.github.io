"""WM_07 toy: 观测控制下的状态演化——「演化停止」何时发生, 何时是最优解

StEvo-Bench (arXiv:2603.13215) 的两个核心经验事实:
  (1) Table 3: 同样的模型 (Veo 3 / Sora 2 Pro 平均)、同样的过程,
      全程可见时状态推进率 84.6%, 加观测控制 (遮挡/关灯/相机移开) 后掉到 17.4%;
  (2) Table 2: 相机控制模型的状态推进率全部 < 5%;
      VMem 能精确回放初始帧 (Finding 5) 但推进率为 0。

Part A 把生成器的「隐期行为」参数化为 (w, sigma_a, sigma_g):
  w       = 单个 episode 里, 遮挡期间过程继续推进的概率
  sigma_a = 遮挡期重绘的外观漂移 (移开再回来, 物体长歪的程度)
  sigma_g = 隐期轨迹噪声 (推进过程中的抖动; 累计倒退超阈值记物理违规)
  可见期统一按真实速率 r 推进, 失败率 V_FAIL=0.15
  (一次性标定自论文 Table 3 全程可见推进率 84.6%, 之后不再动)。
  判定照抄 StEvo 协议: Progress (隐期推进 >= 50% 真实推进量)
  ∧ Physics (无刺眼违规) ∧ Coherence (外观不跳变) ⇒ Success。
  另有基础失败率 PHYS_BASE / COH_BASE, 模拟探测/渲染毛刺——
  即使隐期完美演化, success 也到不了 100% (论文全程可见 success 仅 46.2%)。

Part B 问「w 为什么这么小」: 设训练语料先验
  q = 「遮挡后揭示帧显示过程推进了」的样本占比。
  平方误差最优的揭示帧预测 = 条件均值 = s0 + q·E[r]·Δ,
  即最优生成器的隐期有效演化率恰好等于 q —— 冻结是数据先验下的最优解, 不是 bug。
  静态场景渲染语料 (WorldPlay 的 3DGS 渲染 / LingBot 的 Unreal 渲染) 的 q≈0;
  StEvo 的及格线是隐期推进 >= 50% ⇒ 需要 q >= 0.5。
  数据构成直接决定判分: 数据修复 (q↑) 与架构修复 (给揭示帧一条绕过无信息遮挡帧的
  状态通道) 缺一不可。
"""

import numpy as np

rng = np.random.default_rng(11)

# ---- Part A 参数 (标定一次, 不逐格拟合) ----
T = 120
HID_START, HID_END = 40, 80          # 遮挡区间 [40, 80)
L = HID_END - HID_START              # 隐期长度 40 步
N_EP = 4000                          # 每组配置的 episode 数
V_FAIL = 0.15                        # 可见期演化失败率 (标定: 全程可见推进率 84.6%)
PHYS_BASE = 0.90                     # 物理判定基础通过率 (渲染/探测毛刺)
COH_BASE = 0.70                      # 连贯性判定基础通过率
PHYS_DRAWDOWN = 0.25                 # 累计倒退超过此值 = 刺眼的物理违规
COH_DRIFT = 0.5                      # 外观漂移超过此值 = 不连贯


def episode_metrics(w, sigma_a, sigma_g, occluded):
    """返回 (progress, physics, coherence, success) 四个通过率。"""
    r = rng.uniform(0.5, 1.5, N_EP)            # 每 episode 真实速率
    started = rng.random(N_EP) > V_FAIL        # 过程是否启动 (可见期演化也可能失败)

    if occluded:
        ran = rng.random(N_EP) < w             # 本 episode 隐期是否继续推进
        steps = rng.normal(0, sigma_g, (N_EP, L)) + ran[:, None] * r[:, None]
        adv_gen = steps.sum(axis=1)
        # 物理违规: 隐期累计轨迹的最大回撤
        cum = np.cumsum(steps, axis=1)
        runmax = np.maximum.accumulate(cum, axis=1)
        drawdown = (runmax - cum).max(axis=1)
        phys_pass = (drawdown <= PHYS_DRAWDOWN) & (rng.random(N_EP) < PHYS_BASE)
        # 连贯性: 遮挡期重绘的外观漂移
        drift = rng.normal(0, sigma_a, N_EP) * np.sqrt(L)
        coh_pass = (np.abs(drift) <= COH_DRIFT) & (rng.random(N_EP) < COH_BASE)
        progress = started & (adv_gen >= 0.5 * r * L)
    else:
        # 全程可见: 隐期同样按真实速率推进 (可见演化当前模型做得到)
        phys_pass = rng.random(N_EP) < PHYS_BASE
        coh_pass = rng.random(N_EP) < COH_BASE
        progress = started

    success = progress & phys_pass & coh_pass
    return progress.mean(), phys_pass.mean(), coh_pass.mean(), success.mean()


def part_a():
    print("=" * 72)
    print("Part A: 隐期行为参数 (w, sigma_a, sigma_g) → StEvo 四项判分")
    print("=" * 72)
    configs = [
        # (名字, w, sigma_a, sigma_g, 对应的真实模型)
        ("Freeze-exact ", 0.00, 0.00, 0.00, "VMem / 相机控制模型 (精确回放)"),
        ("Freeze-redraw", 0.08, 0.06, 0.03, "开源视频模型 (WAN 2.2)"),
        ("Partial      ", 0.20, 0.04, 0.01, "闭源视频模型 (Veo 3)"),
        ("World-sim    ", 1.00, 0.01, 0.005, "假想的真·世界模型"),
    ]
    print(f"\n{'配置':<14} {'w':>4} {'σ_a':>5} {'σ_g':>5} | 观测控制下: "
          f"{'Prog':>6} {'Phys':>6} {'Coh':>6} {'Succ':>6} | 全程可见: "
          f"{'Prog':>6} {'Succ':>6}")
    for name, w, sa, sg, note in configs:
        p, ph, c, s = episode_metrics(w, sa, sg, occluded=True)
        pf, _, _, sf = episode_metrics(w, sa, sg, occluded=False)
        print(f"{name:<14} {w:>4.2f} {sa:>5.2f} {sg:>5.2f} |          "
              f"{p:>6.3f} {ph:>6.3f} {c:>6.3f} {s:>6.3f} |        "
              f"{pf:>6.3f} {sf:>6.3f}   ← {note}")
    print("\n论文实测对照 (Table 2/3, %): Veo 3 观测控制 17.4/82.6/66.5/8.7,")
    print("  全程可见 (与 Sora 2 Pro 平均) 84.6/—/—/46.2;")
    print("  WAN 2.2 7.7/52/58.4/0.9; 相机控制模型 Progress 全部 < 5%, VMem ≈ 0。")


def part_b():
    print("\n" + "=" * 72)
    print("Part B: 冻结定理——MSE 最优生成器的隐期有效演化率 = 数据先验 q")
    print("=" * 72)
    # 训练语料: M 条遮挡事件, q 比例的揭示帧显示推进 (其余静态/已完成 → 揭示=离开时状态)
    # MSE 最优预测: advance* = q·r·Δ  (条件均值)
    # 测试 (StEvo 协议, 测试时过程一定推进): progress = P(q·rΔ + N(0, σ_p·rΔ) ≥ 0.5·rΔ)
    M = 200_000
    r = rng.uniform(0.5, 1.5, M)
    print(f"\n{'q (数据先验)':>12} | {'σ_p=0.10':>9} | {'σ_p=0.20':>9}   预测噪声标定")
    for q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        row = []
        for sp in (0.10, 0.20):
            adv = q * r + rng.normal(0, sp, M)
            row.append((adv >= 0.5 * r).mean())
        print(f"{q:>12.1f} | {row[0]:>9.3f} | {row[1]:>9.3f}")
    print("\n读法: 及格线 (推进率 ≥ 50%) 对应 q ≥ 0.5 —— 训练数据中「遮挡后推进」")
    print("至少要和「遮挡后不变」一样多; 而静态场景渲染语料 (3DGS/Unreal) 的 q = 0。")
    print("实测映射 (σ_p=0.1): Veo 3 推进率 0.174 → 有效 q ≈ "
          f"{0.5 - 0.915 * 0.1:.2f}; Sora 2 Pro 0.131 → q ≈ "
          f"{0.5 - 1.125 * 0.1:.2f}; 相机控制模型 < 0.05 → q ≈ 0。")


if __name__ == "__main__":
    part_a()
    part_b()
