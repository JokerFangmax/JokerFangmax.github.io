"""WM_09 toy: disappear-and-reappear 范式下的一致性指标虚高机制

模拟 MemoBench 的 V-D-R 三段结构, 用抽象状态量复现论文的两个核心评测现象:

现象 1 (论文 Table 2, LTX-Video): 相机不动的模型「搭便车」——
  目标从未离开视野, 一致性类指标 (ORS/ObjConsist/3DConsist) 被平凡最大化,
  尽管它完全没有执行相机指令 (Instruction Following 低)。

现象 2 (LiveWorld 论文附录 D.1 对 MIND 基准的批评, 与 MemoBench 的
  per-phase 分析同源): 以「重访时与初始快照的一致性」为记忆指标,
  会系统性惩罚真正推进了状态的模型——正确的演化反而被记为「不一致」。

世界: 单个目标对象, 状态 s_t (如水位) 以速率 r 每步累积演化;
  相机轨迹为 V-D-R: 前 V 步目标可见, 中间 D 步相机转开 (目标不可见),
  最后 R 步相机转回。
三个"模型":
  StaticCam    相机不动(不执行指令)——无运动线索时生成先验把内容渲染成
               近似静止 (对应论文 Fig.6: LTX-Video 全程几乎不动),
               R 段渲染 ≈ 初始快照 s0;
  ActiveFrozen 相机执行轨迹, 但记忆是冻结的——R 段渲染 = 离开时刻的状态;
  ActiveOracle 相机执行轨迹, 记忆是演化式的——R 段渲染 = 真实当前状态。
指标 (状态量上的抽象代理):
  InstFollow   是否执行了相机轨迹 (1/0)
  ObjPresent   R 段目标是否可被检出 (1/0, 对应 ORS 的检出率部分)
  SnapConsist  R 段渲染状态与 *初始快照* 的距离 (小 = "像过去",
               这是 naive 一致性指标奖励的方向)
  StateTruth   R 段渲染状态与 *真实当前状态* 的距离 (小 = 真正记得对)
  R-phase drop V 段 vs R 段的保真差 (对应论文 Table S6 的 per-phase ∆)
"""

import numpy as np

rng = np.random.default_rng(9)

# ---- 参数 ----
V, D, R = 20, 60, 20          # V-D-R 三段帧数 (合成子集平均 gap≈113, 真实≈60-70)
EPISODES = 4000
SIGMA_OBS = 0.1               # 观测/渲染噪声
R_LO, R_HI = 0.02, 0.10       # 状态演化速率 (每帧)


def run_episode():
    r = rng.uniform(R_LO, R_HI)                 # 本 episode 的演化速率
    traj = np.arange(V + D + R, dtype=float)
    true_state = r * traj                       # 真实状态: 从 0 累积
    obs_noise = rng.normal(0, SIGMA_OBS, V + D + R)

    last_seen_state = true_state[V - 1] + obs_noise[V - 1]   # 冻结记忆: 离开时刻

    # 各模型的 R 段渲染状态
    render_static = s0_like = true_state[0] + rng.normal(0, SIGMA_OBS, R)  # 静止渲染≈初始快照
    render_frozen = last_seen_state + rng.normal(0, SIGMA_OBS, R)
    render_oracle = true_state[V + D:] + rng.normal(0, SIGMA_OBS, R)

    return r, true_state, render_static, render_frozen, render_oracle


def main():
    names = ["StaticCam", "ActiveFrozen", "ActiveOracle"]
    inst_follow = {"StaticCam": 0.0, "ActiveFrozen": 1.0, "ActiveOracle": 1.0}
    acc = {k: {"snap": 0.0, "truth": 0.0, "present": 0.0,
               "v_fidel": 0.0, "r_fidel": 0.0} for k in names}

    for _ in range(EPISODES):
        r, true_state, rs, rf, ro = run_episode()
        s0 = true_state[0]                      # 初始快照 (V 段第一帧真值)
        sV_end = true_state[V - 1]              # 离开时刻真值
        renders = {"StaticCam": rs, "ActiveFrozen": rf, "ActiveOracle": ro}

        for k, rend in renders.items():
            a = acc[k]
            # SnapConsist: |渲染 - 初始快照| —— naive 一致性指标 (小=好)
            a["snap"] += np.mean(np.abs(rend - s0))
            # StateTruth: |渲染 - 真实当前状态| —— 真正的记忆正确性 (小=好)
            a["truth"] += np.mean(np.abs(rend - true_state[V + D:]))
            # ObjPresent: 目标状态是否仍"可检出"(状态幅度不因消失而归零) ——
            # StaticCam 全程可见必检出; 两个 Active 模型都渲染出了目标, 也检出
            a["present"] += 1.0
            # per-phase 保真 (对 GT 的渲染误差)
            # V 段: StaticCam 渲染静止快照 (误差≈r·V), 两个 Active 模型直接观测 (≈σ)
            v_err = abs(s0 - sV_end) if k == "StaticCam" else abs(
                sV_end - (sV_end + rng.normal(0, SIGMA_OBS)))
            a["v_fidel"] += v_err
            a["r_fidel"] += np.mean(np.abs(rend - true_state[V + D:]))

    print(f"episodes={EPISODES}, V/D/R = {V}/{D}/{R}, "
          f"r ~ U({R_LO},{R_HI}), 离开期间期望演化量 E[r*D] = "
          f"{(R_LO+R_HI)/2*D:.2f}\n")

    hdr = (f"{'model':>14} | {'InstFollow':>10} | {'ObjPresent':>10} | "
           f"{'SnapConsist↓':>12} | {'StateTruth↓':>11} | {'V-fid↓':>7} | {'R-fid↓':>7}")
    print(hdr)
    print("-" * len(hdr))
    for k in names:
        a = {m: v / EPISODES for m, v in acc[k].items()}
        print(f"{k:>14} | {inst_follow[k]:>10.2f} | {a['present']:>10.2f} | "
              f"{a['snap']:>12.3f} | {a['truth']:>11.3f} | "
              f"{a['v_fidel']:>7.3f} | {a['r_fidel']:>7.3f}")

    print("\n理论值核对:")
    print(f"  E[r*(D+R)] (ActiveFrozen 的 StateTruth / ActiveOracle 的 SnapConsist) "
          f"= {(R_LO+R_HI)/2*(D+R):.3f}")
    print(f"  离开期间演化量 E[r*D] (R 段相对离开时刻的状态差) = {(R_LO+R_HI)/2*D:.3f}")
    print("\n解读:")
    print("  1. 搭便车: StaticCam 不执行相机指令 (InstFollow=0), 但 ObjPresent 满分、")
    print("     SnapConsist 全场最优 —— 目标从未离开视野且渲染静止, 一致性被平凡最大化;")
    print("  2. 冻结: ActiveFrozen 的 SnapConsist 也小 (像离开时刻), 但 StateTruth=演化量;")
    print("  3. 排序倒挂: naive 快照一致性的排序 StaticCam > Frozen > Oracle,")
    print("     与 StateTruth 的正确排序 Oracle >> Frozen ≈ StaticCam 完全相反 ——")
    print("     以「重访像过去」为标准的记忆评测会系统性地惩罚正确的状态演化。")


if __name__ == "__main__":
    main()
