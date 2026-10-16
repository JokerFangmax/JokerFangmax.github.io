"""WM_12 toy: 机制一致性为什么需要显式状态——两个失真机制

游戏状态系统 (街霸式): 技能条 m ∈ [0,1], 血量 HP。规则 (机制):
  超必杀: 仅当 m ≥ τ 时可用, 用后 m 清零, 对手 HP 大幅扣减;
  回合结束: 任一方 HP ≤ 0 → 结算画面。

机制决策 = 状态的阈值函数。比较两种状态来源:
  Pixel 分支  从渲染的 HUD 读状态: 技能条画成 N_PIX 个像素格 (向下取整
             —— 条只画"已充满"的格), 加渲染噪声 σ_r;
  State 分支  直接回归状态 (StatePlay 的显式状态分支), 标注噪声 σ_s ≪ 格宽。

Part A  阈值密度放大器: 决策翻转率 ~ f(τ)·σ_eff。
  玩家会把气攒在阈值附近 (savers), 阈值附近密度 f(τ) 高 → 估计误差在
  最关键的地方被放大成机制错误。且格点量化有方向性: τ 落在两个格点
  之间时, 「m ≥ τ」被读成「m ≥ 下一个格点」→ 系统性漏触发 (macro fail,
  即 ReactiveGWM "气满了也放不出超必杀")。

Part B  事件期视觉屏蔽 + 自由生成: 超必杀触发的瞬间画面闪光, HUD 恰在
  状态跳变最大的时刻不可读。自由 rollout 里像素分支的内部状态只能按
  平均速率漂移 (它没有事件转移模型), 错过清零与扣血; 状态分支按学到的
  转移规则推进。后果: 像素分支的内部技能条停在满槽 → 违规再放超必杀;
  内部 HP 漏掉 -30 → 真实世界已 KO, 它还在生成对打 ("零血攻击")。
"""

import numpy as np

rng = np.random.default_rng(12)

# ---- 参数 ----
TAU = 0.95        # 超必杀阈值
N_PIX = 32        # HUD 技能条像素格数 (32 格)
Q = 1.0 / N_PIX   # 量化格宽
SIGMA_R = 0.005   # 渲染噪声 (像素分支)
SIGMA_S = 0.003   # 标注/回归噪声 (状态分支)
N_A = 200_000     # Part A 样本数

K_FLASH = 15      # 超必杀动画的闪光帧数 (HUD 不可读)
GAIN = 0.08       # 每次命中技能条增量
AVG_RATE = GAIN * 0.1   # 像素分支学到的平均漂移 (命中率 0.1/帧)
DAMAGE_ART = 30.0 # 超必杀伤害
DAMAGE_HIT = 6.0  # 普通命中伤害
HP0 = 40.0
N_B = 4000        # Part B episode 数
T_POST = 40       # 事件后跟踪帧数


def pixel_read(m):
    """HUD 渲染: 只画已充满的像素格 (向下取整) + 渲染噪声。"""
    return np.floor((np.clip(m, 0, 1) + rng.normal(0, SIGMA_R, np.shape(m))) * N_PIX) / N_PIX


def state_read(m):
    return m + rng.normal(0, SIGMA_S, np.shape(m))


def part_a():
    print("=" * 70)
    print("Part A: 阈值密度放大器 —— 决策翻转率 vs 阈值附近密度")
    print("=" * 70)
    mask_saver = rng.random(N_A) < 0.3
    m_saver = np.where(mask_saver,
                       np.clip(rng.normal(TAU, 0.03, N_A), 0, 1),
                       rng.uniform(0, 1, N_A))
    for label, m in [
        ("savers (玩家攒气, 30% 集中在 τ 附近)", m_saver),
        ("uniform (无攒气行为)", rng.uniform(0, 1, N_A)),
    ]:
        dens = ((m >= TAU - 0.01) & (m <= TAU + 0.01)).mean() / 0.02  # f̂(τ)
        fire_true = m >= TAU
        for name, m_hat in [("Pixel", pixel_read(m)), ("State", state_read(m))]:
            fire_hat = m_hat >= TAU
            miss = (fire_true & ~fire_hat).mean()      # 该放没放 (macro fail)
            false_fire = (~fire_true & fire_hat).mean()  # 不该放放了
            print(f"  [{label[:8]:>8}] f(τ)≈{dens:5.2f} | {name:>5}: "
                  f"漏触发 {miss*100:5.2f}%  违规触发 {false_fire*100:5.2f}%  "
                  f"合计 {miss + false_fire:.4f}")
        print(f"            ({label})")


def part_b():
    print()
    print("=" * 70)
    print("Part B: 事件期视觉屏蔽 + 自由生成 (超必杀 @ t*=0, 闪光 %d 帧)" % K_FLASH)
    print("=" * 70)

    viol_pix, viol_state = [], []
    m_err_pix, m_err_state, hp_err_pix, hp_err_state = [], [], [], []
    for _ in range(N_B):
        # --- 真实世界: 攒气到 τ 触发超必杀 ---
        m_pre = TAU + rng.normal(0, 0.01)          # 触发瞬间真实技能条
        hp_pre = HP0 - DAMAGE_HIT * rng.integers(0, 3)  # 触发前对手 HP
        # 事件: m 清零, 对手 -30
        m_true_after = 0.0
        hp_true_after = hp_pre - DAMAGE_ART

        # --- Pixel 分支: 闪光期 HUD 不可读, 内部状态冻结在事件前读数 ---
        m_hat_pre = pixel_read(np.array(m_pre))     # 事件前最后一读 (系统性偏低)
        m_hat_pix = m_hat_pre                        # 闪光期冻结
        hp_hat_pix = hp_pre                          # 漏掉 -30

        # --- State 分支: 按学到的转移规则推进 (回归噪声级) ---
        m_hat_state = state_read(np.array(0.0))
        hp_hat_state = hp_pre - DAMAGE_ART + rng.normal(0, SIGMA_S * 10)

        # 事件后 T_POST 帧自由生成: 真实世界继续命中 (平均速率), 像素分支
        # 的内部状态按平均速率漂移 (从未观察到清零), 状态分支同真实规则。
        viol_p = viol_s = 0.0
        for k in range(T_POST):
            drift = AVG_RATE + rng.normal(0, 0.002)
            m_true_after = min(1.0, m_true_after + drift)
            m_hat_pix = min(1.0, m_hat_pix + drift)     # 冻结值 + 漂移, 永远偏高
            m_hat_state = min(1.0, m_hat_state + drift + rng.normal(0, SIGMA_S))
            # 违规判定: 内部估计 ≥ τ 时模型会生成超必杀 (真实世界没有)
            if k >= K_FLASH:
                viol_p = max(viol_p, float(m_hat_pix >= TAU))
                viol_s = max(viol_s, float(m_hat_state >= TAU))

        viol_pix.append(viol_p)
        viol_state.append(viol_s)
        m_err_pix.append(m_hat_pix - m_true_after)
        m_err_state.append(m_hat_state - m_true_after)
        hp_err_pix.append(hp_hat_pix - hp_true_after)
        hp_err_state.append(hp_hat_state - hp_true_after)

    viol_pix, viol_state = np.array(viol_pix), np.array(viol_state)
    print(f"  事件后 {T_POST} 帧内违规再放超必杀的概率 (内部 m̂ ≥ τ):")
    print(f"    Pixel 分支: {viol_pix.mean()*100:5.1f}%   "
          f"State 分支: {viol_state.mean()*100:5.1f}%")
    print(f"  事件后 {T_POST} 帧的内部状态误差 (均值 ± 标准差):")
    print(f"    技能条  Pixel: {np.mean(m_err_pix):+.3f} ± {np.std(m_err_pix):.3f}   "
          f"State: {np.mean(m_err_state):+.3f} ± {np.std(m_err_state):.3f}")
    print(f"    对手 HP Pixel: {np.mean(hp_err_pix):+.3f} ± {np.std(hp_err_pix):.3f}   "
          f"State: {np.mean(hp_err_state):+.3f} ± {np.std(hp_err_state):.3f}")
    print(f"  → Pixel 的 HP 误差 ≈ 漏掉的 {DAMAGE_ART:.0f} 点超必杀伤害:")
    print(f"    真实 HP = {HP0 - DAMAGE_ART - 6:.0f}~{HP0 - DAMAGE_ART:.0f} 区间再挨 1-2 记普攻即 KO,")
    print(f"    而 Pixel 分支内部 HP 仍 ~{HP0:.0f} → 继续生成对打 (零血攻击)。")


if __name__ == "__main__":
    part_a()
    part_b()
