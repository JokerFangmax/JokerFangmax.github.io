"""WM_14 toy: 程序内核 vs 学习引擎 vs 隐式状态——规则执行的精确性从哪来

对应 PWM (arXiv:2609.10540) 的核心对照: 状态演化机制换成程序执行后,
「规则内」的误差是否归零? 剩下的误差长什么样(系统性 vs 弥散性)?

世界: E=12 个实体 (hp=150, ammo=20), T=200 步交互, 每步 w.p. 0.8 一个事件:
  hit(50%):    存活攻击者→存活目标, hp -= dmg (dmg ∈ {15,20,25}), 攻击者弹药 -1
  heal(20%):   目标 hp += 20 (上限 100)
  poison(20%): 目标之后 5 步每步 hp -= 1 (各方案统一简化为事件时刻一次性 -5)
  shield(10%): 目标下一次受击伤害减半 (一次性, 各方案跟踪 bool)
规则: hp ≤ 0 → status = down (永久, 不再被选中)。真值世界确定性执行全部。
ammo 是非视觉变量(HUD-free 训练数据的镜像): 画面里根本没有它。

四种状态维护方案(同一事件流重放, 差别只在「怎么更新状态」):
  ImplicitState (prompt 切换基线): 无持久状态。死亡可见性: 每步真值里刚转
    down 的实体 w.p. P_VIS=0.6 被「画到尸体」→ 记为 down; 没看见 → 永远报
    存活。hp/弹药画面不可见 → 恒报初值。
  LearnedEngine (MASS 式学习转移): 自持 hp/ammo/shield 估计; 每个事件
    w.p. 1-EPS 精确应用, 否则等概率三选一: 伤害×U(0.5,1.5) / 打错目标 / 漏掉。
    status 由自身 hp 推导(粘滞 down)。
  ProgramEngine (PWM 式程序内核): 覆盖的事件类型确定性精确执行; 未覆盖类型
    完全不生效(冻结)。主对照覆盖 {hit,heal,ammo} (事件覆盖率 c=0.7),
    扫描 c ∈ {0.7, 0.9, 1.0}。
  ProgramBuggy (agent 写错一条规则): 同 c=1.0 但 hit 伤害×1.5
    (把 damage 20 抄成 30 的系统性错误)。

指标 (t = 50/100/200):
  status acc  全实体 alive/down 判对比例
  hp MAE      真值存活实体上 |est_hp - true_hp|
  ammo MAE    |est_ammo - true_ammo| 均值
  重放标准差  同一事件流重放 5 次, 终局 hp 估计的逐实体标准差均值
              (可验证性的操作化: 确定性 → 可复现 → 可单元测试定位)
"""

import numpy as np

rng = np.random.default_rng(14)

E = 12            # 实体数
T = 200           # 交互步数
P_EVENT = 0.8     # 每步发生事件概率
HP0, AMMO0 = 150, 20
P_VIS = 0.6       # ImplicitState 的死亡可见概率
EPS = 0.15        # LearnedEngine 的单事件误应用概率
DMGS = (15, 20, 25)
REPORT_TICKS = (50, 100, 200)
REPLAYS = 5

# 事件类型: 0=hit 1=heal 2=poison 3=shield; 覆盖集合按事件实例占比即 c
TYPE_P = (0.5, 0.2, 0.2, 0.1)
COVERAGE_SETS = {0.7: {0, 1}, 0.9: {0, 1, 2}, 1.0: {0, 1, 2, 3}}


def gen_event(hp):
    """抽一个事件 (type, src, dst, dmg); 攻击者/目标从存活者中抽。"""
    alive = np.where(hp > 0)[0]
    if len(alive) < 2:
        return None
    tp = rng.choice(4, p=TYPE_P)
    src, dst = rng.choice(alive, 2, replace=False)
    return (int(tp), int(src), int(dst), DMGS[rng.integers(3)])


def true_world():
    """确定性执行全部规则, 返回逐 tick 的 (hp, ammo, event)。"""
    hp = np.full(E, HP0, float)
    ammo = np.full(E, AMMO0, int)
    poison = np.zeros(E, int)
    shield = np.zeros(E, bool)
    traj = []
    for _ in range(T):
        hp = hp - (poison > 0)          # 持续中毒
        poison = np.maximum(poison - 1, 0)
        ev = gen_event(hp) if rng.random() < P_EVENT else None
        if ev is not None:
            tp, src, dst, dmg = ev
            if tp == 0:
                d = dmg // 2 if shield[dst] else dmg
                shield[dst] = False
                hp[dst] -= d
                ammo[src] -= 1
            elif tp == 1:
                hp[dst] = min(hp[dst] + 20, HP0)
            elif tp == 2:
                poison[dst] += 5
            else:
                shield[dst] = True
        traj.append((hp.copy(), ammo.copy(), ev))
    return traj


def replay(traj, scheme, coverage=None):
    """按方案重放同一事件流。返回 {t: metrics} 与终局估计(供重放对比)。"""
    est_hp = np.full(E, HP0, float)
    est_ammo = np.full(E, AMMO0, int)
    est_shield = np.zeros(E, bool)
    est_down = np.zeros(E, bool)        # learned: 粘滞; implicit: 见过尸体
    covered = COVERAGE_SETS[coverage] if coverage is not None else None
    out = {}
    prev_true_hp = np.full(E, HP0, float)

    for t, (true_hp, true_ammo, ev) in enumerate(traj):
        if scheme == "implicit":
            # 死亡可见性: 本步真值里刚转 down 的实体, w.p. P_VIS 画到尸体
            just_died = (prev_true_hp > 0) & (true_hp <= 0)
            for i in np.where(just_died)[0]:
                if rng.random() < P_VIS:
                    est_down[i] = True
        elif ev is not None:
            tp, src, dst, dmg = ev
            if scheme == "learned":
                if rng.random() > EPS:                      # 正确应用
                    if tp == 0:
                        d = dmg // 2 if est_shield[dst] else dmg
                        est_shield[dst] = False
                        est_hp[dst] -= d
                        est_ammo[src] -= 1
                    elif tp == 1:
                        est_hp[dst] = min(est_hp[dst] + 20, HP0)
                    elif tp == 2:
                        est_hp[dst] -= 5
                    else:
                        est_shield[dst] = True
                else:                                       # 三选一错误
                    err = rng.integers(3)
                    if err == 0 and tp == 0:                # 伤害幅度错
                        est_hp[dst] -= dmg * rng.uniform(0.5, 1.5)
                        est_ammo[src] -= 1
                    elif err == 1 and tp == 0:              # 打错目标
                        cand = np.where(est_hp > 0)[0]
                        if len(cand):
                            est_hp[rng.choice(cand)] -= dmg
                        est_ammo[src] -= 1
                    # err == 2: 整个事件漏掉
            elif scheme in ("program", "buggy"):
                if tp in covered:
                    if tp == 0:
                        d = dmg * 1.5 if scheme == "buggy" else dmg
                        d = d // 2 if est_shield[dst] else d
                        est_shield[dst] = False
                        est_hp[dst] -= d
                        est_ammo[src] -= 1
                    elif tp == 1:
                        est_hp[dst] = min(est_hp[dst] + 20, HP0)
                    elif tp == 2:
                        est_hp[dst] -= 5
                    else:
                        est_shield[dst] = True
        prev_true_hp = true_hp.copy()

        if scheme == "learned":
            est_down |= est_hp <= 0                         # 粘滞 down

        if (t + 1) in REPORT_TICKS:
            true_status = true_hp > 0
            if scheme == "implicit":
                s_est, h_est, a_est = ~est_down, np.full(E, HP0, float), np.full(E, AMMO0, int)
            elif scheme == "learned":
                s_est, h_est, a_est = ~est_down, est_hp.copy(), est_ammo.copy()
            else:
                s_est, h_est, a_est = est_hp > 0, est_hp.copy(), est_ammo.copy()
            out[t + 1] = dict(
                status=(s_est == true_status).mean(),
                hp_mae=np.abs(h_est[true_status] - true_hp[true_status]).mean()
                       if true_status.any() else 0.0,
                ammo=np.abs(a_est - true_ammo).mean(),
            )
    return out, est_hp


def main():
    schemes = [
        ("ImplicitState", "implicit", None),
        ("LearnedEngine", "learned", None),
        ("Program c=0.7", "program", 0.7),
        ("Program c=0.9", "program", 0.9),
        ("Program c=1.0", "program", 1.0),
        ("ProgramBuggy", "buggy", 1.0),
    ]
    EPISODES = 30
    acc = {name: {t: dict(status=0.0, hp_mae=0.0, ammo=0.0) for t in REPORT_TICKS}
           for name, _, _ in schemes}
    rstd = {name: 0.0 for name, _, _ in schemes}
    n_ev_tot = n_dead_tot = 0

    for _ in range(EPISODES):
        traj = true_world()
        n_ev_tot += sum(1 for _, _, ev in traj if ev is not None)
        n_dead_tot += (traj[-1][0] <= 0).sum()
        for name, scheme, cov in schemes:
            out, _ = replay(traj, scheme, cov)
            for t in REPORT_TICKS:
                for k in ("status", "hp_mae", "ammo"):
                    acc[name][t][k] += out[t][k]
            if scheme == "implicit":
                continue
            finals = np.array([replay(traj, scheme, cov)[1] for _ in range(REPLAYS)])
            rstd[name] += finals.std(axis=0).mean()

    print(f"参数: E={E}, T={T}, P_EVENT={P_EVENT}, P_VIS={P_VIS}, EPS={EPS}, "
          f"episodes={EPISODES}")
    print(f"真值: 平均 {n_ev_tot / EPISODES:.0f} 个事件/episode, "
          f"终局死亡 {n_dead_tot / EPISODES:.1f}/{E}\n")

    print(f"{'方案':>16} | {'t':>4} | {'status acc':>10} | {'hp MAE':>8} | {'ammo MAE':>9}")
    for name, _, _ in schemes:
        for t in REPORT_TICKS:
            m = acc[name][t]
            print(f"{name:>16} | {t:>4} | {m['status'] / EPISODES:10.3f} | "
                  f"{m['hp_mae'] / EPISODES:8.2f} | {m['ammo'] / EPISODES:9.2f}")

    print(f"\n===== 重放 {REPLAYS} 次的终局 hp 估计标准差(逐实体均值, episode 平均) =====")
    for name, _, _ in schemes:
        if name == "ImplicitState":
            print(f"{name:>16}: 0.000 (无状态, hp 恒报 {HP0})")
        else:
            print(f"{name:>16}: {rstd[name] / EPISODES:.3f}")


if __name__ == "__main__":
    main()
