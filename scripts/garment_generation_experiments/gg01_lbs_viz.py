"""
GG 01 §2.5 配图实验：2D toy 线性混合蒙皮（LBS）
================================================

一根带厚度的 2D 手臂（长 2、半厚度 r=0.08，皮肤 = 上/下两条边线），
关节在原点。两根骨骼：
  骨骼 1（上臂）: A1 = I（恒等，不动）
  骨骼 2（前臂）: A2 = R(θ)（绕关节转 θ）
每个顶点的权重 w(x) 是"跟随前臂的比例"，沿手臂从 0（肩端）过渡到 1（腕端）。

对比两种权重分配（θ=90°）：
  硬分配: w = 1{x>0}               —— 关节缝处归属突变
  平滑:   w = smoothstep(x; 带 W)   —— 过渡带内连续爬升

验证/可视化三件事：
  图 1  硬分配 → 位移场在缝处跳变（高度 ≈ r√2），跨缝边拉长数倍（几何撕裂）；
       平滑权重 → 位移场连续爬坡，轮廓连续弯曲（同样的公式，只换 w(x)）；
  图 2  混合点落在两种摆放的【弦】上而非【弧】上：
       半径 ρ(w)=|1−w+w·e^{iθ}| < 1（糖纸塌缩的 2D 版；θ=180°、w=1/2 时 ρ=0）；
       有效转角 φ(w)=arg(1−w+w·e^{iθ}) 从 0 连续爬到 θ（w=1/2 时恰为 θ/2）。

只依赖 numpy + matplotlib。数字全部解析可验：
  r√2 = 0.1131（缝处位移跳变）、ρ(1/2)=cos(θ/2)、φ(1/2)=θ/2。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                    "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/fanghojui/JokerFangmax.github.io/content/post/garment_generation/GG_01_smpl_cloth_basics"

R_ARM = 0.08          # 手臂半厚度（皮肤到中线的距离）
THETA = np.deg2rad(90)
W_BAND = 0.30         # 平滑过渡带半宽


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def weights_hard(x):
    return (x > 0).astype(float)


def weights_smooth(x, band=W_BAND):
    t = np.clip((x + band) / (2 * band), 0, 1)
    return 3 * t**2 - 2 * t**3          # smoothstep


def lbs(v, w, theta):
    """v: (N,2) 顶点，w: (N,) 跟随旋转骨骼的比例。A1=I, A2=R(theta)。"""
    R = rot(theta)
    return (1 - w)[:, None] * v + w[:, None] * (v @ R.T)


def strip(n=81):
    """手臂轮廓顶点：上边线、下边线（x∈[−1,1]）。"""
    x = np.linspace(-1, 1, n)
    top = np.stack([x, np.full_like(x, R_ARM)], axis=1)
    bot = np.stack([x, np.full_like(x, -R_ARM)], axis=1)
    return x, top, bot


def outline(ax, pts_list, **kw):
    for pts in pts_list:
        ax.plot(pts[:, 0], pts[:, 1], **kw)


# ============ 图 1：撕裂 vs 连续弯 ============

def fig1():
    x, top, bot = strip()
    w_h, w_s = weights_hard(x), weights_smooth(x)
    th, ts = lbs(top, w_h, THETA), lbs(top, w_s, THETA)
    bh, bs = lbs(bot, w_h, THETA), lbs(bot, w_s, THETA)
    rest_kw = dict(color="0.65", lw=1.2, ls="--", alpha=0.9)

    # 跨缝位移跳变（上边线，权重发生 0→1 跳变的那对相邻顶点）
    i = int(np.flatnonzero(np.diff(w_h))[0]) + 1    # x[i-1] 与 x[i] 分属两根骨骼
    jump_h = np.linalg.norm(th[i] - th[i - 1])
    rest_len = x[i] - x[i - 1]
    stretch_h = jump_h / rest_len
    # 平滑版的局部最大拉伸（上/下边线取最大）
    d_ts = np.linalg.norm(np.diff(ts, axis=0), axis=1)
    d_bs = np.linalg.norm(np.diff(bs, axis=0), axis=1)
    stretch_s = max(d_ts.max(), d_bs.max()) / rest_len
    print(f"[图1] 上边线缝处相邻顶点：静止间距 {rest_len:.4f}")
    print(f"      硬分配：转后间距 {jump_h:.4f}（拉伸 {stretch_h:.1f}×；位移跳变 "
          f"{np.linalg.norm(th[i]-top[i]):.4f} → 网格加密时 → r√2 = {R_ARM*np.sqrt(2):.4f}）")
    print(f"      平滑权重：局部最大拉伸 {stretch_s:.2f}×")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6),
                             gridspec_kw={"height_ratios": [2.1, 1]})
    (ax1, ax2), (ax3, _) = axes
    for ax, deformed, w, title in [
            (ax1, (th, bh), w_h, "(a) 硬分配：$w=\\mathbf{1}\\{x>0\\}$"),
            (ax2, (ts, bs), w_s, "(b) 平滑权重：$w$ 在过渡带内连续爬升")]:
        outline(ax, [top, bot], **rest_kw)
        outline(ax, list(deformed), color="0.25", lw=1.6)
        # 顶点按 w 着色
        for edge, def_edge in [(top, deformed[0]), (bot, deformed[1])]:
            sc = ax.scatter(def_edge[:, 0], def_edge[:, 1], c=w, s=9,
                            cmap="coolwarm", vmin=0, vmax=1, zorder=3)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")

    # (a) 高亮缝处跳变的边（上边线）
    ax1.plot([th[i-1, 0], th[i, 0]], [th[i-1, 1], th[i, 1]],
             color="crimson", lw=3.2, zorder=4,
             solid_capstyle="round")
    ax1.annotate(f"缝处相邻两点\n间距 {rest_len:.3f} → {jump_h:.3f}\n（拉伸 {stretch_h:.1f}×）",
                 xy=th[i], xytext=(0.28, 0.46), fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
                 bbox=dict(fc="white", ec="crimson", alpha=0.9, boxstyle="round,pad=0.3"))
    ax1.text(-0.98, 0.13, "灰色虚线：rest pose", fontsize=9, color="0.45")
    cb = fig.colorbar(sc, ax=[ax1, ax2], shrink=0.75, pad=0.02, aspect=28)
    cb.set_label("权重 $w$（跟随前臂的比例）", fontsize=9)

    # (b) 标注连续弯曲
    ax2.annotate("过渡带：折角被摊成连续的弯\n（同一条 LBS 公式，只换了 $w(x)$）",
                 xy=(0.02, R_ARM), xytext=(-0.75, 0.40), fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color="0.25", lw=1.2),
                 bbox=dict(fc="white", ec="0.5", alpha=0.9, boxstyle="round,pad=0.3"))

    # (c) 位移场：上边线 |v'−v| 随 x —— 跳变 vs 连续爬坡
    disp_h = np.linalg.norm(th - top, axis=1)
    disp_s = np.linalg.norm(ts - top, axis=1)
    ax3.plot(x, disp_h, color="crimson", lw=2, label="硬分配：位移场在缝处跳变")
    ax3.plot(x, disp_s, color="teal", lw=2, label="平滑权重：位移场连续爬坡")
    ax3.axvline(0, color="0.6", lw=0.8, ls=":")
    ax3.annotate(f"跳变（网格加密时 → r√2 = {R_ARM*np.sqrt(2):.3f}）",
                 xy=(x[i], disp_h[i]), xytext=(0.22, disp_h.max()*0.86),
                 fontsize=9.5, arrowprops=dict(arrowstyle="->", color="crimson"))
    ax3.set_xlabel("沿手臂位置 $x$（关节在 0）")
    ax3.set_ylabel("上边线顶点位移 $|v'-v|$")
    ax3.set_title("(c) 位移场：跳变 vs 连续（撕裂与连续过渡的实义）", fontsize=11)
    ax3.legend(fontsize=9, loc="center left")

    fig.suptitle("图 1 · 2D toy 蒙皮（θ=90°，蓝→红 = 权重 0→1）："
                 "硬分配撕裂，平滑权重连续弯曲", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.91, 0.96))
    fig.savefig(f"{OUT}/gg01_lbs_fig1_tear_vs_smooth.png", dpi=170)
    plt.close(fig)


# ============ 图 2：弦 vs 弧（糖纸塌缩的 2D 版） ============

def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # 左：单位圆（弧）+ 两条弦（θ=90° 与 180°），弦上点按 w 着色
    t = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(t), np.sin(t), color="0.7", lw=1.4)
    ws = np.linspace(0, 1, 11)
    for theta, col, xy, xytext in [
            (np.pi, "crimson", 7, (-0.88, -0.44)),      # 180° 弦，指到 w=0.7 的点
            (np.pi / 2, "teal", 3, (0.30, -0.66))]:     # 90° 弦，指到 w=0.3 的点
        v0 = np.array([1.0, 0.0])                    # 顶点 rest 位置（离关节 1）
        v1 = rot(theta) @ v0                         # 旋转骨骼单独放置它
        pts = (1 - ws)[:, None] * v0 + ws[:, None] * v1
        ax1.plot(pts[:, 0], pts[:, 1], color=col, lw=2, alpha=0.85)
        ax1.scatter(pts[:, 0], pts[:, 1], c=ws, s=26, cmap="coolwarm",
                    vmin=0, vmax=1, zorder=3, edgecolors="white", lw=0.5)
        deg = round(np.rad2deg(theta))
        ax1.annotate(f"θ={deg}° 的弦\n（LBS 混合的轨迹）",
                     xy=pts[xy], xytext=xytext, fontsize=9, color=col,
                     arrowprops=dict(arrowstyle="->", color=col, lw=1))
    ax1.annotate("单位圆 = 弧\n（正确的旋转：模长守 1）", xy=(np.cos(2.2), np.sin(2.2)),
                 xytext=(-0.97, 0.72), fontsize=9, color="0.45",
                 arrowprops=dict(arrowstyle="->", color="0.6"))
    ax1.annotate("w=1/2：半径 0\n完全塌缩", xy=(0, 0), xytext=(-0.95, -0.72),
                 fontsize=9.5, color="crimson",
                 arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
    ax1.set_aspect("equal")
    ax1.set_title("(a) 混合点在弦上，不在弧上", fontsize=11)
    ax1.axis("off")
    ax1.set_xlim(-1.25, 1.25); ax1.set_ylim(-1.05, 1.05)

    # 右：有效半径 ρ(w) 与有效转角 φ(w)/θ
    w = np.linspace(0, 1, 200)
    for theta, col in [(np.pi, "crimson"), (np.deg2rad(135), "darkorange"),
                       (np.pi / 2, "teal"), (np.deg2rad(45), "steelblue")]:
        z = 1 - w + w * np.exp(1j * theta)
        deg = round(np.rad2deg(theta))
        ax2.plot(w, np.abs(z), color=col, lw=2,
                 label=f"ρ(w)，θ={deg}°")
        ax2.plot(w, np.rad2deg(np.angle(z)) / deg, color=col, lw=1.4,
                 ls="--", alpha=0.75)
    ax2.axhline(1, color="0.85", lw=0.8)
    ax2.annotate("ρ=cos(θ/2)=0.707\n（§2.6 手算表的数）", xy=(0.5, np.cos(np.pi/4)),
                 xytext=(0.60, 0.86), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="teal"))
    ax2.annotate("θ=180°：w=1/2 处 ρ=0\n（弯平的糖纸）", xy=(0.5, 0), xytext=(0.05, 0.18),
                 fontsize=9.5, color="crimson",
                 arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
    ax2.set_xlabel("权重 $w$（混合比例）")
    ax2.set_ylabel("有效半径 ρ（实线）      有效转角 φ/θ（虚线）")
    ax2.set_title("(b) 弦的代价：半径缩水；角度仍连续爬升", fontsize=11)
    ax2.legend(fontsize=8.5, loc="center left")
    ax2.grid(alpha=0.25)

    fig.suptitle("图 2 · 弦与弧：$v'=(1{-}w)\\,v + w\\,R_\\theta v$ "
                 "（顶点离关节 1、θ 取 45°–180°）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/gg01_lbs_fig2_chord_arc.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("[done] 图已写入", OUT)
