# VG 系列数值实验脚本

视频生成系列（`content/post/video_generation/`）写作过程中的 toy 实验，numpy/scipy、可独立运行。与 GG 系列不同，这里是**迭代过程文件**（toy → toy2…→ final），保留全部中间版本是为了可追溯——每个中间文件是正文某个论断的第一次验证，final* 是定稿数字的来源：

- `vg01_toy.py` … `vg01_unit.py`：VG 01（扩散基础）的渐进实验——toy DDPM、不同损失加权/刚度（stiff）设置、收敛性与单位检查；
- `vg01_final.py` / `vg01_final2.py`：VG 01 正文数字的最终来源；
- `vg02_sds.py` … `vg02_sds3.py`：VG 02（SDS / 3D 先验）的 score 蒸馏 toy 验证；
- `vg03_janus.py` + `vg03_output.txt`：VG 03（Unique3D）实验一——单视角边际 SDS (A) vs 联合先验蒸馏 (B) vs 联合采样重建 (C) 的 Janus 对照（B 路线 stall 在对称中心的均值回归、C 路线解析恢复率 Φ(1/(√2σ))³ 均在输出里）；
- `vg03_isomer.py` + `vg03_isomer_output.txt`：VG 03 实验二——ISOMER 两机制：2a 法线积分误差结构（随机游走解析锚 0.8419 vs 实测 0.8519、旋转平均 1/√R 收敛到 DFT-L2 投影地板）+ 2b ExplicitTarget cos² 加权 = 逆方差加权（0.0706 ≈ oracle 0.0705；错误视图正面 bin 吃满 δ）。脚本 docstring 记录了 v1/v2 两版被废弃的实现教训（数组旋转三次插值毁场、前向差分半像素错位、非周期接缝、|cos| 分箱稀释）——正文 §6 的自勘误来源。

（GG 系列的整理版脚本在 `../garment_generation_experiments/`，每篇一个干净的单文件。）
