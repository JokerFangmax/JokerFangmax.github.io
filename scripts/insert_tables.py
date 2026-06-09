#!/usr/bin/env python3
"""Insert data tables into index.md next to each figure."""

import re
from pathlib import Path

TABLES = {
    "DIRICHLET": """**Underlying data:**

| mesh | walks | Zombie RMSE | WoSt RMSE | ratio Z/W |
|---|---|---|---|---|
| Bunny | 16 | 0.0349 | 0.0304 | 1.1484 |
| Bunny | 64 | 0.0175 | 0.0157 | 1.1146 |
| Bunny | 256 | 0.0076 | 0.0078 | 0.9705 |
| Bunny | 1024 | 0.0041 | 0.0040 | 1.0321 |
| Spot | 16 | 0.1221 | 0.1158 | 1.0544 |
| Spot | 64 | 0.0672 | 0.0598 | 1.1237 |
| Spot | 256 | 0.0330 | 0.0308 | 1.0707 |
| Spot | 1024 | 0.0142 | 0.0152 | 0.9343 |

""",
    "EPSILON": """**Underlying data:**

| mesh | epsilon | Zombie RMSE | WoSt RMSE |
|---|---|---|---|
| Bunny | 0.01 | 0.0222 | 0.1529 |
| Bunny | 0.001 | 0.0151 | 0.0268 |
| Bunny | 0.0001 | 0.0154 | 0.0125 |
| Bunny | 1e-05 | 0.0176 | 0.0140 |
| Spot | 0.01 | 0.1664 | 0.6269 |
| Spot | 0.001 | 0.1796 | 0.2342 |
| Spot | 0.0001 | 0.1519 | 0.1868 |
| Spot | 1e-05 | 0.2018 | 0.1877 |

""",
    "TOP_CORR": """**Underlying data:**

| mesh | dataset | feature | target | Pearson r | n |
|---|---|---|---|---|---|
| Spot | neumann_pointcloud | nearest_distance_proxy_norm | mean_steps | -0.7667 | 94 |
| Spot | bias_eps_1e-3 | nearest_distance_proxy_norm | mean_steps_epsilon | -0.7404 | 3876 |
| Bunny | neumann_pointcloud | nearest_distance_proxy_norm | mean_steps | -0.7050 | 99 |
| Spot | neumann_pointcloud | nearest_distance_proxy_norm | std_error | -0.6628 | 94 |
| Bunny | bias_eps_1e-3 | nearest_distance_proxy_norm | mean_steps_epsilon | -0.6602 | 4018 |
| Bunny | neumann_pointcloud | nearest_distance_proxy_norm | sample_variance | -0.6397 | 99 |
| Spot | bias_eps_1e-2 | nearest_distance_proxy_norm | abs_error_epsilon | -0.6309 | 3876 |
| Spot | neumann_pointcloud | nearest_distance_proxy_norm | sample_variance | -0.6241 | 94 |
| Spot | bias_eps_1e-3 | nearest_distance_proxy_norm | abs_error_epsilon | -0.6099 | 3876 |
| Spot | bias_eps_1e-2 | nearest_distance_proxy_norm | bias_indicator | -0.5996 | 3876 |

""",
    "ADAPTIVE": """**Underlying data:**

| mesh | method | RMSE | MAE | mean variance | mean samples | runtime (s) |
|---|---|---|---|---|---|---|
| Bunny | fixed_256 | 0.0086 | 0.0062 | 0.0170 | 256.00 | 13.358 |
| Bunny | fixed_512 | 0.0055 | 0.0040 | 0.0170 | 512.00 | 27.057 |
| Bunny | fixed_1024 | 0.0040 | 0.0029 | 0.0170 | 1024.00 | 53.387 |
| Bunny | variance_adaptive_tau_0.003 | 0.0044 | 0.0033 | 0.0169 | 794.80 | 46.611 |
| Bunny | variance_adaptive_tau_0.005 | 0.0054 | 0.0042 | 0.0168 | 562.48 | 33.623 |
| Bunny | variance_adaptive_tau_0.008 | 0.0088 | 0.0065 | 0.0166 | 263.78 | 16.776 |
| Spot | fixed_256 | 0.0308 | 0.0236 | 0.2445 | 256.00 | 1.6529 |
| Spot | fixed_512 | 0.0223 | 0.0166 | 0.2453 | 512.00 | 3.2146 |
| Spot | fixed_1024 | 0.0147 | 0.0109 | 0.2464 | 1024.00 | 6.4561 |
| Spot | variance_adaptive_tau_0.003 | 0.0148 | 0.0110 | 0.2463 | 985.80 | 6.2535 |
| Spot | variance_adaptive_tau_0.005 | 0.0149 | 0.0113 | 0.2463 | 949.71 | 6.0453 |
| Spot | variance_adaptive_tau_0.008 | 0.0153 | 0.0117 | 0.2459 | 899.92 | 5.8975 |

""",
    "ANTITHETIC": """**Underlying data:**

| mesh | method | seed | RMSE | mean variance | elapsed (s) |
|---|---|---|---|---|---|
| Bunny | normal | 95210 | 0.0060 | 0.0177 | 26.326 |
| Bunny | antithetic | 95210 | 0.0045 | 0.0050 | 26.004 |
| Bunny | normal | 96219 | 0.0059 | 0.0172 | 25.934 |
| Bunny | antithetic | 96219 | 0.0043 | 0.0049 | 26.371 |
| Bunny | normal | 97228 | 0.0059 | 0.0180 | 25.269 |
| Bunny | antithetic | 97228 | 0.0043 | 0.0050 | 25.159 |
| Spot | normal | 117186 | 0.0225 | 0.2525 | 3.3485 |
| Spot | antithetic | 117186 | 0.0167 | 0.0697 | 3.3601 |
| Spot | normal | 118195 | 0.0222 | 0.2527 | 3.4299 |
| Spot | antithetic | 118195 | 0.0162 | 0.0688 | 3.2076 |
| Spot | normal | 119204 | 0.0225 | 0.2360 | 3.2682 |
| Spot | antithetic | 119204 | 0.0158 | 0.0661 | 3.3283 |

""",
    "LAZY": """**Underlying data:**

| mesh | method | seed | RMSE | elapsed (s) | refinement ratio |
|---|---|---|---|---|---|
| Bunny | full_exact | 59044 | 0.0066 | 238.26 | 1.0000 |
| Bunny | lazy_threshold_x1 | 59044 | 0.0066 | 27.020 | 0.0929 |
| Bunny | lazy_threshold_x4 | 59044 | 0.0066 | 65.914 | 0.2583 |
| Bunny | lazy_threshold_x16 | 59044 | 0.0066 | 114.34 | 0.4582 |
| Bunny | full_exact | 60053 | 0.0055 | 234.07 | 1.0000 |
| Bunny | lazy_threshold_x1 | 60053 | 0.0055 | 25.455 | 0.0938 |
| Bunny | lazy_threshold_x4 | 60053 | 0.0055 | 64.642 | 0.2609 |
| Bunny | lazy_threshold_x16 | 60053 | 0.0055 | 112.75 | 0.4630 |
| Bunny | full_exact | 61062 | 0.0058 | 240.06 | 1.0000 |
| Bunny | lazy_threshold_x1 | 61062 | 0.0058 | 25.783 | 0.0940 |
| Bunny | lazy_threshold_x4 | 61062 | 0.0058 | 64.268 | 0.2619 |
| Bunny | lazy_threshold_x16 | 61062 | 0.0058 | 111.86 | 0.4652 |
| Spot | full_exact | 81020 | 0.0211 | 25.214 | 1.0000 |
| Spot | lazy_threshold_x1 | 81020 | 0.0211 | 3.0744 | 0.0769 |
| Spot | lazy_threshold_x4 | 81020 | 0.0211 | 6.2891 | 0.2135 |
| Spot | lazy_threshold_x16 | 81020 | 0.0211 | 10.211 | 0.3792 |
| Spot | full_exact | 82029 | 0.0227 | 25.652 | 1.0000 |
| Spot | lazy_threshold_x1 | 82029 | 0.0227 | 3.1120 | 0.0771 |
| Spot | lazy_threshold_x4 | 82029 | 0.0227 | 6.4568 | 0.2140 |
| Spot | lazy_threshold_x16 | 82029 | 0.0227 | 10.497 | 0.3790 |
| Spot | full_exact | 83038 | 0.0221 | 25.981 | 1.0000 |
| Spot | lazy_threshold_x1 | 83038 | 0.0221 | 3.1728 | 0.0773 |
| Spot | lazy_threshold_x4 | 83038 | 0.0221 | 6.4254 | 0.2145 |
| Spot | lazy_threshold_x16 | 83038 | 0.0221 | 10.092 | 0.3803 |

""",
    "EPS_DIST": """**Underlying data:**

| mesh | bin | epsilon | n | mean RMSE | mean bias |
|---|---|---|---|---|---|
| Bunny | 1 | 1e-05 | 3 | 0.0453 | 0.0182 |
| Bunny | 1 | 0.0001 | 3 | 0.0452 | 0.0180 |
| Bunny | 1 | 0.001 | 3 | 0.1171 | 0.0378 |
| Bunny | 1 | 0.01 | 3 | 0.7724 | 0.3101 |
| Bunny | 2 | 1e-05 | 3 | 0.0223 | 0.0127 |
| Bunny | 2 | 0.0001 | 3 | 0.0209 | 0.0153 |
| Bunny | 2 | 0.001 | 3 | 0.0511 | 0.0201 |
| Bunny | 2 | 0.01 | 3 | 0.3358 | 0.1294 |
| Bunny | 3 | 1e-05 | 3 | 0.0142 | 0.0171 |
| Bunny | 3 | 0.0001 | 3 | 0.0143 | 0.0137 |
| Bunny | 3 | 0.001 | 3 | 0.0244 | 0.0179 |
| Bunny | 3 | 0.01 | 3 | 0.1211 | 0.0497 |
| Bunny | 4 | 1e-05 | 3 | 0.0129 | 0.0104 |
| Bunny | 4 | 0.0001 | 3 | 0.0102 | 0.0093 |
| Bunny | 4 | 0.001 | 3 | 0.0126 | 0.0131 |
| Bunny | 4 | 0.01 | 3 | 0.0386 | 0.0211 |
| Spot | 1 | 1e-05 | 3 | 0.1861 | 0.0667 |
| Spot | 1 | 0.0001 | 3 | 0.1633 | 0.0751 |
| Spot | 1 | 0.001 | 3 | 0.1946 | 0.0594 |
| Spot | 1 | 0.01 | 3 | 0.5579 | 0.1786 |
| Spot | 2 | 1e-05 | 3 | 0.0626 | 0.0392 |
| Spot | 2 | 0.0001 | 3 | 0.0683 | 0.0352 |
| Spot | 2 | 0.001 | 3 | 0.0719 | 0.0334 |
| Spot | 2 | 0.01 | 3 | 0.1627 | 0.0585 |
| Spot | 3 | 1e-05 | 3 | 0.0230 | 0.0196 |
| Spot | 3 | 0.0001 | 3 | 0.0264 | 0.0288 |
| Spot | 3 | 0.001 | 3 | 0.0277 | 0.0231 |
| Spot | 3 | 0.01 | 3 | 0.0497 | 0.0224 |

""",
}

path = Path(r"content\project\wost-final-project\index.md")
text = path.read_text(encoding="utf-8")

# Fig 1: Dirichlet - insert after "This validates the baseline pipeline before interpreting mixed Neumann sensitivity."
text = text.replace(
    "This validates the baseline pipeline before interpreting mixed Neumann sensitivity.\n\n---",
    "This validates the baseline pipeline before interpreting mixed Neumann sensitivity.\n\n" + TABLES["DIRICHLET"] + "---"
)

# Fig 3: Epsilon - insert after "Coarse epsilon produces the largest errors in the tested setup."
text = text.replace(
    "**Figure 3.** Mixed Neumann RMSE under different epsilon values at 256 walks. Coarse epsilon produces the largest errors in the tested setup.\n\n![Epsilon-vs-half-epsilon",
    "**Figure 3.** Mixed Neumann RMSE under different epsilon values at 256 walks. Coarse epsilon produces the largest errors in the tested setup.\n\n" + TABLES["EPSILON"] + "![Epsilon-vs-half-epsilon"
)

# Fig 5: Top correlations - insert after "local normal variation appears as a secondary descriptor."
text = text.replace(
    "local normal variation appears as a secondary descriptor.\n\n![Pointwise Neumann",
    "local normal variation appears as a secondary descriptor.\n\n" + TABLES["TOP_CORR"] + "![Pointwise Neumann"
)

# Fig 10a/10b: Adaptive - insert after "Spot remains close to the maximum sample count, indicating widespread high variance."
text = text.replace(
    "indicating widespread high variance.\n\n### 10.2 Live Path Trace",
    "indicating widespread high variance.\n\n" + TABLES["ADAPTIVE"] + "### 10.2 Live Path Trace"
)

# Fig 13: Antithetic - insert after "this addresses estimator variance, not epsilon or geometry-related systematic error."
text = text.replace(
    "this addresses estimator variance, not epsilon or geometry-related systematic error.\n\n### 10.4 Lazy Refinement",
    "this addresses estimator variance, not epsilon or geometry-related systematic error.\n\n" + TABLES["ANTITHETIC"] + "### 10.4 Lazy Refinement"
)

# Fig 14: Lazy - insert after "while preserving the tested mean RMSE."
text = text.replace(
    "while preserving the tested mean RMSE.\n\n### 10.5 Summary Table",
    "while preserving the tested mean RMSE.\n\n" + TABLES["LAZY"] + "### 10.5 Summary Table"
)

# Fig 9a/9b: Heatmap - insert after Appendix Figure B3 caption
text = text.replace(
    "showing stronger near-boundary sensitivity at coarse epsilon.\n\n![BVH versus",
    "showing stronger near-boundary sensitivity at coarse epsilon.\n\n" + TABLES["EPS_DIST"] + "![BVH versus"
)

path.write_text(text, encoding="utf-8")
print("Tables inserted successfully.")
