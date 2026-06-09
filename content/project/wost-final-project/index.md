---
title: "Walk-on-Stars under Mixed Neumann Boundary Conditions"
summary: "Course final project: reproducing and extending Walk-on-Stars (WoSt) for PDE boundary value problems, with emphasis on geometry-sensitive diagnostics under mixed Neumann conditions on Bunny and Spot meshes."
authors: []
tags: ["Monte Carlo", "PDE", "Computer Graphics", "C++"]
categories: []
date: 2025-12-15

external_link: ""

image:
  caption: ""
  focal_point: ""
  preview_only: false

links:
- name: "3D Boundary Bias"
  url: "/files/wost-final-project/boundary_bias_detector_3d.html"
  icon_pack: fas
  icon: cube
- name: "3D Live Trace"
  url: "/files/wost-final-project/live_trace_interactive_3d.html"
  icon_pack: fas
  icon: route
- name: "Command Log"
  url: "command_log.txt"
  icon_pack: fas
  icon: terminal

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""
slides: ""
---

## Abstract

This project studies how a reproduced and extended **Walk-on-Stars (WoSt)** implementation behaves on Bunny and Spot meshes, with emphasis on **mixed Neumann boundary conditions**. Dirichlet experiments serve as sanity checks showing ordinary Monte Carlo convergence. Mixed Neumann experiments reveal strong geometry sensitivity: error, variance, path length, and boundary-bias indicators depend strongly on query placement near the inner boundary. Geometry-sensitive diagnostics identify the **normalized nearest-surface-distance proxy** as the strongest pointwise predictor. A distance-controlled experiment reduces the query-distance confounder and shows Spot remains higher-error than Bunny in matched bins 1–3, but the gap shrinks with distance.

---

## 1. Introduction and Research Question

**Research question:** How does Walk-on-Stars behave under mixed Neumann boundary conditions, and how are its errors affected by boundary proximity, epsilon, and mesh geometry?

The main scientific issue is that Neumann boundary handling depends on reflection and surface normals rather than simple boundary termination. This makes the solver more sensitive to local geometry and to how close query points are to the boundary. The goal is not to declare one solver globally better, but to identify where and why the mixed Neumann setting becomes difficult.

---

## 2. Background

A **boundary value problem** asks for a function inside a domain subject to conditions on the boundary:
- **Dirichlet conditions** prescribe the function value on the boundary; random-walk solvers can terminate at the boundary and evaluate that value.
- **Neumann conditions** prescribe normal derivative information.
- **Mixed conditions** use Dirichlet on part of the boundary and Neumann reflection/derivative contributions on the rest.

**Walk-on-Stars** is a random-walk Monte Carlo method for solving PDE-style boundary value problems. Instead of stepping on a regular grid, it samples larger geometry-aware jumps. In a Dirichlet problem, each path is mostly a termination-and-estimation process. In a mixed Neumann problem, paths may interact repeatedly with the inner boundary, and errors can arise from reflection behavior, normal estimation, epsilon termination, and local mesh quality.

---

## 3. Problem Formulation and Physical Interpretation

The experiments study a boundary value problem on a three-dimensional domain $\Omega$, represented as an outer axis-aligned cube with an inner triangle mesh removed. The manufactured reference solution used by the benchmark code is:

$$
u(x, y, z) = x + y + z, \quad \Delta u = 0
$$

For the Dirichlet validation, both the outer cube and inner mesh prescribe the value of $u$ on the boundary. In the mixed Neumann benchmark, the outer cube remains Dirichlet, while the inner mesh prescribes normal derivative data ($\nabla u \cdot \mathbf{n}$).

This distinction matters physically and numerically. A Dirichlet random walk can terminate when it reaches the boundary. A mixed Neumann walk may instead reflect or otherwise interact with the boundary normal, so the estimate depends on surface orientation, reflection behavior, and boundary proximity. The **epsilon parameter** should be interpreted as a numerical boundary thickness or termination tolerance. A coarse epsilon can stop or reflect paths before local boundary behavior is resolved, especially near the inner Neumann surface.

---

## 4. Methods and Experimental Setup

| Component | Description |
|-----------|-------------|
| **WoSt implementation** | C++ implementation in this repository, run through `build/Release/wost.exe` |
| **Zombie baseline** | Python-driven Zombie baseline for cross-method comparison |
| **Meshes** | Bunny (`obj/Bunny.obj`, 70,580 triangles) and Spot (`spot/spot_triangulated.obj`, 5,856 triangles) |
| **Query sampling** | Random/grid-based depending on benchmark; controlled experiments sample by normalized nearest-surface-distance proxy bins |
| **Walk counts** | 16, 64, 256, and 1024 walks per point |
| **Epsilon** | $10^{-2}$, $10^{-3}$, $10^{-4}$, $10^{-5}$ |
| **Metrics** | RMSE, mean steps, sample variance, mean samples, and epsilon-vs-half-epsilon boundary-bias indicator |
| **Nearest-distance proxy** | Normalized nearest-surface-distance proxy based on nearest triangle/centroid-style geometry features (not exact signed distance) |

---

## 5. Experiment 1: Dirichlet Sanity Check

![Dirichlet RMSE versus walks for Bunny and Spot, comparing WoSt with Zombie baseline](fig1_dirichlet_rmse_vs_walks.png)

**Figure 1.** Dirichlet RMSE versus walks. The intended reading is the convergence trend, not a claim that one method is universally better.

**Main claim:** Both Bunny and Spot show ordinary Monte Carlo convergence — increasing walks reduces RMSE — and WoSt/Zombie agreement is close across tested walk counts. This validates the baseline pipeline before interpreting mixed Neumann sensitivity.

**Underlying data:**

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

**Individual mesh views (Dirichlet):**

![Bunny and Spot Dirichlet RMSE vs walks — side-by-side panels](fig1_merged_dirichlet_panels.png)

*Fig 1a/1b merged.* Bunny (left) and Spot (right) Dirichlet RMSE vs walks — detailed single-mesh views in panel form.

---

## 6. Experiment 2: Mixed Neumann Sensitivity

![Mixed Neumann RMSE versus walks for Bunny and Spot](fig2_mixed_neumann_rmse_vs_walks.png)

**Figure 2.** Mixed Neumann RMSE versus walks. Compared with the Dirichlet panel, this figure highlights less uniform convergence and the harder Spot case.

**Main claim:** Mixed Neumann behavior is less uniform than Dirichlet and strongly mesh-sensitive.

**Underlying data:**

| mesh | walks | Zombie RMSE | WoSt RMSE | ratio Z/W |
|---|---|---|---|---|
| Bunny | 16 | 0.0374 | 0.0414 | 0.9032 |
| Bunny | 64 | 0.0193 | 0.0250 | 0.7734 |
| Bunny | 256 | 0.0160 | 0.0131 | 1.2224 |
| Bunny | 1024 | 0.0140 | 0.0114 | 1.2305 |
| Spot | 16 | 0.7003 | 0.2522 | 2.7770 |
| Spot | 64 | 0.2758 | 0.1933 | 1.4266 |
| Spot | 256 | 0.1425 | 0.1744 | 0.8169 |
| Spot | 1024 | 0.1107 | 0.1671 | 0.6626 |

**Individual mesh views (Mixed Neumann):**

![Bunny and Spot Mixed Neumann RMSE vs walks — side-by-side panels](fig2_merged_neumann_panels.png)

*Fig 2a/2b merged.* Bunny (left) and Spot (right) Mixed Neumann RMSE vs walks. Bunny: WoSt tracks below Zombie initially but the gap narrows. Spot: the most striking case — Zombie continues to improve while WoSt plateaus, suggesting residual systematic error.

Bunny shows improvement with more walks, but the high-walk Neumann error does not drop as cleanly as the Dirichlet case. Spot is substantially harder: WoSt RMSE remains high even at larger walk counts.

### Why can Zombie outperform WoSt on Spot at high walk counts?

At 256 walks, Spot Zombie RMSE is **0.142** while WoSt RMSE is **0.174**; at 1024 walks, Zombie RMSE is **0.111** while WoSt RMSE is **0.167**. WoSt uses much shorter mean paths than Zombie, but shorter paths do not guarantee lower error. This suggests possible residual systematic error from reflection, epsilon handling, local geometry, or implementation differences. Several hypotheses are plausible: WoSt's more aggressive geometry-aware steps may be more sensitive to radius, normal, or reflection errors near rough boundaries; Spot's coarser mesh and higher normal variation may accumulate systematic reflection error; Zombie's longer paths may be more conservative. These hypotheses are consistent with the diagnostics but not proven by the current experiments.

---

## 7. Experiment 3: Epsilon and Boundary-Bias Indicator

![Mixed Neumann RMSE under different epsilon values at 256 walks](fig3_neumann_epsilon_sweep.png)

**Figure 3.** Mixed Neumann RMSE under different epsilon values at 256 walks. Coarse epsilon produces the largest errors.

**Underlying data:**

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

![Epsilon-vs-half-epsilon boundary-bias indicator summary](fig4_boundary_bias_indicator_summary.png)

**Figure 4.** Boundary-bias indicator summary. The quantity is an epsilon sensitivity indicator rather than an exact bias decomposition.

**Main claim:** Coarse epsilon can dominate mixed Neumann error, and the boundary-bias indicator is larger on Spot.

**Underlying data:**

| mesh | mean bias | max bias | p95 bias | mean norm bias | RMSE ε | RMSE ε/2 |
|---|---|---|---|---|---|---|
| Bunny | 0.0094 | 0.1995 | 0.0350 | 0.3141 | 0.0325 | 0.0228 |
| Spot | 0.0434 | 0.6025 | 0.1582 | 0.3205 | 0.1981 | 0.1755 |

The epsilon sweep shows much larger RMSE at coarse epsilon ($10^{-2}$) in the mixed Neumann setting. The boundary-bias indicator compares epsilon and half-epsilon estimates; it is spatially and mesh dependent, consistent with the later controlled distance-bin results.

---

## 8. Experiment 4: Geometry-Sensitive Pointwise Diagnostics

![Top-10 geometry correlations by absolute Pearson correlation](fig5_top10_geometry_correlations.png)

**Figure 5.** Simplified top-10 geometry correlations. The dominant trend is that normalized nearest-distance proxy is the strongest predictor; local normal variation appears as a secondary descriptor.

**Underlying data:**

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

![Pointwise Neumann absolute error scatter](fig5b_pointwise_error_scatter.png)

**Figure 6.** Pointwise Neumann absolute error scatter. Near-boundary regions contain many of the difficult points.

**Main claim:** The normalized nearest-surface-distance proxy is the strongest observed pointwise predictor of high error, high variance, long paths, and boundary-bias indicators.

The geometry-sensitive analysis shows that points close to the inner boundary are consistently harder. Local normal variation and related mesh features are useful secondary descriptors, but they are not a standalone explanation. This motivates distance-controlled comparisons before attributing the Bunny/Spot gap to mesh geometry alone.

---

## 9. Experiment 5: Distance-Controlled Bins

![Matched-bin mean absolute error with 95% confidence intervals](fig6_matched_bin_abs_error_ci.png)

**Figure 7.** Matched-bin mean absolute error with across-query 95% confidence intervals. Spot remains higher-error in bins 1–3; the gap shrinks with distance.

![Matched-bin boundary-bias indicator with 95% confidence intervals](fig7_matched_bin_boundary_bias_ci.png)

**Figure 8.** Matched-bin boundary-bias indicator with 95% confidence intervals. Close-boundary bins show larger epsilon sensitivity, especially for Spot.

![Mean WoSt steps by matched nearest-distance proxy bin](fig8_matched_bin_mean_steps.png)

**Figure 9.** Mean WoSt steps by matched bin. Longer paths concentrate near the boundary, but path length alone does not determine RMSE.

**Main claim:** Spot remains higher-error than Bunny in matched bins 1–3, but the gap shrinks with distance.

The controlled experiment samples query points by normalized nearest-surface-distance proxy bins:

| Bin | Range | Description |
|-----|-------|-------------|
| 1 | [0.05, 0.15] | Very close to boundary |
| 2 | [0.15, 0.30] | Close to boundary |
| 3 | [0.30, 0.60] | Moderate distance |
| 4 | [0.60, 1.00] | Far from boundary |

Spot remains higher-error in matched bins 1–3, with descriptive Spot/Bunny error ratios of about **3.38×**, **3.69×**, and **1.37×**. This supports residual mesh, shape, reflection, or normal effects after reducing the query-distance confounder. The shrinking ratio shows that query-distance distribution was a major confounding factor. Spot has no valid sampled points in bin 4 under the current setup, so far-boundary matched conclusions are incomplete.

### Controlled matched-bin ratios

| bin | Spot/Bunny error | Spot/Bunny bias indicator | Spot/Bunny steps | status |
|-----|-----------------|--------------------------|-----------------|--------|
| 1 | 3.383 | 3.334 | 1.280 | descriptive matched-bin ratio |
| 2 | 3.685 | 2.371 | 0.765 | descriptive matched-bin ratio |
| 3 | 1.371 | 1.833 | 0.746 | descriptive matched-bin ratio |
| 4 | NA | NA | NA | missing mesh/bin pair |

### Recomputed per-query matched-bin statistics

| mesh | bin | n | mean abs error | abs error std | abs error SE | abs error 95% CI | RMSE | mean steps | mean sample variance | mean bias indicator |
|------|-----|---|----------------|---------------|--------------|------------------|------|------------|---------------------|---------------------|
| bunny | 1 | 21 | 0.0357 | 0.0238 | 0.0052 | [0.0255, 0.0459] | 0.0426 | 99.09 | 0.0435 | 0.0156 |
| bunny | 2 | 22 | 0.0152 | 0.0110 | 0.0024 | [0.0106, 0.0198] | 0.0186 | 61.23 | 0.0363 | 0.0147 |
| bunny | 3 | 23 | 0.0120 | 0.0087 | 0.0018 | [0.0084, 0.0155] | 0.0147 | 37.96 | 0.0316 | 0.0129 |
| bunny | 4 | 24 | 0.0060 | 0.0060 | 0.0012 | [0.0036, 0.0084] | 0.0084 | 26.63 | 0.0199 | 0.0069 |
| spot | 1 | 22 | 0.1209 | 0.1165 | 0.0248 | [0.0722, 0.1695] | 0.1660 | 126.84 | 0.6902 | 0.0520 |
| spot | 2 | 24 | 0.0561 | 0.0505 | 0.0103 | [0.0359, 0.0763] | 0.0748 | 46.83 | 0.2288 | 0.0348 |
| spot | 3 | 24 | 0.0164 | 0.0154 | 0.0032 | [0.0102, 0.0226] | 0.0223 | 28.32 | 0.1083 | 0.0237 |

The 95% confidence intervals are computed across valid query points within each distance bin (spatial/query variability, not repeated-seed intervals).

---

## 10. Diagnostic and Optimization Tools

These tools are presented as **diagnostics** rather than general accuracy improvements.

### 10.1 Adaptive Sampling

![Bunny adaptive sampling tradeoff](fig10a_bunny_adaptive_tradeoff.png)

**Figure 10.** Bunny adaptive sampling tradeoff. The figure asks where variance concentrates and how many samples the adaptive rule allocates.

![Spot adaptive sampling tradeoff](fig10b_spot_adaptive_tradeoff.png)

**Figure 11.** Spot adaptive sampling tradeoff. Spot remains close to the maximum sample count, indicating widespread high variance.

**Underlying data:**

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

Adaptive sampling asks where variance concentrates. Spot saturates near the maximum sample count, suggesting high variance is widespread — so adaptive sampling is less useful as a speedup but still useful as a variance diagnostic.

**Spatial distribution of adaptive samples:**

![Bunny variance-adaptive samples map](fig10c_bunny_variance_adaptive_samples_map.png)

*Fig 10c.* Bunny variance-adaptive sampling spatial map (left: sample count per query point; right: samples used vs pilot variance). Bunny shows a clear variance–sample correlation: higher variance points receive more samples.

![Spot variance-adaptive samples map](fig10d_spot_variance_adaptive_samples_map.png)

*Fig 10d.* Spot variance-adaptive sampling spatial map. Most points saturate at the maximum sample count (~1000), confirming that high variance is widespread across the Spot domain.

### 10.2 Live Path Trace

![2D path trace](fig11_spot_live_trace.png)
**Figure 12.** A 2D slice of the live path trace.

Try the [3D live demo](/files/wost-final-project/live_trace_interactive_3d.html) to inspect the reflection-heavy behavior near difficult Neumann regions.

### 10.3 Antithetic Sampling

![Antithetic sampling variance diagnostic](fig13_antithetic_variance_diagnostic.png)

**Figure 13.** Antithetic sampling variance diagnostic. Paired directions reduce measured sample variance in diagnostic runs on both Bunny and Spot.

This addresses estimator variance, not epsilon or geometry-related systematic error.

**Underlying data:**

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

### 10.4 Lazy Refinement

![Lazy star-radius refinement runtime diagnostic](fig14_lazy_refinement_runtime.png)

**Figure 14.** Lazy star-radius refinement runtime diagnostic. The lazy x1 setting substantially reduces runtime while preserving tested mean RMSE.

**Underlying data:**

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

### 10.5 Summary Table

| mesh | normal variance | antithetic variance | variance ratio | normal RMSE | antithetic RMSE | full exact sec | lazy x1 sec | speedup | lazy x1 RMSE | exact refinement ratio |
|------|----------------|---------------------|----------------|-------------|-----------------|----------------|-------------|---------|--------------|------------------------|
| bunny | 0.0176 | 0.0049 | **0.2805** | 0.0059 | 0.0044 | 237.46 | 26.09 | **9.1×** | 0.0060 | 0.0936 |
| spot | 0.2471 | 0.0682 | **0.2760** | 0.0224 | 0.0162 | 25.62 | 3.12 | **8.2×** | 0.0219 | 0.0771 |

---

## 11. Discussion

The main lesson is that Monte Carlo PDE solvers can look healthy under Dirichlet validation while becoming much more sensitive under mixed Neumann conditions. Dirichlet paths terminate at boundary values; mixed Neumann paths interact with normals and reflection behavior. That interaction makes boundary proximity, epsilon termination, and local mesh geometry more important.

The strongest available pointwise signal is the normalized nearest-distance proxy. Mesh features such as local normal variation are plausible contributors, especially for a coarse mesh such as Spot, but Bunny and Spot alone do not isolate causality. The unresolved Spot high-walk anomaly is especially important: Zombie can outperform WoSt on Spot at high walk counts even though WoSt uses shorter paths. That points to residual systematic effects rather than pure Monte Carlo variance.

---

## 12. Practical Takeaways

- Run Dirichlet sanity checks as the first validation step before interpreting mixed Neumann failures.
- Inspect query-distance proxy distributions before comparing meshes.
- Treat near-boundary mixed Neumann queries as high-risk.
- Avoid coarse epsilon such as $10^{-2}$ near the boundary.
- If adaptive sampling saturates near the maximum sample count, interpret it as widespread high variance.
- Do not assume shorter paths imply lower RMSE.
- Use live traces only as qualitative diagnostics.

---

## 13. Limitations

- Only Bunny and Spot tested in the controlled cross-mesh analysis.
- The nearest-distance variable is a proxy, not exact signed distance.
- Matched-bin confidence intervals are across query points, not repeated seeds.
- Matched-bin valid sample counts are small; ratios are descriptive.
- Spot bin 4 is unavailable; far-from-boundary comparison is incomplete.
- Fixed seeds and limited repeated-seed statistics restrict uncertainty analysis.
- Bunny and Spot alone cannot establish general geometry causality.
- The Zombie-vs-WoSt anomaly remains unresolved.
- The epsilon-vs-half-epsilon boundary-bias value is an indicator, not a true exact-solution bias decomposition.

---

## 14. Claim-Evidence-Limitation Table

| Claim | Evidence | Limitation / Caution |
|-------|----------|---------------------|
| Dirichlet validation | Bunny and Spot Dirichlet RMSE decreases with walks in Figure 1 and summary CSVs. | Validates the baseline pipeline, not every boundary condition. |
| Mixed Neumann geometry sensitivity | Figure 2 and Neumann tables show less uniform convergence and larger Spot errors. | Bunny and Spot differ in shape, mesh resolution, and query distribution. |
| Boundary proximity is the strongest observed predictor | Geometry correlations and pointwise scatter identify normalized nearest-distance proxy as the strongest stable predictor. | The variable is a nearest-distance proxy, not exact signed distance. |
| Spot remains harder after distance matching | Controlled bins 1–3 show Spot/Bunny error ratios of ~3.38×, 3.69×, and 1.37×. | Ratios are descriptive; Spot bin 4 is missing. |
| Coarse epsilon induces boundary-sensitive error | Epsilon sweep and boundary-bias indicator show much larger error at coarse epsilon. | The indicator is not an exact bias decomposition. |
| Optimization tools are diagnostic, not general fixes | Adaptive, antithetic, lazy refinement, BVH, and live-trace outputs expose variance/runtime/path behavior. | They should not be presented as guaranteed accuracy improvements. |

---

## 15. Conclusion

The reproduced WoSt pipeline passes the basic Dirichlet sanity check, but mixed Neumann boundary conditions reveal strong geometry sensitivity. Boundary proximity and epsilon handling explain a large part of the error structure. Controlled distance bins show that Spot remains harder than Bunny in matched bins 1–3, while the shrinking gap indicates that the original query-distance distribution was a major confounder. The safest final interpretation is therefore not that one method or mesh property fully explains the behavior, but that **mixed Neumann WoSt requires careful boundary-distance, epsilon, and geometry diagnostics**.

---

## 16. Interactive 3D Demos

Two interactive 3D viewers are available below. Both support **free arcball rotation** (drag to orbit, scroll to zoom) with no gimbal lock or angle clamping.

### 16.1 Boundary Bias Detector 3D

Explore geometry-sensitive diagnostic results on the 3D mesh. Rotate freely to inspect bias distribution from any angle.

**[🎮 Launch Boundary Bias Detector 3D →](/files/wost-final-project/boundary_bias_detector_3d.html)**

### 16.2 Live Trace 3D

Visualize individual walk paths and reflection behavior near difficult Neumann regions.

**[🎮 Launch Live Trace 3D →](/files/wost-final-project/live_trace_interactive_3d.html)**

---

## 17. Appendix

### Appendix A. Extra Diagnostic Figures

![Full dense geometry-correlation figure](fig5_strongest_geometry_correlations.png)

**Appendix Figure B1.** Full dense geometry-correlation figure. Retained for provenance and detailed labels; the simplified top-10 version is used in the main text.

![Bunny epsilon-distance RMSE heatmap](fig9a_bunny_epsilon_distance_rmse.png)

**Appendix Figure B2.** Bunny epsilon-distance RMSE heatmap, showing how epsilon sensitivity varies by nearest-distance proxy bin.

![Spot epsilon-distance RMSE heatmap](fig9b_spot_epsilon_distance_rmse.png)

**Appendix Figure B3.** Spot epsilon-distance RMSE heatmap, showing stronger near-boundary sensitivity at coarse epsilon.

**Underlying data:**

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

![BVH versus brute-force supporting benchmark](fig12_bvh_vs_bruteforce_supporting.png)

**Appendix Figure B4.** BVH versus brute-force geometry-query benchmark. Supporting engineering evidence for acceleration inside the WoSt implementation; not used as a solver-accuracy claim.

### Appendix B. Full Command Log

All experiment commands and raw outputs are preserved in the command log:

**[📄 View Full Command Log →](command_log.txt)**

The log includes complete convergence sweeps, epsilon sweeps, grid benchmarks, geometry benchmarks, mixed Neumann runs, optimization diagnostics (adaptive sampling, antithetic sampling, lazy refinement), variance-adaptive tests, live path traces, and boundary bias detector runs for both Bunny and Spot meshes.
