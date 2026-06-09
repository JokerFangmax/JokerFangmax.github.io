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

---

## 6. Experiment 2: Mixed Neumann Sensitivity

![Mixed Neumann RMSE versus walks for Bunny and Spot](fig2_mixed_neumann_rmse_vs_walks.png)

**Figure 2.** Mixed Neumann RMSE versus walks. Compared with the Dirichlet panel, this figure highlights less uniform convergence and the harder Spot case.

**Main claim:** Mixed Neumann behavior is less uniform than Dirichlet and strongly mesh-sensitive.

Bunny shows improvement with more walks, but the high-walk Neumann error does not drop as cleanly as the Dirichlet case. Spot is substantially harder: WoSt RMSE remains high even at larger walk counts.

### Why can Zombie outperform WoSt on Spot at high walk counts?

At 256 walks, Spot Zombie RMSE is **0.142** while WoSt RMSE is **0.174**; at 1024 walks, Zombie RMSE is **0.111** while WoSt RMSE is **0.167**. WoSt uses much shorter mean paths than Zombie, but shorter paths do not guarantee lower error. This suggests possible residual systematic error from reflection, epsilon handling, local geometry, or implementation differences. Several hypotheses are plausible: WoSt's more aggressive geometry-aware steps may be more sensitive to radius, normal, or reflection errors near rough boundaries; Spot's coarser mesh and higher normal variation may accumulate systematic reflection error; Zombie's longer paths may be more conservative. These hypotheses are consistent with the diagnostics but not proven by the current experiments.

---

## 7. Experiment 3: Epsilon and Boundary-Bias Indicator

![Mixed Neumann RMSE under different epsilon values at 256 walks](fig3_neumann_epsilon_sweep.png)

**Figure 3.** Mixed Neumann RMSE under different epsilon values at 256 walks. Coarse epsilon produces the largest errors.

![Epsilon-vs-half-epsilon boundary-bias indicator summary](fig4_boundary_bias_indicator_summary.png)

**Figure 4.** Boundary-bias indicator summary. The quantity is an epsilon sensitivity indicator rather than an exact bias decomposition.

**Main claim:** Coarse epsilon can dominate mixed Neumann error, and the boundary-bias indicator is larger on Spot.

The epsilon sweep shows much larger RMSE at coarse epsilon ($10^{-2}$) in the mixed Neumann setting. The boundary-bias indicator compares epsilon and half-epsilon estimates; it is spatially and mesh dependent, consistent with the later controlled distance-bin results.

---

## 8. Experiment 4: Geometry-Sensitive Pointwise Diagnostics

![Top-10 geometry correlations by absolute Pearson correlation](fig5_top10_geometry_correlations.png)

**Figure 5.** Simplified top-10 geometry correlations. The dominant trend is that normalized nearest-distance proxy is the strongest predictor; local normal variation appears as a secondary descriptor.

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

Adaptive sampling asks where variance concentrates. Spot saturates near the maximum sample count, suggesting high variance is widespread — so adaptive sampling is less useful as a speedup but still useful as a variance diagnostic.

### 10.2 Live Path Trace

![2D path trace](fig11_spot_live_trace.png)
**Figure 12.** A 2D slice of the live path trace.

Try the [3D live demo](/files/wost-final-project/live_trace_interactive_3d.html) to inspect the reflection-heavy behavior near difficult Neumann regions.

### 10.3 Antithetic Sampling

![Antithetic sampling variance diagnostic](fig13_antithetic_variance_diagnostic.png)

**Figure 13.** Antithetic sampling variance diagnostic. Paired directions reduce measured sample variance in diagnostic runs on both Bunny and Spot.

This addresses estimator variance, not epsilon or geometry-related systematic error.

### 10.4 Lazy Refinement

![Lazy star-radius refinement runtime diagnostic](fig14_lazy_refinement_runtime.png)

**Figure 14.** Lazy star-radius refinement runtime diagnostic. The lazy x1 setting substantially reduces runtime while preserving tested mean RMSE.

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

![BVH versus brute-force supporting benchmark](fig12_bvh_vs_bruteforce_supporting.png)

**Appendix Figure B4.** BVH versus brute-force geometry-query benchmark. Supporting engineering evidence for acceleration inside the WoSt implementation; not used as a solver-accuracy claim.

### Appendix B. Full Command Log

All experiment commands and raw outputs are preserved in the command log:

**[📄 View Full Command Log →](command_log.txt)**

The log includes complete convergence sweeps, epsilon sweeps, grid benchmarks, geometry benchmarks, mixed Neumann runs, optimization diagnostics (adaptive sampling, antithetic sampling, lazy refinement), variance-adaptive tests, live path traces, and boundary bias detector runs for both Bunny and Spot meshes.
