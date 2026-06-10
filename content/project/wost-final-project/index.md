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

## Overview

This project reproduces and extends **Walk-on-Stars (WoSt)**, a Monte Carlo solver for PDE boundary value problems, and studies how it behaves on 3D mesh domains with **mixed Neumann boundary conditions**.

The main finding is that WoSt behaves predictably in Dirichlet validation, but mixed Neumann conditions expose stronger sensitivity to boundary proximity, epsilon termination, and mesh geometry. Bunny and Spot show noticeably different error patterns, especially near the inner Neumann boundary.

## Highlights

- Reproduced a C++ Walk-on-Stars pipeline for 3D boundary value problems.
- Compared Dirichlet and mixed Neumann behavior on Bunny and Spot meshes.
- Identified boundary proximity as the strongest pointwise diagnostic signal.
- Found that coarse epsilon can dominate mixed Neumann error near the boundary.
- Built interactive 3D viewers for boundary bias and live path tracing.

## Interactive Demos

The interactive viewers are the best entry point for exploring the diagnostic behavior directly.

- **[Launch Boundary Bias Detector 3D](/files/wost-final-project/boundary_bias_detector_3d.html)**: inspect spatial boundary-bias indicators on the mesh.
- **[Launch Live Trace 3D](/files/wost-final-project/live_trace_interactive_3d.html)**: visualize reflection-heavy walk paths near difficult Neumann regions.
- **[Launch Spot Neumann 3D Viewer](/files/wost-final-project/neumann_mixed_grid16_walk128_3d.html)**: explore the Spot mixed Neumann scalar field on a 3D grid.

## Problem Setup

The experiments use a 3D domain formed by an outer cube with an inner triangle mesh removed. The manufactured reference solution is:

$$
u(x, y, z) = x + y + z, \quad \Delta u = 0
$$

For Dirichlet validation, both the outer cube and inner mesh prescribe the boundary value of \(u\). For the mixed Neumann benchmark, the outer cube remains Dirichlet while the inner mesh prescribes normal derivative data. This makes the solver depend more directly on surface normals, reflection behavior, local geometry, and the epsilon boundary tolerance.

## Experiment 1: Dirichlet Validation

![Dirichlet RMSE versus walks for Bunny and Spot, comparing WoSt with Zombie baseline](fig1_dirichlet_rmse_vs_walks.png)

**Main claim:** Dirichlet tests show ordinary Monte Carlo convergence and validate the basic pipeline before interpreting the harder mixed Neumann results.

As the number of walks increases, RMSE decreases on both Bunny and Spot. WoSt and the Zombie baseline remain close across tested walk counts, so this stage serves mainly as a sanity check rather than the central scientific result.

Full data: see the [command log](command_log.txt) / [data appendix](data_tables_output.txt).

## Experiment 2: Mixed Neumann Sensitivity

![Mixed Neumann RMSE versus walks for Bunny and Spot](fig2_mixed_neumann_rmse_vs_walks.png)

**Main claim:** Mixed Neumann behavior is less uniform than Dirichlet behavior and is much more mesh-sensitive.

Bunny still improves with more walks, but the convergence is less clean than in the Dirichlet case. Spot is substantially harder: high-walk WoSt error remains elevated, suggesting residual systematic effects from reflection, epsilon handling, local geometry, or implementation differences.

Full data: see the [command log](command_log.txt) / [data appendix](data_tables_output.txt).

## Experiment 3: Epsilon and Boundary Bias

![Mixed Neumann RMSE under different epsilon values at 256 walks](fig3_neumann_epsilon_sweep.png)

![Epsilon-vs-half-epsilon boundary-bias indicator summary](fig4_boundary_bias_indicator_summary.png)

**Main claim:** Coarse epsilon can dominate mixed Neumann error, especially near the inner Neumann boundary.

The epsilon sweep shows that large boundary tolerances are risky in the mixed Neumann setting. The epsilon-vs-half-epsilon indicator is not an exact bias decomposition, but it highlights where estimates change strongly as the boundary tolerance is refined.

Full data: see the [command log](command_log.txt) / [data appendix](data_tables_output.txt).

## Experiment 4: Geometry-Sensitive Diagnostics

![Top-10 geometry correlations by absolute Pearson correlation](fig5_top10_geometry_correlations.png)

![Pointwise Neumann absolute error scatter](fig5b_pointwise_error_scatter.png)

**Main claim:** The normalized nearest-surface-distance proxy is the strongest observed pointwise predictor of difficult queries.

Near-boundary points account for many high-error, high-variance, and long-path cases. Local normal variation and related mesh descriptors are useful secondary signals, but boundary proximity is the clearest diagnostic trend in this experiment.

Full data: see the [command log](command_log.txt) / [data appendix](data_tables_output.txt).

## Experiment 5: Distance-Controlled Comparison

![Matched-bin mean absolute error with 95% confidence intervals](fig6_matched_bin_abs_error_ci.png)

![Matched-bin boundary-bias indicator with 95% confidence intervals](fig7_matched_bin_boundary_bias_ci.png)

![Mean WoSt steps by matched nearest-distance proxy bin](fig8_matched_bin_mean_steps.png)

**Main claim:** Spot remains harder than Bunny after matching by boundary-distance bins, but the gap shrinks with distance.

This reduces the query-distance confounder before comparing meshes. The result suggests that boundary proximity explains a large part of the difficulty, while residual mesh, shape, reflection, or normal effects still matter in close-boundary bins.

Full data: see the [command log](command_log.txt) / [data appendix](data_tables_output.txt).

## Diagnostic Tools

### Adaptive Sampling

![Bunny adaptive sampling tradeoff](fig10a_bunny_adaptive_tradeoff.png)

![Spot adaptive sampling tradeoff](fig10b_spot_adaptive_tradeoff.png)

**Main claim:** Adaptive sampling is most useful here as a variance diagnostic, not as a guaranteed speedup.

Bunny shows clearer variance-dependent sample allocation. Spot tends to saturate near the maximum sample count, which suggests that high variance is widespread across the sampled domain.

Full data: see the [command log](command_log.txt) / [data appendix](data_tables_output.txt).

### Antithetic Sampling

![Antithetic sampling variance diagnostic](fig13_antithetic_variance_diagnostic.png)

**Main claim:** Antithetic pairing reduces measured sample variance in diagnostic runs.

This addresses estimator variance, but it does not directly solve epsilon sensitivity or geometry-related systematic error.

Full data: see the [command log](command_log.txt) / [data appendix](data_tables_output.txt).

### Lazy Refinement

![Lazy star-radius refinement runtime diagnostic](fig14_lazy_refinement_runtime.png)

**Main claim:** Lazy star-radius refinement substantially reduces runtime while preserving tested mean RMSE in these diagnostic runs.

The result is best read as an engineering optimization diagnostic rather than a general solver-accuracy claim.

Full data: see the [command log](command_log.txt) / [data appendix](data_tables_output.txt).

## Practical Takeaways

- Run Dirichlet sanity checks before interpreting mixed Neumann failures.
- Inspect boundary-distance distributions before comparing meshes.
- Treat near-boundary mixed Neumann queries as high-risk.
- Avoid coarse epsilon near the Neumann boundary.
- Do not assume shorter paths imply lower RMSE.
- Use live traces and boundary-bias views as qualitative diagnostics.

## Extra Figures

These supporting figures are retained for provenance and visual inspection, but the detailed numerical tables are intentionally kept out of the homepage.

![Bunny and Spot Dirichlet RMSE vs walks, side-by-side panels](fig1_merged_dirichlet_panels.png)

*Dirichlet single-mesh panels for Bunny and Spot, showing the convergence trend behind the combined validation figure.*

![Bunny and Spot Mixed Neumann RMSE vs walks, side-by-side panels](fig2_merged_neumann_panels.png)

*Mixed Neumann single-mesh panels, making the cleaner Bunny behavior and harder Spot case easier to compare.*

![Full dense geometry-correlation figure](fig5_strongest_geometry_correlations.png)

*Full geometry-correlation diagnostic retained for detailed feature labels and provenance.*

![Bunny epsilon-distance RMSE heatmap](fig9a_bunny_epsilon_distance_rmse.png)

*Bunny epsilon-distance heatmap, showing how boundary tolerance interacts with distance from the inner surface.*

![Spot epsilon-distance RMSE heatmap](fig9b_spot_epsilon_distance_rmse.png)

*Spot epsilon-distance heatmap, highlighting stronger near-boundary sensitivity at coarse epsilon.*

![Bunny variance-adaptive samples map](fig10c_bunny_variance_adaptive_samples_map.png)

*Bunny adaptive-sampling map, showing where the variance-based rule allocates more samples.*

![Spot variance-adaptive samples map](fig10d_spot_variance_adaptive_samples_map.png)

*Spot adaptive-sampling map, where many points approach the maximum sample budget.*

![2D live path trace](fig11_spot_live_trace.png)

*A 2D slice of a live walk trace, useful as a qualitative view of reflection-heavy behavior.*

![BVH versus brute-force supporting benchmark](fig12_bvh_vs_bruteforce_supporting.png)

*Supporting geometry-query benchmark for BVH acceleration; included as engineering context rather than a solver-accuracy result.*

## Full Materials

The homepage keeps the main story compact. Complete experiment commands, raw outputs, convergence sweeps, epsilon sweeps, diagnostic runs, and generated tables are preserved in the project materials:

- **[View Full Command Log](command_log.txt)**
- **[View Data Appendix](data_tables_output.txt)**
- Generated table scripts in `scripts/`
- Interactive demos in `/files/wost-final-project/`
