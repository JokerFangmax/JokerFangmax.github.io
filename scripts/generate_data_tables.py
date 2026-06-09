#!/usr/bin/env python3
"""Generate data tables from raw CSVs for the project page."""

import csv
import math
from pathlib import Path

ROOT = Path(r"C:\THU\projects\WoSt_Final_project-1")
RERUN = ROOT / "experiments/rerun_cross_mesh_20260606"
GEOM = ROOT / "experiments/geometry_sensitive_analysis_20260606"
CONTROLLED = ROOT / "experiments/controlled_geometry_experiments_20260606"


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value, digits=4):
    x = f(value)
    if not math.isfinite(x):
        return "NA"
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 10:
        return f"{x:.3f}"
    return f"{x:.{digits}f}"


def md_table(rows, fields, labels=None):
    labels = labels or fields
    out = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out)


# ========== Fig 1: Dirichlet convergence ==========
def table_dirichlet():
    out = []
    for mesh in ["bunny", "spot"]:
        rows = read_csv(RERUN / f"zombie_{mesh}_dirichlet/zombie_vs_wost_summary.csv")
        rows = [r for r in rows if r["benchmark_name"] == "convergence"]
        rows = sorted(rows, key=lambda r: f(r["walks_per_point"]))
        for r in rows:
            out.append({
                "mesh": mesh.capitalize(),
                "walks": int(f(r["walks_per_point"])),
                "Zombie RMSE": fmt(r["zombie_rmse"]),
                "WoSt RMSE": fmt(r["wost_rmse"]),
                "ratio Z/W": fmt(f(r["zombie_rmse"]) / f(r["wost_rmse"])),
            })
    return md_table(out, ["mesh", "walks", "Zombie RMSE", "WoSt RMSE", "ratio Z/W"])


# ========== Fig 2: Neumann convergence ==========
def table_neumann_convergence():
    out = []
    for mesh in ["bunny", "spot"]:
        rows = read_csv(RERUN / f"zombie_{mesh}_neumann/zombie_vs_wost_neumann_summary.csv")
        rows = [r for r in rows if r["benchmark_name"] == "neumann_convergence"]
        rows = sorted(rows, key=lambda r: f(r["walks_per_point"]))
        for r in rows:
            out.append({
                "mesh": mesh.capitalize(),
                "walks": int(f(r["walks_per_point"])),
                "Zombie RMSE": fmt(r["zombie_rmse"]),
                "WoSt RMSE": fmt(r["wost_rmse"]),
                "ratio Z/W": fmt(f(r["zombie_rmse"]) / f(r["wost_rmse"])),
            })
    return md_table(out, ["mesh", "walks", "Zombie RMSE", "WoSt RMSE", "ratio Z/W"])


# ========== Fig 3: Epsilon sweep ==========
def table_epsilon():
    out = []
    for mesh in ["bunny", "spot"]:
        rows = read_csv(RERUN / f"zombie_{mesh}_neumann/zombie_vs_wost_neumann_summary.csv")
        rows = [r for r in rows if r["benchmark_name"] == "neumann_epsilon"]
        rows = sorted(rows, key=lambda r: f(r["epsilon"]), reverse=True)
        for r in rows:
            out.append({
                "mesh": mesh.capitalize(),
                "epsilon": r["epsilon"],
                "Zombie RMSE": fmt(r["zombie_rmse"]),
                "WoSt RMSE": fmt(r["wost_rmse"]),
            })
    return md_table(out, ["mesh", "epsilon", "Zombie RMSE", "WoSt RMSE"])


# ========== Fig 4: Boundary bias ==========
def table_boundary_bias():
    out = []
    for mesh in ["bunny", "spot"]:
        rows = read_csv(RERUN / f"wost_{mesh}/diagnostics/boundary_bias_summary.csv")
        for r in rows:
            out.append({
                "mesh": mesh.capitalize(),
                "mean bias": fmt(r["mean_bias"]),
                "max bias": fmt(r["max_bias"]),
                "p95 bias": fmt(r["p95_bias"]),
                "mean norm bias": fmt(r["mean_normalized_bias"]),
                "RMSE ε": fmt(r["rmse_epsilon"]),
                "RMSE ε/2": fmt(r["rmse_epsilon_half"]),
            })
    return md_table(out, ["mesh", "mean bias", "max bias", "p95 bias", "mean norm bias", "RMSE ε", "RMSE ε/2"])


# ========== Fig 5: Top correlations ==========
def table_top_correlations():
    rows = read_csv(GEOM / "geometry_correlations.csv")
    selected = []
    for r in rows:
        pv = f(r.get("pearson_r"))
        if not math.isfinite(pv):
            continue
        if r.get("dataset") == "dirichlet_pointcloud":
            continue
        selected.append({**r, "abs_r": abs(pv)})
    selected = sorted(selected, key=lambda r: r["abs_r"], reverse=True)[:10]
    out = []
    for r in selected:
        out.append({
            "mesh": r["mesh"].capitalize(),
            "dataset": r["dataset"],
            "feature": r["feature"],
            "target": r["target"],
            "Pearson r": fmt(r["pearson_r"]),
            "n": r["n"],
        })
    return md_table(out, ["mesh", "dataset", "feature", "target", "Pearson r", "n"])


# ========== Fig 10: Adaptive sampling ==========
def table_adaptive():
    out = []
    for mesh in ["bunny", "spot"]:
        rows = read_csv(RERUN / f"wost_{mesh}/diagnostics/variance_adaptive_comparison.csv")
        for r in rows:
            out.append({
                "mesh": mesh.capitalize(),
                "method": r["method"],
                "RMSE": fmt(r["rmse"]),
                "MAE": fmt(r["mae"]),
                "mean variance": fmt(r["mean_sample_variance"]),
                "mean samples": fmt(r["mean_samples_used"], 1),
                "runtime (s)": fmt(r["runtime_seconds"]),
            })
    return md_table(out, ["mesh", "method", "RMSE", "MAE", "mean variance", "mean samples", "runtime (s)"])


# ========== Fig 13: Antithetic ==========
def table_antithetic():
    out = []
    for mesh in ["bunny", "spot"]:
        rows = read_csv(RERUN / f"wost_{mesh}/experiments/optimization_summary.csv")
        rows = [r for r in rows if r["experiment"] == "antithetic_compare"]
        for r in rows:
            out.append({
                "mesh": mesh.capitalize(),
                "method": r["method"],
                "seed": r["seed"],
                "RMSE": fmt(r["rmse"]),
                "mean variance": fmt(r["mean_sample_variance"]),
                "elapsed (s)": fmt(r["elapsed_seconds"]),
            })
    return md_table(out, ["mesh", "method", "seed", "RMSE", "mean variance", "elapsed (s)"])


# ========== Fig 14: Lazy refinement ==========
def table_lazy():
    out = []
    for mesh in ["bunny", "spot"]:
        rows = read_csv(RERUN / f"wost_{mesh}/experiments/optimization_summary.csv")
        rows = [r for r in rows if r["experiment"] == "lazy_refinement"]
        for r in rows:
            out.append({
                "mesh": mesh.capitalize(),
                "method": r["method"],
                "seed": r["seed"],
                "RMSE": fmt(r["rmse"]),
                "elapsed (s)": fmt(r["elapsed_seconds"]),
                "refinement ratio": fmt(r["refinement_ratio"]),
            })
    return md_table(out, ["mesh", "method", "seed", "RMSE", "elapsed (s)", "refinement ratio"])


# ========== Fig 9: Epsilon-distance heatmap ==========
def table_epsilon_distance():
    rows = read_csv(CONTROLLED / "epsilon_distance_sweep.csv")
    # Aggregate by mesh, distance_bin, epsilon (mean across walks)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["mesh"], r["distance_bin"], r["epsilon"])
        groups[key].append(r)
    out = []
    for (mesh, db, eps), grp in sorted(groups.items(), key=lambda x: (x[0][0], int(x[0][1]), f(x[0][2]))):
        rmses = [f(g["rmse"]) for g in grp]
        biases = [f(g["mean_bias"]) for g in grp]
        out.append({
            "mesh": mesh.capitalize(),
            "bin": db,
            "epsilon": eps,
            "n": len(grp),
            "mean RMSE": fmt(sum(rmses) / len(rmses)),
            "mean bias": fmt(sum(biases) / len(biases)),
        })
    return md_table(out, ["mesh", "bin", "epsilon", "n", "mean RMSE", "mean bias"])


if __name__ == "__main__":
    tables = {
        "DIRICHLET": table_dirichlet(),
        "NEUMANN_CONV": table_neumann_convergence(),
        "EPSILON": table_epsilon(),
        "BOUNDARY_BIAS": table_boundary_bias(),
        "TOP_CORR": table_top_correlations(),
        "ADAPTIVE": table_adaptive(),
        "ANTITHETIC": table_antithetic(),
        "LAZY": table_lazy(),
        "EPS_DIST": table_epsilon_distance(),
    }
    for name, content in tables.items():
        print(f"\n<!-- TABLE_{name}_START -->\n")
        print(content)
        print(f"\n<!-- TABLE_{name}_END -->\n")
