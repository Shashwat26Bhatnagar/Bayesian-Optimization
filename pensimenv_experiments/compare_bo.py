"""
Compare baseline Bayesian Optimisation yields with BND yields.

Usage:
    python compare_bo.py \
        --baseline_dir pensimpy_1010_samples \
        --bnd_dir /path/to/pensim_bnd \
        --out comparison.html
"""
import argparse
import csv
import os
import sys

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("plotly not installed. Run: pip install plotly")


BASE_YIELD = 3640.0  # baseline recipe, no optimisation


def batch_yield(path):
    """Sum 'Yield Per Step' using csv stdlib — no pandas needed."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = header.index("Yield Per Step")
        return sum(float(row[col]) for row in reader if row)


def collect(directory, prefix, n):
    return np.array([batch_yield(os.path.join(directory, f"{prefix}_{i}.csv"))
                     for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default="pensimpy_1010_samples",
                    help="folder with 10 random + 1000 gpei baseline CSVs")
    ap.add_argument("--bnd_dir", required=True,
                    help="folder with gpei + bnd_iter0 CSVs")
    ap.add_argument("--out", default="comparison.html")
    args = ap.parse_args()

    # --- count files automatically ---
    n_baseline_random = len([f for f in os.listdir(args.baseline_dir)
                             if f.startswith("random_batch_")])
    n_baseline_gpei = len([f for f in os.listdir(args.baseline_dir)
                           if f.startswith("gpei_batch_")])
    n_bnd_gpei = len([f for f in os.listdir(args.bnd_dir)
                      if f.startswith("gpei_batch_")])
    n_bnd_iter0 = len([f for f in os.listdir(args.bnd_dir)
                       if f.startswith("bnd_iter0_batch_")])

    print(f"Baseline: {n_baseline_random} random + {n_baseline_gpei} gpei")
    print(f"BND:      {n_bnd_gpei} gpei + {n_bnd_iter0} bnd_iter0")

    # --- load yields ---
    bl_rand = collect(args.baseline_dir, "random_batch", n_baseline_random)
    bl_gpei = collect(args.baseline_dir, "gpei_batch", n_baseline_gpei)

    bnd_gpei = collect(args.bnd_dir, "gpei_batch", n_bnd_gpei)
    bnd_iter0 = collect(args.bnd_dir, "bnd_iter0_batch", n_bnd_iter0)

    # running best / avg for baseline (random + gpei combined)
    bl_all = np.concatenate([bl_rand, bl_gpei])
    bl_best = np.maximum.accumulate(bl_all)
    bl_avg = np.cumsum(bl_all) / np.arange(1, bl_all.size + 1)
    bl_x = np.arange(bl_all.size)

    # running best / avg for bnd (gpei + bnd_iter0)
    bnd_all = np.concatenate([bnd_gpei, bnd_iter0])
    bnd_best = np.maximum.accumulate(bnd_all)
    bnd_avg = np.cumsum(bnd_all) / np.arange(1, bnd_all.size + 1)
    bnd_x = np.arange(bnd_all.size)

    # --- colours ---
    C_BASE = "#b91c1c"       # dark red diamond
    C_BL_RAND = "#15803d"    # green
    C_BL_GPEI = "#2563eb"    # blue
    C_BL_BEST = "#dc2626"    # red line
    C_BL_AVG = "#06b6d4"     # cyan dashed
    C_BND_GPEI = "#9333ea"   # purple
    C_BND_ITER = "#f97316"   # orange
    C_BND_BEST = "#c2410c"   # dark orange line
    C_BND_AVG = "#a855f7"    # light purple dashed

    fig = go.Figure()

    # --- base yield ---
    fig.add_trace(go.Scatter(
        x=[0], y=[BASE_YIELD], mode="markers",
        marker=dict(symbol="diamond", size=12, color=C_BASE),
        name=f"Base yield ({BASE_YIELD:.0f} kg)",
    ))

    # --- baseline random ---
    fig.add_trace(go.Scatter(
        x=np.arange(n_baseline_random), y=bl_rand, mode="markers",
        marker=dict(symbol="star", size=8, color=C_BL_RAND),
        name=f"Baseline random ({n_baseline_random})",
        hovertemplate="batch %{x}<br>yield: %{y:.1f} kg",
    ))

    # --- baseline gpei ---
    fig.add_trace(go.Scatter(
        x=np.arange(n_baseline_random, bl_all.size), y=bl_gpei,
        mode="markers",
        marker=dict(symbol="circle", size=3, color=C_BL_GPEI, opacity=0.5),
        name=f"Baseline GPEI ({n_baseline_gpei})",
        hovertemplate="batch %{x}<br>yield: %{y:.1f} kg",
    ))

    # --- baseline running best ---
    fig.add_trace(go.Scatter(
        x=bl_x, y=bl_best, mode="lines",
        line=dict(color=C_BL_BEST, width=1.5),
        name=f"Baseline best so far ({bl_all.max():.1f} kg)",
    ))

    # --- baseline running avg ---
    fig.add_trace(go.Scatter(
        x=bl_x, y=bl_avg, mode="lines",
        line=dict(color=C_BL_AVG, width=1.5, dash="dash"),
        name=f"Baseline avg so far ({bl_all.mean():.1f} kg)",
    ))

    # --- bnd gpei ---
    fig.add_trace(go.Scatter(
        x=np.arange(n_bnd_gpei), y=bnd_gpei, mode="markers",
        marker=dict(symbol="diamond", size=8, color=C_BND_GPEI),
        name=f"BND GPEI ({n_bnd_gpei})",
        hovertemplate="batch %{x}<br>yield: %{y:.1f} kg",
    ))

    # --- bnd iter0 ---
    fig.add_trace(go.Scatter(
        x=np.arange(n_bnd_gpei, bnd_all.size), y=bnd_iter0, mode="markers",
        marker=dict(symbol="hexagon2", size=9, color=C_BND_ITER),
        name=f"BND iter0 ({n_bnd_iter0})",
        hovertemplate="batch %{x}<br>yield: %{y:.1f} kg",
    ))

    # --- bnd running best ---
    fig.add_trace(go.Scatter(
        x=bnd_x, y=bnd_best, mode="lines",
        line=dict(color=C_BND_BEST, width=2),
        name=f"BND best so far ({bnd_all.max():.1f} kg)",
    ))

    # --- bnd running avg ---
    fig.add_trace(go.Scatter(
        x=bnd_x, y=bnd_avg, mode="lines",
        line=dict(color=C_BND_AVG, width=2, dash="dash"),
        name=f"BND avg so far ({bnd_all.mean():.1f} kg)",
    ))

    fig.update_layout(
        title=dict(
            text="Baseline BO vs BND — Penicillin Batch Yield",
            font=dict(size=18),
        ),
        xaxis_title="Batch index",
        yaxis_title="Yield (kg)",
        template="plotly_white",
        legend=dict(
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ddd",
            borderwidth=1,
        ),
        hovermode="closest",
        height=600,
        margin=dict(t=60, b=50),
    )

    fig.write_html(args.out, include_plotlyjs="cdn")

    # --- summary table ---
    print(f"\n{'':30s} {'mean':>10s} {'std':>10s} {'best':>10s} {'worst':>10s}")
    print("-" * 72)
    for label, arr in [
        (f"Baseline random ({n_baseline_random})", bl_rand),
        (f"Baseline GPEI ({n_baseline_gpei})", bl_gpei),
        (f"BND GPEI ({n_bnd_gpei})", bnd_gpei),
        (f"BND iter0 ({n_bnd_iter0})", bnd_iter0),
    ]:
        print(f"{label:30s} {arr.mean():10.1f} {arr.std():10.1f} "
              f"{arr.max():10.1f} {arr.min():10.1f}")
    print("-" * 72)
    print(f"{'Base yield':30s} {BASE_YIELD:10.1f}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
