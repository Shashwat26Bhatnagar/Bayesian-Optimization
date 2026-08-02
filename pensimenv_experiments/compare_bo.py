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

    # auto-detect all bnd_iter* groups (iter0, iter1, iter2, ...)
    import re
    iter_ids = sorted({int(m.group(1))
                       for f in os.listdir(args.bnd_dir)
                       for m in [re.match(r"bnd_iter(\d+)_batch_\d+\.csv", f)]
                       if m})
    bnd_iters = {}  # {iter_id: np.array of yields}
    for it in iter_ids:
        n = len([f for f in os.listdir(args.bnd_dir)
                 if f.startswith(f"bnd_iter{it}_batch_")])
        bnd_iters[it] = collect(args.bnd_dir, f"bnd_iter{it}_batch", n)

    n_bnd_total = sum(len(v) for v in bnd_iters.values())
    print(f"Baseline: {n_baseline_random} random + {n_baseline_gpei} gpei")
    iter_summary = " + ".join(f"{len(bnd_iters[it])} iter{it}" for it in iter_ids)
    print(f"BND:      {n_bnd_gpei} gpei + {iter_summary}")

    # --- load yields ---
    bl_rand = collect(args.baseline_dir, "random_batch", n_baseline_random)
    bl_gpei = collect(args.baseline_dir, "gpei_batch", n_baseline_gpei)
    bnd_gpei = collect(args.bnd_dir, "gpei_batch", n_bnd_gpei)

    # running best / avg for baseline (random + gpei combined)
    bl_all = np.concatenate([bl_rand, bl_gpei])
    bl_best = np.maximum.accumulate(bl_all)
    bl_avg = np.cumsum(bl_all) / np.arange(1, bl_all.size + 1)
    bl_x = np.arange(bl_all.size)

    # running best / avg for bnd (gpei + all iters in order)
    bnd_all = np.concatenate([bnd_gpei] + [bnd_iters[it] for it in iter_ids])
    bnd_best = np.maximum.accumulate(bnd_all)
    bnd_avg = np.cumsum(bnd_all) / np.arange(1, bnd_all.size + 1)
    bnd_x = np.arange(bnd_all.size)

    # --- high-contrast palette ---
    # top subplot (baseline)
    C_BASE      = "#000000"   # black diamond
    C_BL_RAND   = "#2ca02c"   # vivid green
    C_BL_GPEI   = "#1f77b4"   # strong blue
    C_BL_BEST   = "#d62728"   # bright red
    C_BL_AVG    = "#ff7f0e"   # orange dashed

    # bottom subplot (bnd)
    C_BND_GPEI  = "#9467bd"   # purple
    ITER_COLORS = ["#e377c2", "#17becf", "#bcbd22", "#d62728",
                   "#8c564b", "#1f77b4", "#ff7f0e", "#2ca02c"]
    ITER_SYMBOLS = ["hexagon2", "triangle-up", "square", "cross",
                    "pentagon", "star-triangle-up", "bowtie", "hourglass"]
    C_BND_BEST  = "#d62728"   # bright red
    C_BND_AVG   = "#ff7f0e"   # orange dashed

    # shared y-axis range for fair visual comparison
    all_yields = np.concatenate([bl_all, bnd_all])
    y_lo = min(all_yields.min(), BASE_YIELD) - 100
    y_hi = all_yields.max() + 100

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Baseline BO", "BND"],
        shared_xaxes=False,
        vertical_spacing=0.12,
    )

    # ===================== TOP: Baseline =====================

    fig.add_trace(go.Scatter(
        x=[0], y=[BASE_YIELD], mode="markers",
        marker=dict(symbol="diamond", size=12, color=C_BASE,
                    line=dict(width=1, color="#555")),
        name=f"Base yield ({BASE_YIELD:.0f} kg)",
        legendgroup="baseline",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=np.arange(n_baseline_random), y=bl_rand, mode="markers",
        marker=dict(symbol="star", size=9, color=C_BL_RAND),
        name=f"Random starts ({n_baseline_random})",
        hovertemplate="batch %{x}<br>yield: %{y:.1f} kg",
        legendgroup="baseline",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=np.arange(n_baseline_random, bl_all.size), y=bl_gpei,
        mode="markers",
        marker=dict(symbol="circle", size=3, color=C_BL_GPEI, opacity=0.45),
        name=f"GPEI ({n_baseline_gpei})",
        hovertemplate="batch %{x}<br>yield: %{y:.1f} kg",
        legendgroup="baseline",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bl_x, y=bl_best, mode="lines",
        line=dict(color=C_BL_BEST, width=2),
        name=f"Best so far ({bl_all.max():.1f} kg)",
        legendgroup="baseline",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bl_x, y=bl_avg, mode="lines",
        line=dict(color=C_BL_AVG, width=2, dash="dash"),
        name=f"Avg so far ({bl_all.mean():.1f} kg)",
        legendgroup="baseline",
    ), row=1, col=1)

    # ===================== BOTTOM: BND =====================

    fig.add_trace(go.Scatter(
        x=[0], y=[BASE_YIELD], mode="markers",
        marker=dict(symbol="diamond", size=12, color=C_BASE,
                    line=dict(width=1, color="#555")),
        name=f"Base yield ({BASE_YIELD:.0f} kg)",
        legendgroup="bnd", showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=np.arange(n_bnd_gpei), y=bnd_gpei, mode="markers",
        marker=dict(symbol="diamond", size=9, color=C_BND_GPEI),
        name=f"GPEI ({n_bnd_gpei})",
        hovertemplate="batch %{x}<br>yield: %{y:.1f} kg",
        legendgroup="bnd",
    ), row=2, col=1)

    offset = n_bnd_gpei
    for i, it in enumerate(iter_ids):
        arr = bnd_iters[it]
        c = ITER_COLORS[i % len(ITER_COLORS)]
        sym = ITER_SYMBOLS[i % len(ITER_SYMBOLS)]
        fig.add_trace(go.Scatter(
            x=np.arange(offset, offset + len(arr)), y=arr, mode="markers",
            marker=dict(symbol=sym, size=9, color=c),
            name=f"iter{it} ({len(arr)})",
            hovertemplate="batch %{x}<br>yield: %{y:.1f} kg",
            legendgroup="bnd",
        ), row=2, col=1)
        offset += len(arr)

    fig.add_trace(go.Scatter(
        x=bnd_x, y=bnd_best, mode="lines",
        line=dict(color=C_BND_BEST, width=2),
        name=f"Best so far ({bnd_all.max():.1f} kg)",
        legendgroup="bnd",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=bnd_x, y=bnd_avg, mode="lines",
        line=dict(color=C_BND_AVG, width=2, dash="dash"),
        name=f"Avg so far ({bnd_all.mean():.1f} kg)",
        legendgroup="bnd",
    ), row=2, col=1)

    # ===================== Layout =====================

    fig.update_yaxes(title_text="Yield (kg)", range=[y_lo, y_hi], row=1, col=1)
    fig.update_yaxes(title_text="Yield (kg)", range=[y_lo, y_hi], row=2, col=1)
    fig.update_xaxes(title_text="Batch index", row=1, col=1)
    fig.update_xaxes(title_text="Batch index", row=2, col=1)

    fig.update_layout(
        title=dict(
            text="Penicillin Batch Yield — Baseline BO vs BND",
            font=dict(size=18),
        ),
        template="plotly_white",
        legend=dict(
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        hovermode="closest",
        height=800,
        margin=dict(t=70, b=50),
    )

    fig.write_html(args.out, include_plotlyjs="cdn")

    # --- summary table ---
    print(f"\n{'':30s} {'mean':>10s} {'std':>10s} {'best':>10s} {'worst':>10s}")
    print("-" * 72)
    rows = [
        (f"Baseline random ({n_baseline_random})", bl_rand),
        (f"Baseline GPEI ({n_baseline_gpei})", bl_gpei),
        (f"BND GPEI ({n_bnd_gpei})", bnd_gpei),
    ] + [(f"BND iter{it} ({len(bnd_iters[it])})", bnd_iters[it])
         for it in iter_ids]
    for label, arr in rows:
        print(f"{label:30s} {arr.mean():10.1f} {arr.std():10.1f} "
              f"{arr.max():10.1f} {arr.min():10.1f}")
    print("-" * 72)
    print(f"{'Base yield':30s} {BASE_YIELD:10.1f}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
