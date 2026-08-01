"""
Reproduce the PenSimEnv Bayesian-Optimization figure from the SMPL docs.

The 10 random + 1000 GPEI batches are already shipped in
    smpl-experiments/pensimenv_experiments/pensimpy_1010_samples/
as one CSV per batch. Batch yield (kg) = sum of the 'Yield Per Step' column.

Usage:
    python plot_bo_yields.py --data_dir path/to/pensimpy_1010_samples --out bayes_opt.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_YIELD = 3640.0  # baseline recipe, no optimisation (per SMPL docs)


def batch_yield(path):
    return pd.read_csv(path, usecols=["Yield Per Step"])["Yield Per Step"].sum()


def collect(data_dir, prefix, n):
    return np.array([batch_yield(os.path.join(data_dir, f"{prefix}_{i}.csv"))
                     for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="pensimpy_1010_samples")
    ap.add_argument("--n_random", type=int, default=10)
    ap.add_argument("--n_gpei", type=int, default=1000)
    ap.add_argument("--out", default="bayes_opt.png")
    ap.add_argument("--time_costs", default=None,
                    help="optional .npy of per-step gpei wall-clock seconds")
    args = ap.parse_args()

    rand = collect(args.data_dir, "random_batch", args.n_random)
    gpei = collect(args.data_dir, "gpei_batch", args.n_gpei)

    x_rand = np.arange(-args.n_random, 0)
    x_gpei = np.arange(args.n_gpei)

    # running statistics over the full search (random starts included)
    allyield = np.concatenate([rand, gpei])
    best_so_far = np.maximum.accumulate(allyield)
    avg_so_far = np.cumsum(allyield) / np.arange(1, allyield.size + 1)
    x_all = np.concatenate([x_rand, x_gpei])

    time_costs = np.load(args.time_costs) if args.time_costs else None
    nrows = 2 if time_costs is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 4.2 * nrows), squeeze=False)
    ax = axes[0][0]

    ax.plot([0], [BASE_YIELD], "D", color="red", ms=7, label="base yield")
    ax.plot(x_rand, rand, "*", color="green", ms=7, label="random yield")
    ax.plot(x_gpei, gpei, "*", color="blue", ms=3, alpha=0.65, label="gpei yield")
    ax.plot(x_all, best_so_far, "-", color="red", lw=1.4, label="best gpei yield so far")
    ax.plot(x_all, avg_so_far, "--", color="cyan", lw=1.4, label="avg gpei yield so far")

    ax.set_title("Bayesian Opt")
    ax.set_xlabel("batch id")
    ax.set_ylabel("yield [kg]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    if time_costs is not None:
        ax2 = axes[1][0]
        ax2.plot(np.arange(time_costs.size), time_costs, "-", color="red",
                 lw=0.8, label="gpei time cost")
        ax2.set_xlabel("batch id")
        ax2.set_ylabel("time cost [s]")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)

    print(f"base yield            : {BASE_YIELD:.1f} kg")
    print(f"best yield            : {allyield.max():.1f} kg "
          f"({100 * (allyield.max() / BASE_YIELD - 1):.1f}% over base)")
    print(f"mean yield (all 1010) : {allyield.mean():.1f} kg "
          f"({100 * (allyield.mean() / BASE_YIELD - 1):.1f}% over base)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
