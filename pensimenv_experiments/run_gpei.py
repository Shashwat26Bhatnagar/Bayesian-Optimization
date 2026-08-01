"""
GPEI (Gaussian-Process Bayesian Optimisation, Expected Improvement) baseline for
PenSimEnv / PenSimPy -- 10 random starts + N GPEI trials, one full 230 h
fermentation batch per trial.

Search space: every non-zero setpoint of the seven default recipe profiles
(Fs, Foil, Fg, pressure, discharge, Fw, Fpaa), each free to move within
+/- 10% of its default value -- this is what the SMPL docs mean by
"restricted the input search space to be within +/- 10% of setpoint inputs",
and it is consistent with the shipped pensimpy_1010_samples CSVs.

Objective: total batch yield in kg (maximise).

Writes:
  out_dir/random_batch_{i}.csv   (10)
  out_dir/gpei_batch_{i}.csv     (N)
  out_dir/time_costs.npy         wall-clock seconds spent in the optimiser per trial
so that plot_bo_yields.py can draw the figure directly.

Requires: fastodeint, pensimpy, ax-platform.
"""
import argparse
import copy
import os
import time

import numpy as np

from ax.service.ax_client import AxClient, ObjectiveProperties
from pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE,
    PAA_DEFAULT_PROFILE,
)
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.peni_env_setup import PenSimEnv

DEFAULTS = {
    FS: FS_DEFAULT_PROFILE,
    FOIL: FOIL_DEFAULT_PROFILE,
    FG: FG_DEFAULT_PROFILE,
    PRES: PRESS_DEFAULT_PROFILE,
    DISCHARGE: DISCHARGE_DEFAULT_PROFILE,
    WATER: WATER_DEFAULT_PROFILE,
    PAA: PAA_DEFAULT_PROFILE,
}


def build_search_space(bound=0.10):
    """One tunable parameter per non-zero setpoint, within +/- `bound` of default."""
    params = []
    for name, profile in DEFAULTS.items():
        for idx, sp in enumerate(profile):
            v = float(sp["value"])
            if v == 0.0:          # a zero setpoint stays zero (e.g. discharge off)
                continue
            params.append({
                "name": f"{name}__{idx}",
                "type": "range",
                "bounds": [v * (1 - bound), v * (1 + bound)],
                "value_type": "float",
            })
    return params


def recipe_from_params(params):
    recipe_dict = {}
    for name, profile in DEFAULTS.items():
        sp_list = copy.deepcopy(profile)
        for idx, sp in enumerate(sp_list):
            key = f"{name}__{idx}"
            if key in params:
                sp["value"] = float(params[key])
        recipe_dict[name] = Recipe(sp_list, name)
    return RecipeCombo(recipe_dict=recipe_dict)


def evaluate(params, seed):
    env = PenSimEnv(recipe_combo=recipe_from_params(params))
    df, batch_yield = env.get_batches(random_seed=seed, include_raman=False)
    return df, float(batch_yield)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_random", type=int, default=10)
    ap.add_argument("--n_gpei", type=int, default=1000)
    ap.add_argument("--bound", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0,
                    help="env random_seed; keep fixed so batches are comparable")
    ap.add_argument("--out_dir", default="my_bo_samples")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    client = AxClient(random_seed=args.seed, verbose_logging=False)
    client.create_experiment(
        name="pensim_gpei",
        parameters=build_search_space(args.bound),
        objectives={"yield": ObjectiveProperties(minimize=False)},
        choose_generation_strategy_kwargs={
            "num_initialization_trials": args.n_random,   # Sobol/random starts
            "max_parallelism_override": 1,
        },
    )

    time_costs, best = [], -np.inf
    total = args.n_random + args.n_gpei
    for t in range(total):
        t0 = time.time()
        params, trial_index = client.get_next_trial()   # the expensive GP step
        time_costs.append(time.time() - t0)

        df, y = evaluate(params, seed=args.seed)
        client.complete_trial(trial_index=trial_index, raw_data={"yield": (y, 0.0)})

        if t < args.n_random:
            fname = f"random_batch_{t}.csv"
        else:
            fname = f"gpei_batch_{t - args.n_random}.csv"
        df.to_csv(os.path.join(args.out_dir, fname), index=False)

        best = max(best, y)
        print(f"[{t + 1}/{total}] yield={y:9.2f} kg | best={best:9.2f} kg "
              f"| optimiser {time_costs[-1]:6.2f}s", flush=True)

    np.save(os.path.join(args.out_dir, "time_costs.npy"), np.array(time_costs))
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
