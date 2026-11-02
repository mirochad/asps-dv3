"""Run exact/TMC convergence on one fixed game and three sampling seeds."""
import argparse
import itertools
import json
import resource
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exact_shapley import run
from tmc_shapley import tmc_shapley
from valuation_game import LogisticGame


def benchmark(output, checkpoints, split_seed=0, tolerance=0.01, patience=3):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    # Recompute the reference, never silently reuse a stale file.
    exact_metrics = run(split_seed, destination / "exact")
    exact = pd.read_csv(destination / "exact/values.csv")["value"].to_numpy()
    rows, value_rows, stability_rows = [], [], []
    snapshots = {}
    for seed in (0, 1, 2):
        # Data/model seed is fixed; only the permutation seed changes.
        game = LogisticGame(split_seed)
        for snapshot in tmc_shapley(game, game.n, checkpoints, seed, tolerance, patience):
            values = snapshot.pop("values")
            budget = snapshot["permutations"]
            snapshots[seed, budget] = values
            rows.append(dict(seed=seed, **snapshot, utility_evaluations=game.calls,
                             model_fits=game.fits, rmse=float(np.sqrt(np.mean((values-exact)**2))),
                             spearman=float(spearmanr(values, exact).statistic),
                             efficiency_residual=float(values.sum()-exact.sum()),
                             peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
            value_rows.extend(dict(seed=seed, permutations=budget, point=i, value=float(v), exact=float(exact[i])) for i,v in enumerate(values))
        print(f"Finished estimator seed {seed}", flush=True)
    for budget in sorted(set(checkpoints)):
        for a,b in itertools.combinations((0,1,2), 2):
            stability_rows.append(dict(permutations=budget, seed_a=a, seed_b=b,
                                       spearman=float(spearmanr(snapshots[a,budget], snapshots[b,budget]).statistic)))
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "convergence.csv", index=False)
    pd.DataFrame(value_rows).to_csv(destination / "values.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(destination / "stability.csv", index=False)
    (destination / "config.json").write_text(json.dumps(dict(split_seed=split_seed, estimator_seeds=[0,1,2], checkpoints=sorted(set(checkpoints)), tolerance=tolerance, patience=patience, caching=False), indent=2))
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for seed, group in frame.groupby("seed"):
        axes[0].plot(group.permutations,group.rmse,"o-",label=f"seed {seed}")
        axes[1].plot(group.permutations,group.spearman,"o-")
        axes[2].plot(group.wall_seconds,group.rmse,"o-")
    axes[2].scatter([exact_metrics["wall_seconds"]],[0],marker="*",s=100,color="black",label="Exact")
    axes[2].legend()
    for ax, xlabel, ylabel in zip(axes,["Permutations","Permutations","Valuation time (seconds)"],["RMSE vs exact","Spearman vs exact","RMSE vs exact"]):
        ax.set(xlabel=xlabel,ylabel=ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(destination / "convergence.png",dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5))
    for point in range(len(exact)):
        group = pd.DataFrame(value_rows).query("seed == 0 and point == @point")
        line, = ax.plot(group.permutations,group.value,label=str(point))
        ax.axhline(exact[point],color=line.get_color(),linestyle="--",alpha=0.4)
    ax.set(xlabel="Permutations (seed 0)",ylabel="Estimated value",title="Dashed lines: exact values")
    ax.legend(title="Point",ncol=4)
    fig.tight_layout()
    fig.savefig(destination / "value_convergence.png",dpi=160)
    plt.close(fig)
    print(frame.groupby("permutations")[["rmse","spearman","wall_seconds","model_fits"]].mean().to_string())
    return exact_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/tmc")
    parser.add_argument("--checkpoints",type=int,nargs="+",default=[10,25,50,100,200])
    parser.add_argument("--split-seed",type=int,default=0)
    parser.add_argument("--tolerance",type=float,default=0.01)
    parser.add_argument("--patience",type=int,default=3)
    args = parser.parse_args()
    benchmark(args.output,args.checkpoints,args.split_seed,args.tolerance,args.patience)
