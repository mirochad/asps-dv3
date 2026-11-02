"""Exact, TMC and balanced pyDVL stratified sampling on one fixed game."""
import argparse
import itertools
import json
import resource
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmark_tmc import benchmark, plt
from stratified_shapley import stratified_shapley
from valuation_game import LogisticGame


def compare(output="results/comparison", split_seed=0,
            permutations=(10,25,50,100,200), rounds=(1,2,4,8,16), tolerance=0.01):
    root = Path(output)
    exact_metrics = benchmark(root/"tmc",permutations,split_seed,tolerance)
    exact = pd.read_csv(root/"tmc/exact/values.csv")["value"].to_numpy()
    frame = pd.read_csv(root/"tmc/convergence.csv").rename(columns={"permutations":"budget"})
    frame["method"] = "TMC"
    frame["budget_unit"] = "permutations"
    value_frame = pd.read_csv(root/"tmc/values.csv").rename(columns={"permutations":"budget"})
    value_frame["method"] = "TMC"
    value_frame["budget_unit"] = "permutations"
    stability = pd.read_csv(root/"tmc/stability.csv").rename(columns={"permutations":"budget"})
    stability["method"] = "TMC"
    stability["budget_unit"] = "permutations"
    rows, values, snapshots, counts = [], [], {}, []
    for seed in (0,1,2):
        game = LogisticGame(split_seed)
        for snapshot in stratified_shapley(game,game.n,rounds,seed):
            estimate = snapshot.pop("values")
            coverage = snapshot.pop("stratum_counts")
            budget = snapshot.pop("rounds")
            snapshots[seed,budget] = estimate
            rows.append(dict(method="Stratified",budget_unit="rounds",seed=seed,budget=budget,
                             **snapshot,utility_evaluations=game.calls,model_fits=game.fits,
                             rmse=float(np.sqrt(np.mean((estimate-exact)**2))),
                             spearman=float(spearmanr(estimate,exact).statistic),
                             efficiency_residual=float(estimate.sum()-exact.sum()),
                             peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
            values.extend(dict(method="Stratified",budget_unit="rounds",seed=seed,budget=budget,
                               point=i,value=float(v),exact=float(exact[i])) for i,v in enumerate(estimate))
            counts.extend(dict(seed=seed,rounds=budget,point=i,size=k,count=int(coverage[i,k]))
                          for i in range(game.n) for k in range(game.n))
        print(f"Finished stratified seed {seed}",flush=True)
    pairs = []
    for budget in sorted(set(rounds)):
        for a,b in itertools.combinations((0,1,2),2):
            pairs.append(dict(method="Stratified",budget_unit="rounds",budget=budget,seed_a=a,seed_b=b,
                              spearman=float(spearmanr(snapshots[a,budget],snapshots[b,budget]).statistic)))
    frame = pd.concat([frame,pd.DataFrame(rows)],ignore_index=True)
    frame.to_csv(root/"convergence.csv",index=False)
    pd.concat([value_frame,pd.DataFrame(values)],ignore_index=True).to_csv(root/"values.csv",index=False)
    pd.concat([stability,pd.DataFrame(pairs)],ignore_index=True).to_csv(root/"stability.csv",index=False)
    pd.DataFrame(counts).to_csv(root/"stratum_counts.csv",index=False)
    (root/"config.json").write_text(json.dumps(dict(split_seed=split_seed,estimator_seeds=[0,1,2],
        permutations=list(permutations),rounds=list(rounds),tolerance=tolerance,patience=3,
        stratified_sampler="pyDVL StratifiedSampler / ConstantSampleSize / balanced rounds",caching=False),indent=2))
    # Compare actual effort, not rounds vs permutations (different work units).
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for method, group in frame.groupby("method"):
        mean = group.groupby("budget").mean(numeric_only=True)
        for ax,x,y in zip(axes,["utility_evaluations","wall_seconds","model_fits"],["rmse","rmse","spearman"]):
            ax.plot(mean[x],mean[y],"o-",label=method)
    axes[0].scatter([exact_metrics["coalition_evaluations"]],[0],marker="*",s=100,color="black",label="Exact")
    axes[1].scatter([exact_metrics["wall_seconds"]],[0],marker="*",s=100,color="black",label="Exact")
    axes[2].scatter([exact_metrics["valuation_model_fits"]],[1],marker="*",s=100,color="black",label="Exact")
    for ax,x,y in zip(axes,["Utility evaluations","Valuation time (seconds)","Model fits"],["RMSE vs exact","RMSE vs exact","Spearman vs exact"]):
        ax.set(xlabel=x,ylabel=y)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(root/"quality_cost.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method,group in frame.groupby("method"):
        for seed,run in group.groupby("seed"):
            axes[0].plot(run.utility_evaluations,run.rmse,"o-",label=f"{method} seed {seed}")
    all_stability=pd.concat([stability,pd.DataFrame(pairs)],ignore_index=True)
    effort=frame.groupby(["method","budget"])["utility_evaluations"].mean().rename("utility_evaluations").reset_index()
    all_stability=all_stability.merge(effort,on=["method","budget"],validate="many_to_one")
    for method,group in all_stability.groupby("method"):
        for pair,run in group.groupby(["seed_a","seed_b"]):
            axes[1].plot(run.utility_evaluations,run.spearman,"o-",label=f"{method} {pair}")
    axes[0].set(xlabel="Utility evaluations",ylabel="RMSE vs exact")
    axes[1].set(xlabel="Mean utility evaluations across seeds",ylabel="Between-seed Spearman")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(root/"seed_convergence.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    strat_values=pd.DataFrame(values)
    for point in range(len(exact)):
        group=strat_values.query("seed == 0 and point == @point")
        line,=ax.plot(group.budget,group.value,"o-",label=str(point))
        ax.axhline(exact[point],color=line.get_color(),linestyle="--",alpha=0.4)
    ax.set(xlabel="Balanced stratified rounds (seed 0)",ylabel="Estimated value",title="Dashed lines: exact values")
    ax.legend(title="Point",ncol=4)
    fig.tight_layout()
    fig.savefig(root/"stratified_values.png",dpi=160)
    plt.close(fig)
    print(frame.groupby(["method","budget"])[["rmse","spearman","wall_seconds","model_fits"]].mean().to_string())
    return frame


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/comparison")
    parser.add_argument("--split-seed",type=int,default=0)
    parser.add_argument("--permutations",type=int,nargs="+",default=[10,25,50,100,200])
    parser.add_argument("--rounds",type=int,nargs="+",default=[1,2,4,8,16])
    parser.add_argument("--tolerance",type=float,default=0.01)
    args=parser.parse_args()
    compare(args.output,args.split_seed,args.permutations,args.rounds,args.tolerance)
