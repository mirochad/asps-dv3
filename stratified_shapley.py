"""Balanced coalition-size stratification using pyDVL's sampler.

The aggregation is explicit: equal weight for each size stratum, then mean
over rounds. This uses pyDVL StratifiedSampler, not its convenience valuation
class (which randomly selects sizes/indices instead of balancing each round).
"""
import time

import numpy as np
from pydvl.valuation.samplers.stratified import ConstantSampleSize, StratifiedSampler


def stratified_shapley(utility, n, checkpoints=(1, 2, 4, 8, 16), seed=0):
    checkpoints = sorted(set(checkpoints))
    if n < 1 or not checkpoints or checkpoints[0] < 1:
        raise ValueError("Positive n and round checkpoints required")
    started = time.perf_counter()
    sampler = StratifiedSampler(sample_sizes=ConstantSampleSize(n_samples=n), seed=seed)
    totals = np.zeros((n, n))
    counts = np.zeros((n, n), dtype=int)
    elapsed = time.perf_counter() - started
    for round_number in range(1, checkpoints[-1] + 1):
        started = time.perf_counter()
        # One uniform subset per point i and size k=0,...,n-1 each round.
        for sample in sampler.generate(np.arange(n)):
            point = int(sample.idx)
            size = len(sample.subset)
            mask = sum(1 << int(i) for i in sample.subset)
            before = utility(mask)
            after = utility(mask | (1 << point))
            totals[point, size] += after - before
            counts[point, size] += 1
        elapsed += time.perf_counter() - started
        if round_number in checkpoints:
            yield dict(rounds=round_number, marginal_samples=round_number*n*n,
                       values=(totals/counts).mean(axis=1),
                       stratum_counts=counts.copy(), wall_seconds=elapsed)
