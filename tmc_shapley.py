"""Permutation Monte Carlo with explicit, optional tail truncation."""
import time
import numpy as np


def tmc_shapley(utility, n, checkpoints=(10, 25, 50, 100, 200), seed=0,
                tolerance=0.01, patience=3):
    checkpoints = sorted(set(checkpoints))
    if not checkpoints or checkpoints[0] < 1 or patience < 1 or tolerance < 0:
        raise ValueError("Positive checkpoints/patience and nonnegative tolerance required")
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    empty = utility(0)
    full = utility((1 << n) - 1)
    elapsed = time.perf_counter() - started
    total = np.zeros(n)
    truncated = 0
    for iteration in range(1, checkpoints[-1] + 1):
        started = time.perf_counter()
        mask, previous, consecutive = 0, empty, 0
        for position, point in enumerate(rng.permutation(n)):
            mask |= 1 << int(point)
            current = utility(mask)
            total[point] += current - previous
            previous = current
            consecutive = consecutive + 1 if abs(current - full) <= tolerance else 0
            # tolerance=0 disables truncation, giving ordinary permutation MC.
            if tolerance > 0 and consecutive >= patience and position < n - 1:
                truncated += 1
                break
        elapsed += time.perf_counter() - started
        if iteration in checkpoints:
            yield dict(permutations=iteration, values=total.copy() / iteration,
                       wall_seconds=elapsed, truncated_permutations=truncated)
