import numpy as np
import itertools
from exact_shapley import shapley_from_utilities
from tmc_shapley import tmc_shapley


def test_all_permutations_match_exact_on_nonadditive_game(monkeypatch):
    orders = iter(itertools.permutations(range(3)))
    class Exhaustive:
        def permutation(self, n):
            return np.array(next(orders))
    monkeypatch.setattr(np.random, "default_rng", lambda seed: Exhaustive())
    utilities = [0,0.1,0.3,0.8,0.2,0.4,0.7,1.0]
    estimate = list(tmc_shapley(utilities.__getitem__,3,[6],tolerance=0))[-1]["values"]
    np.testing.assert_allclose(estimate,shapley_from_utilities(utilities,3))


def test_untruncated_additive_game():
    contributions = np.array([0.2,-0.1,0.7])
    def utility(mask):
        return sum(v for i,v in enumerate(contributions) if mask & (1 << i))
    result = list(tmc_shapley(utility,3,[1,5],seed=3,tolerance=0))
    for snapshot in result:
        np.testing.assert_allclose(snapshot["values"],contributions)
        assert snapshot["truncated_permutations"] == 0


def test_truncation_saves_evaluations_and_is_reproducible():
    calls = []
    def utility(mask):
        calls.append(mask)
        return float(mask != 0)
    first = list(tmc_shapley(utility,5,[10],seed=4,tolerance=0.01,patience=1))[-1]
    assert len(calls) == 12  # empty/full reference + one prefix per permutation
    assert first["truncated_permutations"] == 10
    np.testing.assert_allclose(first["values"].sum(),1)
    second = list(tmc_shapley(utility,5,[10],seed=4,tolerance=0.01,patience=1))[-1]
    np.testing.assert_array_equal(first["values"],second["values"])
