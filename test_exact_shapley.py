import numpy as np
from exact_shapley import shapley_from_utilities


def test_additive_game():
    contributions = np.array([0.2, -0.1, 0.7, 0.3])
    utilities = [sum(contributions[i] for i in range(4) if mask & (1 << i)) for mask in range(16)]
    np.testing.assert_allclose(shapley_from_utilities(utilities, 4), contributions)


def test_symmetric_game_and_efficiency():
    utilities = [float(mask.bit_count() >= 2) for mask in range(8)]
    values = shapley_from_utilities(utilities, 3)
    np.testing.assert_allclose(values, np.full(3, 1 / 3))
    np.testing.assert_allclose(values.sum(), utilities[-1] - utilities[0])
