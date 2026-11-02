import numpy as np
from exact_shapley import shapley_from_utilities
from stratified_shapley import stratified_shapley


def test_every_point_and_size_sampled_with_correct_cost():
    calls=[]
    contributions=np.array([0.2,-0.1,0.7,0.3])
    def utility(mask):
        calls.append(mask)
        return sum(v for i,v in enumerate(contributions) if mask & (1<<i))
    for snapshot in stratified_shapley(utility,4,[1,3],seed=7):
        np.testing.assert_array_equal(snapshot["stratum_counts"],np.full((4,4),snapshot["rounds"]))
        np.testing.assert_allclose(snapshot["values"],contributions)
        assert len(calls)==2*4*4*snapshot["rounds"]


def test_two_point_interaction_is_exact_and_seed_reproducible():
    utilities=[0,0.2,0.3,1.0]
    snapshot=list(stratified_shapley(utilities.__getitem__,2,[1],seed=0))[-1]
    np.testing.assert_allclose(snapshot["values"],shapley_from_utilities(utilities,2))
    utility=lambda mask: float(mask.bit_count()>=2)
    first=list(stratified_shapley(utility,4,[2],seed=9))[-1]
    second=list(stratified_shapley(utility,4,[2],seed=9))[-1]
    np.testing.assert_array_equal(first["values"],second["values"])
