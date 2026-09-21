import numpy as np
import pytest

from recsys_ope_lab.data.world import PreferenceWorld
from recsys_ope_lab.data.logging_policy import (
    SoftmaxLoggingPolicy,
    EpsilonGreedyLoggingPolicy,
    PopularityPolicy,
    log_bandit_data,
)


@pytest.fixture
def world():
    return PreferenceWorld(n_users=40, n_items=25, n_factors=6, seed=0)


def test_relevance_bounds(world):
    assert world.true_relevance.shape == (40, 25)
    assert np.all(world.true_relevance > 0) and np.all(world.true_relevance < 1)


def test_popularity_is_distribution(world):
    assert pytest.approx(world.item_popularity.sum(), abs=1e-9) == 1.0
    assert np.all(world.item_popularity > 0)


def test_softmax_logging_propensities_sum_to_one(world):
    pol = SoftmaxLoggingPolicy(world.popularity_scores(), temperature=0.5)
    p = pol.probs(0)
    assert pytest.approx(p.sum(), abs=1e-9) == 1.0
    assert np.all(p > 0)


def test_epsilon_greedy_mixture(world):
    pop = PopularityPolicy(world)
    eg = EpsilonGreedyLoggingPolicy(exploit=pop, n_items=world.n_items, epsilon=0.2)
    p = eg.probs(3)
    assert pytest.approx(p.sum(), abs=1e-9) == 1.0
    # More mass on popular items than uniform, but every item > 0
    assert p.min() >= 0.2 / world.n_items - 1e-12


def test_log_bandit_data_shapes(world):
    pol = SoftmaxLoggingPolicy(world.popularity_scores(), temperature=1.0)
    rng = np.random.default_rng(1)
    data = log_bandit_data(world, pol, n_impressions=200, rng=rng)
    assert len(data) == 200
    assert data.features.shape[0] == 200
    assert np.all(data.propensities > 0)
    assert set(np.unique(data.rewards)).issubset({0.0, 1.0})


def test_policy_value_between_zero_one(world):
    pol = SoftmaxLoggingPolicy(world.oracle_scores(), temperature=1.0)
    v = world.policy_value(pol)
    assert 0.0 < v < 1.0
