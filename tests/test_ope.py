import numpy as np
import pytest

from recsys_ope_lab.ope.estimators import ips, snips, doubly_robust, naive_on_policy


def test_ips_unbiased_when_same_policy():
    rng = np.random.default_rng(0)
    n = 5000
    # π_e = π_b ⇒ w = 1 ⇒ IPS = mean reward
    rewards = rng.binomial(1, 0.3, size=n).astype(float)
    prop = np.full(n, 0.2)
    res = ips(rewards, prop, prop)
    assert abs(res.estimate - rewards.mean()) < 1e-9


def test_snips_equals_ips_when_weights_constant():
    rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    b = np.full(5, 0.25)
    e = np.full(5, 0.25)
    assert abs(ips(rewards, b, e).estimate - snips(rewards, b, e).estimate) < 1e-12


def test_ips_reweights_correctly():
    # Two actions; target puts all mass on action rarely logged
    rewards = np.array([1.0, 0.0, 1.0, 0.0])
    behavior = np.array([0.9, 0.9, 0.1, 0.1])
    target = np.array([0.1, 0.1, 0.9, 0.9])
    res = ips(rewards, behavior, target)
    # contributions: 1*(0.1/0.9), 0, 1*(0.9/0.1), 0 → mean = (0.111... + 9) / 4
    expected = (0.1 / 0.9 + 0.0 + 0.9 / 0.1 + 0.0) / 4
    assert res.estimate == pytest.approx(expected)


def test_dr_recovers_when_q_perfect_even_if_weights_wrong_scale():
    # If q̂ = true reward and we evaluate deterministic same action, DR = q
    rewards = np.array([1.0, 0.0, 1.0])
    b = np.array([0.5, 0.5, 0.5])
    e = np.array([0.5, 0.5, 0.5])
    q_a = rewards.copy()
    q_e = np.array([0.7, 0.7, 0.7])
    res = doubly_robust(rewards, b, e, q_a, q_e)
    # samples = 0.7 + 1*(r - r) = 0.7
    assert res.estimate == pytest.approx(0.7)


def test_clipping_reduces_max_weight_effect():
    rewards = np.array([1.0, 0.0])
    b = np.array([0.001, 0.5])
    e = np.array([1.0, 0.5])
    unclipped = ips(rewards, b, e, clip=None)
    clipped = ips(rewards, b, e, clip=10.0)
    assert clipped.estimate < unclipped.estimate


def test_naive_is_mean():
    r = np.array([1.0, 0.0, 1.0])
    assert naive_on_policy(r).estimate == pytest.approx(2 / 3)
