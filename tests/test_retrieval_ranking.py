import numpy as np

from recsys_ope_lab.data.world import PreferenceWorld
from recsys_ope_lab.data.logging_policy import SoftmaxLoggingPolicy, log_bandit_data
from recsys_ope_lab.retrieval.popularity import PopularityRetriever
from recsys_ope_lab.retrieval.two_tower import MatrixFactorizationScorer, MFConfig
from recsys_ope_lab.ranking.pointwise import PointwiseRanker
from recsys_ope_lab.ranking.pairwise import PairwiseRanker


def _small_log(seed=1):
    world = PreferenceWorld(n_users=50, n_items=30, n_factors=6, seed=seed)
    pol = SoftmaxLoggingPolicy(world.popularity_scores(), temperature=0.8)
    data = log_bandit_data(world, pol, n_impressions=800, rng=np.random.default_rng(seed))
    return world, data


def test_popularity_retriever():
    world, data = _small_log()
    ret = PopularityRetriever().fit_from_logs(data, world.n_items)
    top = ret.top_k(5)
    assert len(top) == 5
    sm = ret.score_matrix(world.n_users)
    assert sm.shape == (world.n_users, world.n_items)


def test_mf_fit_and_score():
    world, data = _small_log()
    mf = MatrixFactorizationScorer(
        world.n_users, world.n_items, MFConfig(n_factors=4, n_epochs=3, seed=0)
    )
    mf.fit(data)
    scores = mf.score_matrix()
    assert scores.shape == (world.n_users, world.n_items)
    assert np.isfinite(scores).all()


def test_pointwise_ranker():
    world, data = _small_log()
    ranker = PointwiseRanker(seed=0).fit(data)
    cands = np.arange(10)
    s = ranker.score_candidates(world, 0, cands)
    assert s.shape == (10,)
    assert np.all((s >= 0) & (s <= 1))


def test_pairwise_ranker():
    world, data = _small_log(seed=2)
    ranker = PairwiseRanker(max_pairs=500, seed=0).fit(data, world)
    sm = ranker.build_full_score_matrix(world)
    assert sm.shape == (world.n_users, world.n_items)
