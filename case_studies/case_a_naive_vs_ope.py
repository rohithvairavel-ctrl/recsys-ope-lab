"""Case A — Naive replay lies; IPS/SNIPS/DR recover a better ranking policy.

Logging policy: ε-greedy on popularity (guarantees a propensity floor).
Target policy: softmax over true relevance (oracle scores) — clearly better
online, but its actions are only partially present in the popularity-heavy log.

True online value V(π_e) > V(π_b). Naive mean logged reward ≈ V(π_b), so
"replaying" the log as if it evaluated the new ranker is a lie. IPS / SNIPS /
DR reweight toward π_e and move closer to the oracle online value.
"""

from __future__ import annotations

import numpy as np

from recsys_ope_lab.data.world import PreferenceWorld
from recsys_ope_lab.data.logging_policy import (
    SoftmaxLoggingPolicy,
    EpsilonGreedyLoggingPolicy,
    PopularityPolicy,
    log_bandit_data,
)
from recsys_ope_lab.retrieval.two_tower import MatrixFactorizationScorer, MFConfig
from recsys_ope_lab.ope.estimators import (
    evaluate_policy_ope,
    target_propensities_for_logged_actions,
)
from recsys_ope_lab.ope.reward_model import RewardModel


def run(seed: int = 42, n_impressions: int = 12000) -> dict:
    world = PreferenceWorld(
        n_users=250,
        n_items=60,
        n_factors=10,
        popularity_alpha=1.4,
        quality_popularity_corr=0.15,
        seed=seed,
    )
    rng = np.random.default_rng(seed)

    # Behavior: ε-greedy on popularity — head-heavy but every item has π ≥ ε/|A|
    pop = PopularityPolicy(world)
    behavior = EpsilonGreedyLoggingPolicy(exploit=pop, n_items=world.n_items, epsilon=0.25)
    data = log_bandit_data(world, behavior, n_impressions=n_impressions, rng=rng)

    # Also fit MF as a "learned candidate" story; use oracle-softmax as target
    # so the online lift is unambiguous for the case study.
    mf = MatrixFactorizationScorer(
        world.n_users,
        world.n_items,
        MFConfig(n_factors=8, n_epochs=12, seed=seed),
    )
    mf.fit(data)
    _ = mf  # trained to show the retrieval stage is real; target below is oracle

    target = SoftmaxLoggingPolicy(world.oracle_scores(), temperature=0.7)

    v_behavior = world.policy_value(behavior)
    v_target = world.policy_value(target)
    v_oracle = world.greedy_policy_value(world.oracle_scores())

    pi_e = target_propensities_for_logged_actions(target, data.user_ids, data.item_ids)

    n = len(data)
    idx = rng.permutation(n)
    fit_idx, eval_idx = idx[: n // 2], idx[n // 2 :]
    rm = RewardModel(kind="logistic", seed=seed)
    rm.fit(data.subset(fit_idx))
    eval_data = data.subset(eval_idx)
    pi_e_eval = pi_e[eval_idx]
    q_logged = rm.predict_pairs(world, eval_data.user_ids, eval_data.item_ids)
    q_target = rm.predict_policy_value_per_context(world, eval_data.user_ids, target)

    results = evaluate_policy_ope(
        eval_data.rewards,
        eval_data.propensities,
        pi_e_eval,
        q_hat_logged=q_logged,
        q_hat_target=q_target,
        clip=15.0,
    )

    abs_err = {
        "naive": abs(results["naive"].estimate - v_target),
        "ips": abs(results["ips"].estimate - v_target),
        "snips": abs(results["snips"].estimate - v_target),
        "dr": abs(results["dr"].estimate - v_target),
    }

    return {
        "oracle_online": {
            "V_behavior": v_behavior,
            "V_target": v_target,
            "V_oracle_greedy": v_oracle,
            "lift_target_vs_behavior": v_target - v_behavior,
        },
        "ope": {k: v.as_dict() for k, v in results.items()},
        "abs_error_vs_true_target": abs_err,
        "disagreement": {
            "naive_minus_true_target": results["naive"].estimate - v_target,
            "ips_minus_true_target": results["ips"].estimate - v_target,
            "snips_minus_true_target": results["snips"].estimate - v_target,
            "dr_minus_true_target": results["dr"].estimate - v_target,
            "naive_minus_true_behavior": results["naive"].estimate - v_behavior,
            "note": (
                "Naive tracks logging-policy reward (≈ V_behavior), not the candidate. "
                "With ε-greedy overlap, SNIPS/DR should beat naive on |error| vs V_target."
            ),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
