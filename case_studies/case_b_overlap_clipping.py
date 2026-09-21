"""Case B — Poor overlap → extreme propensities → high IPS variance; clipping tradeoff.

We deliberately use a very peaked logging policy (low temperature) and a target
that prefers the long tail (oracle relevance). Importance weights explode;
ESS collapses; unclipped IPS has huge SE. Clipping buys variance reduction at
the cost of bias — the sweep makes the tradeoff visible.
"""

from __future__ import annotations

import numpy as np

from recsys_ope_lab.data.world import PreferenceWorld
from recsys_ope_lab.data.logging_policy import SoftmaxLoggingPolicy, log_bandit_data
from recsys_ope_lab.ope.estimators import (
    target_propensities_for_logged_actions,
    ips,
    snips,
)
from recsys_ope_lab.ope.diagnostics import overlap_report, clipping_sensitivity
from recsys_ope_lab.ope.reward_model import RewardModel


def run(seed: int = 42, n_impressions: int = 6000) -> dict:
    world = PreferenceWorld(
        n_users=250,
        n_items=100,
        n_factors=10,
        popularity_alpha=1.8,
        quality_popularity_corr=0.1,  # quality almost independent of popularity
        seed=seed,
    )
    rng = np.random.default_rng(seed)

    # Extremely peaked popularity logger
    behavior = SoftmaxLoggingPolicy(world.popularity_scores(), temperature=0.08)
    data = log_bandit_data(world, behavior, n_impressions=n_impressions, rng=rng)

    # Target: softmax on true relevance (wants the long tail the logger never shows)
    target = SoftmaxLoggingPolicy(world.oracle_scores(), temperature=0.25)

    v_behavior = world.policy_value(behavior)
    v_target = world.policy_value(target)

    pi_e = target_propensities_for_logged_actions(target, data.user_ids, data.item_ids)
    report = overlap_report(data.propensities, pi_e, ess_warn_ratio=0.15)

    rm = RewardModel(kind="logistic", seed=seed)
    rm.fit(data)
    q_logged = rm.predict_pairs(world, data.user_ids, data.item_ids)
    q_target = rm.predict_policy_value_per_context(world, data.user_ids, target)

    sweep = clipping_sensitivity(
        data.rewards,
        data.propensities,
        pi_e,
        clips=[None, 5.0, 10.0, 20.0, 50.0, 100.0],
        q_hat_logged=q_logged,
        q_hat_target=q_target,
    )

    unclipped = ips(data.rewards, data.propensities, pi_e, clip=None)
    clipped = ips(data.rewards, data.propensities, pi_e, clip=10.0)
    snips_u = snips(data.rewards, data.propensities, pi_e, clip=None)

    return {
        "oracle_online": {
            "V_behavior": v_behavior,
            "V_target": v_target,
            "gap": v_target - v_behavior,
        },
        "overlap": report,
        "unclipped_ips": unclipped.as_dict(),
        "clipped_ips_10": clipped.as_dict(),
        "unclipped_snips": snips_u.as_dict(),
        "clipping_sweep": sweep,
        "takeaway": (
            "When ESS/n is tiny, do not trust unclipped IPS point estimates. "
            "Report ESS, sweep clips, and prefer SNIPS/DR with explicit bias discussion."
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, default=str))
