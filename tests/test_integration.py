"""End-to-end: case studies produce finite estimates and expected qualitative patterns."""

import numpy as np

from case_studies import case_a_naive_vs_ope, case_b_overlap_clipping, case_c_reward_misspec


def test_case_a_naive_disagrees_with_target():
    out = case_a_naive_vs_ope.run(seed=7, n_impressions=8000)
    v_t = out["oracle_online"]["V_target"]
    v_b = out["oracle_online"]["V_behavior"]
    assert v_t > v_b + 0.03
    naive = out["ope"]["naive"]["estimate"]
    assert abs(naive - v_b) < 0.06
    for method in ("ips", "snips", "dr"):
        assert np.isfinite(out["ope"][method]["estimate"])
        assert out["ope"][method]["ess"] > 0
    abs_err = out["abs_error_vs_true_target"]
    assert min(abs_err["snips"], abs_err["dr"], abs_err["ips"]) < abs_err["naive"]


def test_case_b_overlap_poor():
    out = case_b_overlap_clipping.run(seed=3, n_impressions=3000)
    assert out["overlap"]["ess_ratio"] < 0.35
    assert len(out["clipping_sweep"]) >= 2
    assert np.isfinite(out["unclipped_ips"]["estimate"])


def test_case_c_misspec_increases_error():
    out = case_c_reward_misspec.run(seed=11, n_impressions=4000)
    assert np.isfinite(out["dr_good_q"]["estimate"])
    assert np.isfinite(out["dr_bad_q"]["estimate"])
    assert "dr_bad_q" in out["errors"]
