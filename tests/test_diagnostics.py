import numpy as np

from recsys_ope_lab.ope.diagnostics import (
    effective_sample_size,
    propensity_summary,
    overlap_report,
    clipping_sensitivity,
)


def test_ess_uniform_weights():
    w = np.ones(100)
    assert abs(effective_sample_size(w) - 100) < 1e-9


def test_ess_one_dominant_weight():
    w = np.array([1000.0, 1.0, 1.0, 1.0])
    ess = effective_sample_size(w)
    assert ess < 5


def test_overlap_warns_on_extreme_weights():
    b = np.array([1e-8, 0.5, 0.5])
    e = np.array([1.0, 0.1, 0.1])
    report = overlap_report(b, e)
    assert report["ok"] is False
    assert len(report["warnings"]) >= 1


def test_clipping_sensitivity_runs():
    rng = np.random.default_rng(0)
    n = 200
    r = rng.binomial(1, 0.2, size=n).astype(float)
    b = rng.uniform(0.05, 0.5, size=n)
    e = rng.uniform(0.05, 0.5, size=n)
    rows = clipping_sensitivity(r, b, e, clips=[None, 10.0])
    assert len(rows) == 2
    assert "ips_estimate" in rows[0]
