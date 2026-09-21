# recsys-ope-lab

**Recommender systems + off-policy evaluation (OPE) lab** — a portfolio project aimed at the Google recsys / data-science hiring bar: logging policies, selection bias, IPS / SNIPS / Doubly Robust, propensity diagnostics, and offline→online honesty.

This is a **working simulation lab**, not a tutorial notebook. Synthetic user–item interactions are generated under an explicit logging policy with known propensities; candidate retrieval and ranking policies are trained offline; OPE estimators evaluate a *new* policy without deploying it; diagnostics show when those estimates are trustworthy.

---

## Problem

Production recommenders almost never train or evaluate on the target traffic distribution. Logs are collected under a **behavior / logging policy** π_b (epsilon-greedy, softmax-on-popularity, existing ranker, etc.). If you train a new policy π_e and then “evaluate offline” by replaying logged rewards naively, you are estimating **V(π_b)**, not **V(π_e)**. That gap is how teams ship regressions that look fine in dashboards.

OPE asks: given logged tuples `(x, a, r, π_b(a|x))`, estimate the value of π_e under identification assumptions — without an online A/B test (yet).

---

## What this repo contains

| Module | Role |
|--------|------|
| `recsys_ope_lab.data` | Latent-factor preference world, Zipf popularity skew, softmax / ε-greedy logging with recorded propensities |
| `recsys_ope_lab.retrieval` | Popularity baseline + matrix-factorization / two-tower scorer (numpy SGD) |
| `recsys_ope_lab.ranking` | Pointwise logistic and pairwise RankNet-style rankers on pair features |
| `recsys_ope_lab.ope` | IPS, SNIPS, Doubly Robust; reward model; ESS / overlap / clipping sweeps |
| `case_studies/` | Three staff-readable experiments (naive vs OPE, overlap/clipping, reward misspec) |

---

## Identification assumptions (read these)

OPE is not free. The estimators here assume:

1. **Unconfoundedness / logged propensities are correct** — π_b(a|x) is known (or consistently estimated). We *simulate* known propensities; in prod you must log them or fit a propensity model.
2. **Common support (overlap)** — π_e(a|x) > 0 ⇒ π_b(a|x) > 0. If the candidate policy wants actions the logger never explores, no amount of clever estimation recovers the truth.
3. **Stable unit / i.i.d. (or correct dependence model)** — our bandit sim is i.i.d. impressions. Real recommenders have slate effects, position bias, and user feedback loops.
4. **For DR** — either propensities **or** the reward model q̂ is correctly specified (double robustness). Misspecify both and DR can be worse than SNIPS.

When overlap is poor, **variance explodes**. Clipping importance weights reduces variance and introduces bias. The lab surfaces that tradeoff instead of hiding it.

---

## Threats to validity (what can still go wrong)

- **Propensity misspecification** — if π_b is estimated poorly, IPS/SNIPS/DR inherit that bias.
- **Slate / ranking OPE ≠ bandit OPE** — we use a single-action (top recommendation) bandit abstraction. Full slate IPS requires propensities over permutations or a factorization assumption.
- **Feedback loops** — training on OPE-reweighted data and redeploying changes π_b; one-shot OPE does not model that.
- **Reward definition** — clicks ≠ long-term value; proxy rewards can invert policy rankings.
- **Simulation optimism** — synthetic latent factors and known propensities are cleaner than production logs.

---

## What this is NOT

- Not a production ranking stack (no serving, no feature store, no slate UI).
- Not a claim that OPE replaces online experiments — it **prioritizes** which policies deserve expensive A/B tests.
- Not inverse-propensity learning of an optimal policy (OPRL / bandit learning); evaluation only.
- Not a fairness / multi-stakeholder allocator (see companion labs for exposure fairness).

---

## Quickstart

```bash
cd recsys-ope-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run case studies (from repo root so `case_studies` imports resolve):

```bash
PYTHONPATH=src:. python case_studies/case_a_naive_vs_ope.py
PYTHONPATH=src:. python case_studies/case_b_overlap_clipping.py
PYTHONPATH=src:. python case_studies/case_c_reward_misspec.py
# or
PYTHONPATH=src:. python -m recsys_ope_lab.cli all
```

---

## Case studies

### A — Naive replay vs IPS / SNIPS / DR

Logging policy: **ε-greedy on popularity** (propensity floor ⇒ overlap). Target: softmax over **true relevance**. Online oracle shows V(π_e) > V(π_b); naive mean logged reward stays near V(π_b). IPS / SNIPS / DR reweight and cut absolute error vs the online truth (MF is still trained in-pipeline as the retrieval stage).

### B — Poor overlap & clipping

Very peaked logger + target that prefers the long tail ⇒ extreme importance weights, low ESS. Clipping sweep shows bias–variance: unclipped IPS SEs are unusable; aggressive clips stabilize estimates but pull them toward the logger.

### C — Reward-model misspecification

DR with a decent logistic q̂ vs a constant mean q̂. Double robustness fails gracefully only when *one* of the two models is right; under mediocre overlap, a bad q̂ can make DR lose to SNIPS.

---

## Estimators (formulas)

Importance weight: `w = π_e(a|x) / π_b(a|x)`

| Estimator | Formula | Notes |
|-----------|---------|-------|
| **IPS** | `mean(w · r)` | Unbiased if propensities correct + overlap; high variance |
| **SNIPS** | `sum(w·r) / sum(w)` | Self-normalized; lower variance, small bias |
| **DR** | `mean( q̂(x,π_e) + w·(r − q̂(x,a)) )` | Consistent if π **or** q̂ is correct |
| **Naive** | `mean(r)` | Estimates logging policy — included as the lie to avoid |

95% CIs use the normal approximation on per-sample contributions (influence-function form for SNIPS). Treat them as diagnostics, not gospel, under heavy tails.

---

## Design choices that signal hiring-bar taste

- **Propensities are first-class** — logged and threaded into every estimator; no silent uniform assumption.
- **Diagnostics before decisions** — ESS, weight quantiles, clipping sweeps, explicit warnings.
- **Case studies show failure modes**, not only happy paths (misspecified DR, vanity naive metrics).
- **Lean stack** — numpy / scipy / scikit-learn only; MF is readable SGD, not a 2k-line framework wrap.

---

## License

MIT © 2026 Rohith Vairavel
