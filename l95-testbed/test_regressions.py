#!/usr/bin/env python3
"""Regression tests for the estimator failure modes found on the l95 testbed.

    python3 test_regressions.py

Needs numpy and quadprog only. Runs in under a minute (lambda is fixed at the
calibrated default 1e-1, so no cross-validation; fold logic is tested
separately). Exit 0 on pass.

Covers:
  1. over-dispersed kernel: the solve fragments, the mass-based trim must
     still keep the bulk (the longest-run trim kept a 2.7%-mass tail ramp)
  2. the shift gauge: a shifted noise density is recovered under an unbiased
     ensemble; an ensemble bias shifts the recovery by the same amount
  3. grouped CV folds never split an observation's samples
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_doee_reg as R  # noqa: E402

warnings.filterwarnings("ignore")
fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


def moments(xg, pi):
    dx = xg[1] - xg[0]
    p = np.maximum(pi, 0.0)
    p = p / (p.sum() * dx)
    mu = float((p * xg).sum() * dx)
    return mu, float(np.sqrt((p * (xg - mu) ** 2).sum() * dx))


def estimate(obs, hofx):
    grid, f_d, f_k, _ = R.histograms_from_ensemble(obs, hofx, seed=9)
    return R.estimate_from_histograms(grid, f_d, f_k, lam=1e-1)


print("estimator regressions")
rng = np.random.default_rng(7)

# --- 1. over-dispersion: kept mass and location -----------------------------
n, K = 1200, 100
field = 4.0 * (np.sin(np.arange(n) / 37.0) + 0.5 * np.cos(np.arange(n) / 11.0))
pick = rng.random(n) < 0.85
eps = np.where(pick, rng.normal(0, 0.5, n), rng.normal(0, 2.0, n))

for tag, tsd, msd, mass_min in (("reliable kernel", 0.55, 0.55, 0.95),
                                ("over-dispersed x1.24", 0.54, 0.67, 0.80),
                                ("over-dispersed x1.5", 0.50, 0.75, 0.80)):
    truth = field + rng.normal(0, tsd, n)
    members = field[:, None] + rng.normal(0, msd, (n, K))
    xg, pi, cache = estimate(truth + eps, members)
    argmax = float(xg[int(np.argmax(pi))])
    check(f"{tag}: kept mass > {mass_min:.0%}", cache["kept_mass"] > mass_min)
    check(f"{tag}: argmax within 0.3 of zero", abs(argmax) < 0.3)

# --- 2. the shift gauge -----------------------------------------------------
n, K, sb = 2000, 40, 0.5
field = 4.0 * np.sin(np.arange(n) / 37.0)
truth = field + rng.normal(0, sb, n)
members = field[:, None] + rng.normal(0, sb, (n, K))


def heavy(shift):
    pick = rng.random(n) < 0.85
    return np.where(pick, rng.normal(0, 0.25, n), rng.normal(0, 1.0, n)) + shift


xg, pi, _ = estimate(truth + heavy(0.6), members)
mu, _ = moments(xg, pi)
check("shifted noise recovered under an unbiased ensemble",
      abs(mu - 0.6) < 0.1)

xg, pi, _ = estimate(truth + heavy(0.0), members + 0.6)
mu, _ = moments(xg, pi)
check("ensemble bias shifts the recovery by -bias (the gauge)",
      abs(mu + 0.6) < 0.1)

# --- 3. grouped folds keep each observation whole ---------------------------
n_obs, n_mem, folds = 500, 8, 4
groups = R.innovation_groups(n_obs, n_mem)
rng2 = np.random.default_rng(0)
uniq = rng2.permutation(np.unique(groups))
parts = [np.where(np.isin(groups, gp))[0]
         for gp in np.array_split(uniq, folds)]
covered = np.sort(np.concatenate(parts))
ok = covered.size == groups.size and np.array_equal(covered,
                                                    np.arange(groups.size))
for p in parts:
    ok = ok and len(np.unique(groups[p])) * n_mem == len(p)
check("grouped folds partition all samples and never split an observation",
      ok)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
