#!/usr/bin/env python3
"""Regression battery for the EXPERIMENTAL adaptive estimator
(estimate_adaptive in stable_doee_reg): whitened data term with a
measured per-bin noise level, resolution-normalized penalty, smoothing
chosen by the discrepancy principle against the parameter-free target
chi2 = nb. Bounds below are set from the measured 2026-07-21 battery
with slack; they pin behavior, they do not certify optimality.

Measured scoreboard against the legacy path on identical data:
  gaussian 2000x20      adaptive better (L1 0.125 vs 0.182)
  heavy-big 1200x100    adaptive matches hand-tuned lam 30 (0.070 vs 0.077)
  over-dispersed x1.24  adaptive recovers (L1 0.18) where legacy fragments
  bimodal (irregular)   adaptive resolves both modes at every n; mu falls
                        with n (the data decides the resolution)
  heavy 2000x20         KNOWN GAP: adaptive ~2x worse (0.27-0.30 vs
                        0.08-0.16 over seeds); the gap case is asserted
                        loosely below so improvement is visible and
                        regression beyond the known level fails.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_doee_reg as R  # noqa: E402

FAILS = []


def check(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


def score(xg, pi, tf):
    dx = xg[1] - xg[0]
    p = np.maximum(pi, 0.0)
    p /= p.sum() * dx
    tru = tf(xg)
    tru /= tru.sum() * dx
    L1 = float(np.abs(p - tru).sum() * dx)
    m1 = (p * xg).sum() * dx
    sd = np.sqrt((p * (xg - m1) ** 2).sum() * dx)
    ku = (p * (xg - m1) ** 4).sum() * dx / sd ** 4 - 3
    pk = p.max()
    nm = int((np.diff(np.r_[0, (p > 0.1 * pk).astype(int), 0]) == 1).sum())
    return L1, sd, ku, nm


def tp_gauss(x):
    return np.exp(-x * x / 0.5) / np.sqrt(0.5 * np.pi)


def tp_heavy(x, s=1.0):
    g = lambda z, sg: np.exp(-z * z / (2 * sg * sg)) / (sg * np.sqrt(2 * np.pi))
    return 0.85 * g(x, 0.25 * s) + 0.15 * g(x, 1.0 * s)


def tp_bi(x):
    g = lambda z, m, sg: np.exp(-(z - m) ** 2 / (2 * sg * sg)) \
        / (sg * np.sqrt(2 * np.pi))
    return 0.6 * g(x, -1.2, 0.35) + 0.4 * g(x, 1.5, 0.5)


def make(seed, n, K, kind, sigma_b=0.5):
    rng = np.random.default_rng(seed)
    f0 = 4.0 * np.sin(np.arange(n) / 37.0)
    if kind == "gauss":
        eps = rng.normal(0, 0.5, n)
    elif kind == "heavy":
        pick = rng.random(n) < 0.85
        eps = np.where(pick, rng.normal(0, 0.25, n), rng.normal(0, 1.0, n))
    elif kind == "heavy2":
        pick = rng.random(n) < 0.85
        eps = np.where(pick, rng.normal(0, 0.5, n), rng.normal(0, 2.0, n))
    elif kind == "bimodal":
        pick = rng.random(n) < 0.6
        eps = np.where(pick, rng.normal(-1.2, 0.35, n),
                       rng.normal(1.5, 0.5, n))
    truth = f0 + rng.normal(0, sigma_b, n)
    mem = f0[:, None] + rng.normal(0, sigma_b, (n, K))
    return truth + eps, mem


# A. Gaussian null: no hallucinated tails, tight recovery
obs, mem = make(7, 2000, 20, "gauss")
xg, pi, c = R.estimate_adaptive_from_ensemble(obs, mem, seed=9)
L1, sd, ku, nm = score(xg, pi, tp_gauss)
check("gaussian null", L1 < 0.20 and abs(ku) < 1.5 and abs(sd - 0.5) < 0.08,
      f"L1 {L1:.3f} sd {sd:.3f} kurt {ku:+.2f}")

# B. Record-class configuration: must match hand-tuned quality
obs, mem = make(11, 1200, 100, "heavy2", sigma_b=0.55)
xg, pi, c = R.estimate_adaptive_from_ensemble(obs, mem, seed=9)
L1, sd, ku, nm = score(xg, pi, lambda x: tp_heavy(x, 2.0))
check("record-class 1200x100", L1 < 0.15 and abs(sd - 0.911) < 0.12
      and 4.0 < ku < 13.0, f"L1 {L1:.3f} sd {sd:.3f} kurt {ku:+.2f}")

# C. Irregular bimodal truth: both modes at moderate n, mu falls with n
mus = []
for n in (2400, 20000):
    obs, mem = make(21 + n, n, 20, "bimodal")
    xg, pi, c = R.estimate_adaptive_from_ensemble(obs, mem, seed=9)
    L1, sd, ku, nm = score(xg, pi, tp_bi)
    mus.append(c["lambda"])
    check(f"bimodal n={n}", nm == 2 and L1 < 0.30
          and abs(sd - 1.386) < 0.15,
          f"modes {nm} L1 {L1:.3f} sd {sd:.3f} mu {c['lambda']:.0e}")
check("adaptivity (mu falls with n)", mus[1] <= mus[0],
      f"mu {mus[0]:.0e} -> {mus[1]:.0e}")

# D. Over-dispersed x1.24: recover, do not fragment
rng = np.random.default_rng(7)
n = 1200
f0 = 4.0 * (np.sin(np.arange(n) / 37.0) + 0.5 * np.cos(np.arange(n) / 11.0))
pick = rng.random(n) < 0.85
eps = np.where(pick, rng.normal(0, 0.5, n), rng.normal(0, 2.0, n))
tr = f0 + rng.normal(0, 0.54, n)
mm = f0[:, None] + rng.normal(0, 0.67, (n, 100))
xg, pi, c = R.estimate_adaptive_from_ensemble(tr + eps, mm, seed=9)
L1, sd, ku, nm = score(xg, pi, lambda x: tp_heavy(x, 2.0))
check("over-dispersed x1.24", L1 < 0.35 and sd > 0.6,
      f"L1 {L1:.3f} sd {sd:.3f} kurt {ku:+.2f}")

# E. KNOWN GAP, pinned: heavy 2000x20 currently ~2x worse than legacy
obs, mem = make(12, 2000, 20, "heavy")
xg, pi, c = R.estimate_adaptive_from_ensemble(obs, mem, seed=9)
L1, sd, ku, nm = score(xg, pi, tp_heavy)
check("heavy 2000x20 (known gap)", L1 < 0.45 and 2.0 < ku < 12.0,
      f"L1 {L1:.3f} (legacy ~0.13) sd {sd:.3f} kurt {ku:+.2f}")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
