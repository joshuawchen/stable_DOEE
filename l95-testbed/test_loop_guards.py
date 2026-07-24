#!/usr/bin/env python3
"""Guards for the loop hardening and export machinery added in the
Stage C sessions: every failure mode fixed after being observed in a
run gets a test that fails on the pre-fix behavior.

    python3 test_loop_guards.py

Needs numpy and quadprog only; runs in seconds. Exit 0 on pass.

Covers:
  1. sigma_at_mode grid invariance on an exact Gaussian: the
     left-Riemann reconstruction plus a float-asymmetric fit window
     read 0.348 instead of 0.400 at dx = sigma/10
  2. raw feedback with interior exact zeros (NNLS positivity licenses
     them): nll and score finite everywhere, tails decaying -- log(0)
     poisoned the sampler with nan matmuls and singular solves
  3. export_gap on a deranged spec: finite or nan under
     warnings-as-errors -- exp of an unnormalized log density
     overflowed
  4. defensive-mixture LOO weights: on a thin sampling density the
     undefended weights collapse (measured ESS ~1) and delta = 0.01
     restores them (measured ~190)
  5. smooth_pdf: mass preserved, bin-scale spikes flattened
"""

import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from jedi_export import doee_to_yaml as DY              # noqa: E402
from stage_c_smoothing import (export_gap, loo_hofx,    # noqa: E402
                               raw_nll_from_estimate, smooth_pdf)

warnings.filterwarnings("ignore")
fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


def gauss_cache(div):
    """Exact-Gaussian cache (sigma 0.4) at dx = sigma/div, the
    oracle-cache construction."""
    s = 0.4
    dx = s / div
    n = 2 * int(round(5.0 * s / dx)) + 1
    xg = (np.arange(n) - (n - 1) / 2) * dx
    logf = -0.5 * (xg / s) ** 2
    sl = np.empty_like(logf)
    sl[0] = (logf[1] - logf[0]) / dx
    sl[-1] = (logf[-1] - logf[-2]) / dx
    sl[1:-1] = (logf[2:] - logf[:-2]) / (2 * dx)
    return {"dx": dx, "stable_min": float(xg[0]),
            "stable_max": float(xg[-1]), "slopes_log": sl,
            "left_log_slope": float(sl[0]),
            "left_dd": min(float((sl[1] - sl[0]) / dx), 0.0),
            "right_log_slope": float(sl[-1]),
            "right_dd": min(float((sl[-1] - sl[-2]) / dx), 0.0)}


print("loop and export guards")

# --- 1. sigma_at_mode grid invariance ---------------------------------------
for div in (5, 10, 25):
    spec, _ = DY.to_spec(gauss_cache(div))
    check(f"sigma@mode exact on a Gaussian at dx=s/{div} "
          f"({spec['sigma at mode']:.6f})",
          abs(spec["sigma at mode"] - 0.4) < 1e-6)

# --- 2. raw feedback with interior exact zeros ------------------------------
xg = np.arange(-4.0, 4.0001, 0.02)
pi = np.exp(-0.5 * (xg / 0.4) ** 2)
pi[(xg > 0.3) & (xg < 0.4)] = 0.0
nll_r, dnll_r, _ = raw_nll_from_estimate(xg, pi)
tst = np.arange(-8.0, 8.0001, 0.05)
check("raw feedback finite with interior zeros",
      bool(np.all(np.isfinite(nll_r(tst)))
           and np.all(np.isfinite(dnll_r(tst)))))
check("raw feedback tails decay",
      float(nll_r(np.array([8.0]))[0] - nll_r(np.array([2.0]))[0]) > 0)

# --- 3. export_gap never overflows ------------------------------------------
spec0, _ = DY.to_spec(gauss_cache(10))
spec_bad = {**spec0, "log slopes": [60.0 * s for s in spec0["log slopes"]],
            "left log slope": 60.0 * spec0["left log slope"],
            "right log slope": 60.0 * spec0["right log slope"]}
ok3 = True
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        g = export_gap(spec_bad, xg, np.exp(-0.5 * (xg / 0.4) ** 2),
                       0.4, 0.4)
    ok3 = np.isfinite(g) or np.isnan(g)
except RuntimeWarning:
    ok3 = False
check("export_gap survives a deranged spec (log-space normalization)", ok3)

# --- 4. defensive-mixture LOO weights ---------------------------------------
rngt = np.random.default_rng(3)
n_, kref = 80, 400
r = rngt.normal(0.0, 0.4, (n_, kref))
y_ = np.zeros(n_)
hofx = y_[:, None] - r


def nll_thin(e):
    return 0.5 * (np.asarray(e, float) / 0.08) ** 2


_, ess0 = loo_hofx(y_, hofx, nll_thin, 50, np.random.default_rng(5), 0.0)
_, ess1 = loo_hofx(y_, hofx, nll_thin, 50, np.random.default_rng(5), 0.01)
check(f"undefended weights collapse on a thin density (ESS {ess0:.1f})",
      ess0 < 50)
check(f"defense restores the ESS ({ess0:.1f} -> {ess1:.1f})", ess1 > 100)

# --- 5. smooth_pdf ----------------------------------------------------------
pf = np.zeros_like(xg)
pf[len(xg) // 2] = 1.0 / 0.02
sm = smooth_pdf(pf, 0.02, 0.1)
check("smooth_pdf preserves mass", abs(float(sm.sum()) * 0.02 - 1.0) < 1e-9)
check("smooth_pdf flattens bin-scale spikes",
      float(sm.max()) < 0.2 * float(pf.max()))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
