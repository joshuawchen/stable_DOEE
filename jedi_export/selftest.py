#!/usr/bin/env python3
"""Self-test for doee_to_yaml: `python3 selftest.py`.

Needs only numpy -- no JEDI, no notebook, no diagnostic files.
"""
import numpy as np
import doee_to_yaml as D

fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


def make(noise=0.0, ragged=(), seed=7, dx=0.25, lo=-5.0, hi=5.0):
    """A skewed unimodal density cache, optionally noisy or with wrong-sign
    bins of the kind an ill-posed deconvolution produces."""
    rng = np.random.default_rng(seed)
    n = int(round((hi - lo) / dx))
    c = lo + (np.arange(n) + 0.5) * dx
    s = -c / (0.30 + 0.25 * np.abs(c) + 0.10 * c ** 2)
    if noise:
        s = s + rng.normal(0, noise, n)
    for i, v in ragged:
        s[i] = v
    return dict(dx=dx, stable_min=lo, stable_max=hi, slopes_log=s,
                left_log_slope=float(s[0]), left_dd=-0.05,
                right_log_slope=float(s[-1]), right_dd=-0.05), c


print("doee_to_yaml self-test")

cache, c = make()
spec, nf = D.to_spec(cache)
check("clean estimate: mode near 0", abs(spec["mode"]) < 0.1)
check("clean estimate: no slopes projected", nf == 0)
check("clean estimate: sigma at mode positive", spec["sigma at mode"] > 0)
check("clean estimate: variance valid everywhere", not D.check(spec))

cache, c = make(noise=0.08, seed=11)
spec, nf = D.to_spec(cache)
check("noisy estimate: mode still near 0", abs(spec["mode"]) < 0.3)
check("noisy estimate: variance valid everywhere", not D.check(spec))

# ragged bins with the WRONG SIGN, as an ill-posed deconvolution can produce
cache, c = make(ragged=[(3, -0.4), (30, +0.3)])
check("mode robust to ragged bins", abs(D.find_mode(cache)) < 0.3)
try:
    D.to_spec(cache, enforce="strict")
    ok = False
except ValueError:
    ok = True
check("strict refuses ragged bins", ok)
spec, nf = D.to_spec(cache, enforce="monotone")
check("monotone repairs exactly the bad bins", nf == 2)
check("repaired density valid everywhere", not D.check(spec))
sl = np.asarray(spec["log slopes"])
cc = D._centers(cache)
check("repaired density is unimodal",
      not ((((cc < spec["mode"]) & (sl < 0))
            | ((cc > spec["mode"]) & (sl > 0))).any()))

# tails that grow without bound must be refused
bad, _ = make()
bad["right_dd"] = +0.5
try:
    D.to_spec(bad)
    ok = False
except ValueError:
    ok = True
check("positive tail curvature refused", ok)

# a Gaussian must come back with the right sigma
dx, lo, hi, s2 = 0.1, -6.0, 6.0, 0.36
n = int(round((hi - lo) / dx))
cg = lo + (np.arange(n) + 0.5) * dx
g = dict(dx=dx, stable_min=lo, stable_max=hi, slopes_log=-cg / s2,
         left_log_slope=-lo / s2, left_dd=-1.0 / s2,
         right_log_slope=-hi / s2, right_dd=-1.0 / s2)
spec, _ = D.to_spec(g)
check("Gaussian: sigma at mode recovers 0.6",
      abs(spec["sigma at mode"] - np.sqrt(s2)) < 0.02)
f = D.Density(spec)
check("Gaussian: effective sigma ~constant away from the mode",
      max(abs(f.variance(d) - s2) for d in (1.0, 2.0, 3.0, -1.5, -3.0)) < 0.05 * s2)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
