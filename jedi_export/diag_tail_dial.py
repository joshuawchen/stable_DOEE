#!/usr/bin/env python3
"""Part D of the it1 fingerprint diagnosis: the tail dial in isolation.

The it0 export's interior is held fixed; ONLY the continuation
curvatures (left/right) are relaxed. If departure kurtosis and the
recovered tail rise smoothly with fed tail width while the flow stays
healthy, the fed spec's tail slopes are the controlling variable for
tail release. A flow-health trace (the PFF's own norm, start/min/final)
separates flow instability from estimator collapse -- the JEDI drift
rows were 'flow healthy' by exactly this metric.
"""

import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "l95-testbed"))
sys.path.insert(0, os.path.join(HERE, ".."))

import l95_mirror as M                                      # noqa: E402
from make_parity_fixtures import (density as menu_density,  # noqa: E402
                                  MENU, oracle_cache)
import doee_to_yaml as DY                                   # noqa: E402
from map_reference import spec_nll                          # noqa: E402

NPZ = os.path.join(HERE, "vm_data", "phase3_fields.npz")
FINE = np.arange(-6, 6.0001, 0.01)
PT = menu_density(MENU["heavy"], FINE)


def flow_traced(world, dnll, eps0, T, ctcheck, bandwidth_sd=0.6):
    """M.flow verbatim, plus the collective norm trajectory."""
    X = world.Xb.copy()
    N = X.shape[0]
    xbar = X.mean(0)
    alpha = 1.0 / N
    h2 = alpha * bandwidth_sd * bandwidth_sd
    B = world.B
    eps = np.full(N, eps0)
    ct = np.zeros(N, int)
    n1 = np.full(N, -1.0)
    y = world.y
    trace = []
    for it in range(T):
        r = y[None, :] - M.hofx(X)
        g = dnll(r.ravel()).reshape(N, 3 * M.D)
        rr = g.reshape(N, 3, M.D).sum(1)
        Fm = np.empty_like(X)
        for i in range(N):
            Kc = np.exp(-(X[i][None] - X) ** 2 / (2 * h2))
            t = Kc * ((B @ rr.T).T - (X - xbar[None]))
            rep = (X[i][None] - X) * Kc / alpha
            rep[i] = 0.0
            Fm[i] = (t + rep).sum(0) / N
        nrm = np.sqrt((Fm.sum(0) ** 2).sum() / (N * M.D))
        trace.append(nrm)
        for i in range(N):
            if n1[i] < 0:
                n1[i] = nrm; X[i] += eps[i] * Fm[i]; ct[i] += 1
            elif nrm > 1.02 * n1[i]:
                eps[i] /= 1.5; ct[i] = 0
            elif ct[i] >= ctcheck:
                ct[i] = 0; X[i] += eps[i] * 1.5 * Fm[i]
            else:
                X[i] += eps[i] * Fm[i]; ct[i] += 1
            n1[i] = nrm
    return X, np.array(trace)


def kurt(r):
    r = np.asarray(r, float).ravel()
    return float(((r - r.mean()) ** 4).mean() / r.var() ** 2) - 3.0


def one(tag, spec0, dd_left, dd_right):
    spec = dict(spec0)
    spec["left curvature"] = dd_left
    spec["right curvature"] = dd_right
    world = M.FieldWorld(NPZ)
    # run_loop protocol: FIXED backgrounds every iteration (the JEDI
    # driver's fixed-window template) -- it1 analyzes the original Xb
    # under the edited fed spec
    Xa1, tr = flow_traced(world, spec_nll(spec, 4.8)[1],
                          0.05, 210, 10 ** 9)
    dep = world.y[None, :] - M.hofx(Xa1)
    est = SimpleNamespace(lam=None, adaptive=True, assumed_error=0.4,
                          export_gap_max=0.4)
    from stage_c_smoothing import density_to_spec, loo_hofx
    nll_w = spec_nll(spec, 4.8)[0]
    h_loo, ess = loo_hofx(world.y, (world.y[None, :] - dep).T, nll_w,
                          Xa1.shape[0], np.random.default_rng(8), 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, sd, (xg, pi) = density_to_spec(world.y, h_loo, est, 12)
    pe = np.interp(FINE, xg, pi, left=0, right=0)
    l1 = float(np.trapezoid(np.abs(pe - PT), FINE))
    tail = float(pe[np.abs(FINE) > 0.8].mean()
                 / PT[np.abs(FINE) > 0.8].mean())
    core = float(pe[np.abs(FINE) < 0.3].mean()
                 / PT[np.abs(FINE) < 0.3].mean())
    st = (np.sqrt(-1.0 / dd_left), np.sqrt(-1.0 / dd_right))
    print(f"  {tag:26s} tail-sig {st[0]:.2f}/{st[1]:.2f}  "
          f"dep sd {dep.std():.3f} kurt {kurt(dep):+.2f}  "
          f"L1 {l1:.3f} core {core:.2f} tail {tail:.2f} ESS {ess:.0f}  "
          f"norm 0/min/fin {tr[0]:.1f}/{tr.min():.2f}/{tr[-1]:.2f} "
          f"({100 * tr[-1] / tr[0]:.0f}%)")


def main():
    # reproduce the it0 export once (same path as the baseline run)
    rows = M.run_loop(members=40, iters=1, seed=7, fields=NPZ,
                      quiet=True)
    spec0 = rows[0].spec
    ddl, ddr = spec0["left curvature"], spec0["right curvature"]
    print("fed-spec tail-curvature sweep, interior fixed at the it0 "
          "export:")
    one("baseline (as exported)", spec0, ddl, ddr)
    one("dd -8 (sig 0.35)", spec0, -8.0, -8.0)
    one("dd -2 (sig 0.71)", spec0, -2.0, -2.0)
    one("dd -1 (sig 1.00)", spec0, -1.0, -1.0)
    one("dd -0.5 (sig 1.41)", spec0, -0.5, -0.5)

    print("\nreference: it1 under the ANALYTIC heavy spec, traced:")
    cache = oracle_cache(MENU["heavy"])
    spec_h, _ = DY.to_spec(cache)
    one("analytic heavy", spec_h,
        spec_h["left curvature"], spec_h["right curvature"])


if __name__ == "__main__":
    main()
