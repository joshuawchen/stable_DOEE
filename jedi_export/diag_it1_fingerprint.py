#!/usr/bin/env python3
"""it1 fingerprint diagnosis (HANDOFF_STAGE_C session-close item 4).

Question: JEDI's fed-back analysis releases tails (fingerprint 0.97);
the mirror's stays near-Gaussian (0.33). Suspect: the it0-exported
spec's tail slopes on this draw.

Three parts, all offline on the real-field snapshot:
  A. Anatomy of the mirror's it0 exported spec: every Format A field,
     the implied density's tail ratio against heavy truth, effective
     tail sigmas from the curvatures.
  B. The JEDI side, reconstructed: pooled moments of the snapshot's
     REAL oman (a genuine fed-back JEDI analysis on this draw) against
     the mirror's it1 departures, plus the native estimator chain run
     directly on oman under both candidate weight densities.
  C. Cross-drive: the mirror's it1 analysis under the ANALYTIC heavy
     spec (true tails, no estimation error). If the mirror releases
     tails here, the flow is exonerated and the fed spec's content is
     the discriminating variable; if not, the flow itself suppresses.
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
from stage_c_smoothing import density_to_spec, loo_hofx     # noqa: E402

NPZ = os.path.join(HERE, "vm_data", "phase3_fields.npz")
FINE = np.arange(-6, 6.0001, 0.01)
PT = menu_density(MENU["heavy"], FINE)
CORE = np.abs(FINE) < 0.3
TAIL = np.abs(FINE) > 0.8


def spec_density(spec, cap=4.8):
    nll = spec_nll(spec, cap)[0]
    ln = -np.asarray(nll(FINE), float)
    ln -= np.nanmax(ln)
    q = np.exp(ln)
    q /= np.trapezoid(q, FINE)
    return q


def fingerprint(p):
    return (float(p[CORE].mean() / PT[CORE].mean()),
            float(p[TAIL].mean() / PT[TAIL].mean()))


def show_spec(tag, spec):
    print(f"  [{tag}] spec fields:")
    for k, v in spec.items():
        if isinstance(v, (list, np.ndarray)):
            v = np.asarray(v, float)
            print(f"    {k:18s} len {v.size}  "
                  f"[{v[0]:+.4f} .. {v[-1]:+.4f}]")
        else:
            print(f"    {k:18s} {v}")
    for side in ("left curvature", "right curvature"):
        dd = float(spec.get(side, np.nan))
        if dd < 0:
            print(f"    -> {side}: tail sigma {np.sqrt(-1.0 / dd):.3f}")
    q = spec_density(spec)
    c, t = fingerprint(q)
    print(f"    -> implied density vs heavy truth: core {c:.2f}  "
          f"tail {t:.2f}")
    return q


def pooled_stats(tag, r):
    r = np.asarray(r, float).ravel()
    m2 = r.var()
    k = float(((r - r.mean()) ** 4).mean() / m2 ** 2) - 3.0
    frac = float((np.abs(r - r.mean()) > 0.8).mean())
    print(f"  [{tag}] pooled: sd {r.std():.3f}  excess kurtosis "
          f"{k:+.3f}  frac|r|>0.8 {frac:.3f}  n {r.size}")


def estimate_on(tag, y, h, nll_w, est, seed):
    h_loo, ess = loo_hofx(y, h, nll_w, h.shape[1],
                          np.random.default_rng(seed), 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec, sd, (xg, pi) = density_to_spec(y, h_loo, est, seed + 5)
    pe = np.interp(FINE, xg, pi, left=0, right=0)
    l1 = float(np.trapezoid(np.abs(pe - PT), FINE))
    c, t = fingerprint(pe)
    gate = "accepted" if spec is not None else "REFUSED"
    print(f"  [{tag}] estimate: L1 {l1:.3f}  ESS {ess:.0f}  "
          f"core {c:.2f}  tail {t:.2f}  sdhat {sd:.3f}  export {gate}")
    return spec


def main():
    est = SimpleNamespace(lam=None, adaptive=True, assumed_error=0.4,
                          export_gap_max=0.4)

    print("== A. mirror it0/it1 with spec anatomy ==")
    rows = M.run_loop(members=40, iters=2, seed=7, fields=NPZ)
    spec0 = rows[0].spec
    show_spec("mirror it0 export", spec0)
    print(f"  mirror it1 measured: L1 {rows[1].l1:.3f}  "
          f"core {rows[1].core:.2f}  tail {rows[1].tail:.2f}")

    print("\n== B. the JEDI side from the snapshot's real oman ==")
    d = np.load(NPZ)
    y, oman, Xb = d["y"], d["oman"], d["Xb"]
    world = M.FieldWorld(NPZ)
    dep1 = world.y[None, :] - M.hofx(
        M.flow(world, spec_nll(spec0, 4.8)[1], 0.05, 210, 10 ** 9))
    pooled_stats("JEDI oman (fed-back analysis)", oman)
    pooled_stats("mirror it1 departures", dep1)
    pooled_stats("background departures", y[None, :] - Xb)
    h_jedi = (y[None, :] - oman).T
    estimate_on("oman | weights GAUSS0", y, h_jedi,
                spec_nll(M.GAUSS0, 4.8)[0], est, 7)
    estimate_on("oman | weights mirror-it0-spec", y, h_jedi,
                spec_nll(spec0, 4.8)[0], est, 7)

    print("\n== C. cross-drive: mirror it1 under the ANALYTIC heavy "
          "spec ==")
    cache = oracle_cache(MENU["heavy"])
    spec_h, _ = DY.to_spec(cache)
    show_spec("analytic heavy export", spec_h)
    rows_c = M.run_loop(members=40, iters=2, seed=7, fields=NPZ,
                        spec_edit=spec_h, quiet=True)
    print(f"  it1 under analytic heavy: sd {rows_c[1].sd:.3f}  "
          f"L1 {rows_c[1].l1:.3f}  ESS {rows_c[1].ess:.0f}  "
          f"core {rows_c[1].core:.2f}  tail {rows_c[1].tail:.2f}")


if __name__ == "__main__":
    main()
