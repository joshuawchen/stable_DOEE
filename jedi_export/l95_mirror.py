#!/usr/bin/env python3
"""Full-loop offline twin of the Phase 3 JEDI cycle: real L96 dynamics,
the exact ErrorCovarianceL95 B, the three-time obs geometry, the
verbatim conformant PFF as the analysis, and the NATIVE estimator chain
(loo_hofx + density_to_spec) -- the analysis is COMPUTED, not modeled.

Two worlds: a synthesized one (World) and the REAL-FIELD one
(FieldWorld, from extract_mirror_data.py's npz: exact member
backgrounds, exact truth-at-obs-times, exact injected errors --
calibration by construction). No conclusion is drawn from an
uncalibrated configuration.

    python3 l95_mirror.py --fields jedi_export/vm_data/phase3_fields.npz \
        --members 40 --iters 2
"""

import argparse
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "l95-testbed"))
sys.path.insert(0, os.path.join(HERE, ".."))

from make_parity_fixtures import density as menu_density, MENU  # noqa
from map_reference import spec_nll                              # noqa
from stage_c_smoothing import density_to_spec, loo_hofx         # noqa

D = 40
DT = 0.0125
F = 8.0
GAUSS0 = {"mode": 0.0, "grid spacing": 2e-6, "stable min": -1e-6,
          "stable max": 1e-6, "log slopes": [0.0],
          "left log slope": 6.25e-6, "left curvature": -6.25,
          "right log slope": -6.25e-6, "right curvature": -6.25,
          "sigma at mode": 0.4, "mode window": 0.0}


def l96_tend(x):
    return ((np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F)


def l96_step(x, nsteps):
    for _ in range(nsteps):
        k1 = l96_tend(x)
        k2 = l96_tend(x + 0.5 * DT * k1)
        k3 = l96_tend(x + 0.5 * DT * k2)
        k4 = l96_tend(x + DT * k3)
        x = x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def make_B(sigb=0.6):
    idx = np.arange(D)
    dist = np.minimum(np.abs(idx[:, None] - idx[None, :]),
                      D - np.abs(idx[:, None] - idx[None, :]))
    C = np.exp(-0.5 * dist ** 2)
    lam, U = np.linalg.eigh(C)
    Ch = U @ np.diag(np.sqrt(np.maximum(lam, 0))) @ U.T
    return sigb * sigb * C, sigb * Ch


def draw_heavy(n, rng, scale=1.0):
    pick = rng.random(n) < 0.85
    return np.where(pick, rng.normal(0, 0.25 * scale, n),
                    rng.normal(0, 1.0 * scale, n))


class FieldWorld:
    """World built from the REAL JEDI fields."""

    def __init__(self, npz, density="heavy"):
        d = np.load(npz)
        self.Xb = d["Xb"][:, :D].copy()          # states (triplicated)
        self.htruth = d["htruth"].copy()
        self.y = d["y"].copy()
        self.r_true = self.y - self.htruth
        self.B, self.Bh = make_B()
        self.density = density

    def new_obs(self, seed):
        rng = np.random.default_rng(seed + 12345)
        self.r_true = draw_heavy(3 * D, rng)
        self.y = self.htruth + self.r_true


class World:
    """Synthesized truth/backgrounds (NOT gate-calibrated; kept for
    forecast-step experiments the field snapshot cannot vary)."""

    def __init__(self, members, seed, an_err=0.55, density="heavy"):
        rng = np.random.default_rng(seed)
        self.B, self.Bh = make_B()
        x = l96_step(rng.standard_normal(D), 2000)
        self.truth_obs = [l96_step(x, 15)]
        self.truth_obs.append(l96_step(self.truth_obs[0], 1))
        self.truth_obs.append(l96_step(self.truth_obs[1], 1))
        self.htruth = np.concatenate(self.truth_obs)
        xan0 = x + an_err * (self.Bh @ rng.standard_normal(D)) / 0.6
        self.Xb = np.empty((members, D))
        for k in range(members):
            ic = xan0 + self.Bh @ rng.standard_normal(D)
            self.Xb[k] = l96_step(ic, 16)
        self.density = density
        self.new_obs(seed)

    def new_obs(self, seed):
        rng = np.random.default_rng(seed + 12345)
        self.r_true = draw_heavy(3 * D, rng)
        self.y = self.htruth + self.r_true


def hofx(X):
    return np.tile(X, (1, 3)) if X.ndim == 2 else np.tile(X, 3)


def flow(world, dnll, eps0, T, ctcheck, bandwidth_sd=0.6):
    """The verbatim conformant PFF on the world's backgrounds."""
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
    for it in range(T):
        r = y[None, :] - hofx(X)
        g = dnll(r.ravel()).reshape(N, 3 * D)
        rr = g.reshape(N, 3, D).sum(1)
        Fm = np.empty_like(X)
        for i in range(N):
            Kc = np.exp(-(X[i][None] - X) ** 2 / (2 * h2))
            t = Kc * ((B @ rr.T).T - (X - xbar[None]))
            rep = (X[i][None] - X) * Kc / alpha
            rep[i] = 0.0
            Fm[i] = (t + rep).sum(0) / N
        nrm = np.sqrt((Fm.sum(0) ** 2).sum() / (N * D))
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
    return X


def cycle_metrics(world, Xa, spec, est, seed, fine, pt):
    y = world.y
    dep = y[None, :] - hofx(Xa)
    nll = spec_nll(spec, 4.8)[0]
    h_loo, ess = loo_hofx(y, (y[None, :] - dep).T, nll,
                          Xa.shape[0], np.random.default_rng(seed),
                          0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_spec, sd, (xg, pi) = density_to_spec(y, h_loo, est,
                                                 seed + 5)
    pe = np.interp(fine, xg, pi, left=0, right=0)
    l1 = float(np.trapezoid(np.abs(pe - pt), fine))
    core = np.abs(fine) < 0.3
    tail = np.abs(fine) > 0.8
    return SimpleNamespace(
        sd=float(dep.std()), ess=float(ess), l1=l1,
        core=float(pe[core].mean() / pt[core].mean()),
        tail=float(pe[tail].mean() / pt[tail].mean()),
        spec=new_spec, sdhat=sd)


def run_loop(members=40, iters=2, seed=7, eps=0.05, outer=210,
             ctcheck=10 ** 9, refresh=False, an_err=0.55, quiet=False,
             fields=None, bandwidth_sd=0.6, spec_edit=None):
    est = SimpleNamespace(lam=None, adaptive=True, assumed_error=0.4,
                          export_gap_max=0.4)
    fine = np.arange(-6, 6.0001, 0.01)
    pt = menu_density(MENU["heavy"], fine)
    world = (FieldWorld(fields) if fields
             else World(members, seed, an_err=an_err))
    if world.Xb.shape[0] < members:
        raise SystemExit("snapshot has fewer members than requested")
    world.Xb = world.Xb[:members]
    ombg = float((world.y[None, :] - hofx(world.Xb)).std())
    if not quiet:
        print(f"  ombg sd {ombg:.3f}")
    spec = dict(GAUSS0)
    rows = []
    for it in range(iters):
        if refresh and it > 0:
            world.new_obs(seed + 1000 * it)
        use = dict(spec)
        if spec_edit and it > 0:
            use.update(spec_edit)
        dnll = spec_nll(use, 4.8)[1]
        Xa = flow(world, dnll, eps, outer, ctcheck, bandwidth_sd)
        m = cycle_metrics(world, Xa, use, est, seed + it, fine, pt)
        m.ombg = ombg
        rows.append(m)
        if not quiet:
            gate = "accepted" if m.spec is not None else "REFUSED"
            print(f"  it {it}: sd {m.sd:.3f}  L1 {m.l1:.3f}  "
                  f"ESS {m.ess:.0f}  core {m.core:.2f}  "
                  f"tail {m.tail:.2f}  export {gate}")
        if m.spec is not None:
            spec = m.spec
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=40)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--outer", type=int, default=210)
    ap.add_argument("--ctcheck", type=int, default=10 ** 9)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--an-err", type=float, default=0.55)
    ap.add_argument("--fields", default=None)
    ap.add_argument("--bandwidth-sd", type=float, default=0.6)
    a = ap.parse_args()
    run_loop(a.members, a.iters, a.seed, a.eps, a.outer, a.ctcheck,
             a.refresh, a.an_err, fields=a.fields,
             bandwidth_sd=a.bandwidth_sd)


if __name__ == "__main__":
    main()
