#!/usr/bin/env python3
"""MAP reference test: score density error by its effect on the analysis mode.

One 3D window in state space, solved to machine precision three ways that
differ ONLY in the observation-error density inside Jo:

    gaussian   the assumed R (the ObsError value a control run believes)
    estimated  the DOEE density, pushed through the SAME Format A export the
               JEDI branch assimilates (gaussian_tails retraction included)
    true       the injected density, analytic value and score, no export

plus a fourth arm by default,

    true-fmt   the injected density pushed through Format A on the DOEE grid,

whose distance to `true` bounds the representation error of the export, and
whose agreement with the JEDI branch's own true-density run is the plumbing
cross-check.

The problem is deterministic: J(x) = 0.5 (x-x_b)' C^-1 (x-x_b)
- sum_i log pi(y_i - (Hx)_i), a 40-dimensional optimization with an exact
analytic gradient, so the comparison of the arms carries no sampling or
minimizer noise. The prior is exactly specified by construction (the
background error is drawn from C), so the ONLY mis-specification anywhere is
the observation-error density -- the quantity under test.

Verdict metrics, per replicate, all against the true-density MAP x*_true:

    rmse      state-space RMSE of each arm's MAP from x*_true
    regret    (J_true(x*_arm) - J_true(x*_true)) / n_obs, in nats per ob;
              nonnegative by construction, so a negative value is a
              built-in multistart failure detector, not a result

Multistart is on by default (background, truth, perturbed starts) because
heavy or bimodal densities make the posterior potentially multimodal and the
MAP can jump basins; the count of distinct basins found is reported.

Differences from the injector worth knowing: mirrored_gamma is drawn here
with the FIXED population scale (c = 0.4 scale / (2 sqrt 2)), not the
injector's sample-std rescale, so the analytic `true` arm is exact for all
four kinds; and its log density is given a smooth quadratic continuation
below t0 = 1e-3 so line searches remain finite outside the support, which
does not move an interior MAP.

    python3 map_reference.py --density heavy --replicates 8 --lam 30
    python3 map_reference.py --selftest
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import stable_doee_reg as R                      # noqa: E402
from jedi_export import doee_to_yaml as DY       # noqa: E402
from stage_a_end_to_end import (analytic_density, gaussian_tails,  # noqa: E402
                                moments_on, oracle_cache)

NGRID = 40


class Quiet:
    """rep shim for gaussian_tails when only its side effect is wanted."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def info(self, txt):
        if self.verbose:
            print(f"    {txt}")


# ---------------------------------------------------------------------------
# geometry: periodic grid, Gaussian-correlation prior, interpolation operator
# ---------------------------------------------------------------------------

def prior_cov(sd=0.6, length_scale=1.0, n=NGRID):
    d = np.arange(n, dtype=float)
    dist = np.minimum(d, n - d)
    row = sd * sd * np.exp(-0.5 * (dist / length_scale) ** 2)
    C = np.empty((n, n))
    for i in range(n):
        C[i] = np.roll(row, i)
    C = 0.5 * (C + C.T) + 1e-10 * sd * sd * np.eye(n)
    return C


def interp_operator(n_obs, n=NGRID):
    """Linear interpolation on the circle at n_obs equally spaced locations,
    the l95 convention that lets the observation density exceed the grid."""
    locs = np.linspace(0.0, n, n_obs, endpoint=False)
    j0 = np.floor(locs).astype(int) % n
    w = locs - np.floor(locs)
    H = np.zeros((n_obs, n))
    H[np.arange(n_obs), j0] += 1.0 - w
    H[np.arange(n_obs), (j0 + 1) % n] += w
    return H


# ---------------------------------------------------------------------------
# error draws and analytic likelihoods (nll and its epsilon-derivative)
# ---------------------------------------------------------------------------

def draw_errors(kind, rng, n, scale=1.0):
    """The injector's kinds. mirrored_gamma uses the fixed population scale
    so the analytic density below is EXACT for the draws, not approximate."""
    if kind == "gaussian":
        s = 0.4 * scale
        return rng.normal(0.0, s, n), {"kind": "gaussian", "sigma": s}
    if kind == "heavy":
        s1, s2, w = 0.25 * scale, 1.0 * scale, 0.85
        pick = rng.random(n) < w
        e = np.where(pick, rng.normal(0, s1, n), rng.normal(0, s2, n))
        return e, {"kind": "mixture", "w": w, "sigma1": s1, "sigma2": s2}
    if kind == "laplace":
        b = 0.3 * scale
        return rng.laplace(0.0, b, n), {"kind": "laplace", "b": b}
    if kind == "mirrored_gamma":
        c = 0.4 * scale / (2.0 * np.sqrt(2.0))
        g = rng.gamma(shape=2.0, scale=2.0, size=n)
        e = (2.0 - g) * c
        return e, {"kind": "mirrored_gamma", "shape": 2.0, "scale": 2.0,
                   "rescaled_to_sigma": 0.4 * scale}
    raise ValueError(f"unknown density '{kind}'")


def sample_sigma_of(spec):
    k = spec["kind"]
    if k == "gaussian":
        return spec["sigma"]
    if k == "mixture":
        return float(np.sqrt(spec["w"] * spec["sigma1"] ** 2
                             + (1 - spec["w"]) * spec["sigma2"] ** 2))
    if k == "laplace":
        return float(np.sqrt(2.0) * spec["b"])
    if k == "mirrored_gamma":
        return spec["rescaled_to_sigma"]
    raise ValueError(k)


def analytic_nll(spec):
    """(nll, dnll) as vectorized callables of the innovation epsilon = y-Hx,
    where nll = -log pi up to an additive constant. Laplace is smoothed at
    delta = 1e-3 b for the optimizer; mirrored_gamma's log t is continued
    quadratically below t0 so the objective is finite everywhere."""
    k = spec["kind"]
    if k == "gaussian":
        s2 = spec["sigma"] ** 2

        def nll(e):
            return 0.5 * np.asarray(e) ** 2 / s2

        def dnll(e):
            return np.asarray(e) / s2
        return nll, dnll
    if k == "mixture":
        w, s1, s2 = spec["w"], spec["sigma1"], spec["sigma2"]

        def parts(e):
            e = np.asarray(e, float)
            p1 = w * np.exp(-0.5 * e * e / s1 ** 2) / (s1 * np.sqrt(2 * np.pi))
            p2 = (1 - w) * np.exp(-0.5 * e * e / s2 ** 2) \
                / (s2 * np.sqrt(2 * np.pi))
            return p1, p2

        def nll(e):
            p1, p2 = parts(e)
            return -np.log(np.maximum(p1 + p2, 1e-300))

        def dnll(e):
            e = np.asarray(e, float)
            p1, p2 = parts(e)
            f = np.maximum(p1 + p2, 1e-300)
            fp = -e * (p1 / s1 ** 2 + p2 / s2 ** 2)
            return -fp / f
        return nll, dnll
    if k == "laplace":
        b = spec["b"]
        d2 = (1e-3 * b) ** 2

        def nll(e):
            return np.sqrt(np.asarray(e, float) ** 2 + d2) / b

        def dnll(e):
            e = np.asarray(e, float)
            return e / (b * np.sqrt(e * e + d2))
        return nll, dnll
    if k == "mirrored_gamma":
        c = spec["rescaled_to_sigma"] / (2.0 * np.sqrt(2.0))
        t0 = 1e-3

        def t_of(e):
            return 2.0 - np.asarray(e, float) / c

        def nll(e):
            t = t_of(e)
            reg = t > t0
            out = np.empty_like(t)
            out[reg] = -(np.log(t[reg]) - 0.5 * t[reg])
            dt = t[~reg] - t0
            out[~reg] = -(np.log(t0) - 0.5 * t0
                          + (1.0 / t0 - 0.5) * dt
                          - 0.5 * dt * dt / t0 ** 2)
            return out

        def dnll(e):
            t = t_of(e)
            reg = t > t0
            dldt = np.empty_like(t)
            dldt[reg] = 1.0 / t[reg] - 0.5
            dldt[~reg] = (1.0 / t0 - 0.5) - (t[~reg] - t0) / t0 ** 2
            return dldt / c            # dnll/de = -dl/dt * dt/de, dt/de=-1/c
        return nll, dnll
    raise ValueError(k)


def spec_nll(fmt_spec, half_width):
    """(nll, dnll) in epsilon from a Format A spec, using the exact
    Density.score the C++ applies (at d = -epsilon, the export's reflected
    variable) and a fine tabulated integral of the score for the value, so
    line searches and the score stay consistent."""
    den = DY.Density(fmt_spec)
    dg = np.linspace(-half_width, half_width, 20001)
    sc = np.array([den.score(float(d)) for d in dg])
    logf = np.concatenate([[0.0], np.cumsum(-0.5 * (sc[1:] + sc[:-1])
                                            * np.diff(dg))])

    def nll(e):
        d = -np.asarray(e, float)
        return -np.interp(d, dg, logf)

    def dnll(e):
        d = -np.asarray(e, float)
        return -np.interp(d, dg, sc)
    return nll, dnll


# ---------------------------------------------------------------------------
# the MAP solve: curvature-clipped Newton with Armijo backtracking
# ---------------------------------------------------------------------------

def solve_map(Cinv, H, xb, y, nll, dnll, x0, tol=1e-9, maxit=200):
    x = x0.copy()
    n = x.size
    h = 1e-5

    def J(xv):
        e = y - H @ xv
        r = xv - xb
        return 0.5 * r @ Cinv @ r + float(np.sum(nll(e)))

    for _ in range(maxit):
        e = y - H @ x
        r = x - xb
        g = Cinv @ r + H.T @ dnll(e)
        if np.max(np.abs(g)) < tol:
            return x, J(x), True
        # per-ob curvature of nll by central difference, clipped to >= 0
        c = (dnll(e + h) - dnll(e - h)) / (2 * h)
        c = np.clip(c, 0.0, None)
        Hn = Cinv + (H.T * c) @ H + 1e-12 * np.eye(n)
        try:
            p = -np.linalg.solve(Hn, g)
        except np.linalg.LinAlgError:
            p = -g
        if g @ p > -1e-14:
            p = -g
        j0, gp = J(x), g @ p
        step = 1.0
        for _ in range(60):
            xn = x + step * p
            jn = J(xn)
            if np.isfinite(jn) and jn <= j0 + 1e-4 * step * gp:
                break
            step *= 0.5
        else:
            return x, j0, False
        x = xn
    return x, J(x), np.max(np.abs(Cinv @ (x - xb)
                                  + H.T @ dnll(y - H @ x))) < 100 * tol


def multistart_map(Cinv, H, xb, y, nll, dnll, starts):
    best, sols = None, []
    ok_any = False
    for x0 in starts:
        x, j, ok = solve_map(Cinv, H, xb, y, nll, dnll, x0)
        ok_any = ok_any or ok
        sols.append((j, x))
        if best is None or j < best[0]:
            best = (j, x)
    basins = []
    for j, x in sols:
        if not any(np.max(np.abs(x - b)) < 1e-3 for b in basins):
            basins.append(x)
    return best[1], best[0], len(basins), ok_any


# ---------------------------------------------------------------------------
# one replicate
# ---------------------------------------------------------------------------

def estimate_density(y, hofx_members, seed, lam, adaptive):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if adaptive:
            xg, pi, cache = R.estimate_adaptive_from_ensemble(
                y, hofx_members, seed=seed)
        else:
            grid, f_d, f_k, innov = R.histograms_from_ensemble(
                y, hofx_members, seed=seed)
            groups = R.innovation_groups(len(y), hofx_members.shape[1])
            xg, pi, cache = R.estimate_from_histograms(
                grid, f_d, f_k, lam=lam, innov=innov, groups=groups)
    return np.asarray(xg), np.asarray(pi), cache


def replicate(a, seed, quiet):
    rng = np.random.default_rng(seed)
    C = prior_cov(a.sigma_b, a.length_scale)
    Cs = np.linalg.cholesky(C)
    Cinv = np.linalg.inv(C)
    H = interp_operator(a.n_obs)

    base = 4.0 * (np.sin(np.arange(NGRID) / 5.5)
                  + 0.5 * np.cos(np.arange(NGRID) / 2.1))
    truth = base + Cs @ rng.standard_normal(NGRID)
    xb = truth + Cs @ rng.standard_normal(NGRID)     # background error ~ N(0,C)
    eps, spec_inj = draw_errors(a.density, rng, a.n_obs, a.scale)
    spec_inj["sample_sigma"] = sample_sigma_of(spec_inj)
    y = H @ truth + eps

    # DOEE ensemble: members exchangeable with the truth (the Stage A
    # construction), independent of xb, mapped through the same H
    members = truth[:, None] \
        + Cs @ rng.standard_normal((NGRID, a.members))
    xg, pi, cache = estimate_density(y, H @ members, seed + 2, a.lam,
                                     a.adaptive)
    sd, ku = moments_on(xg, pi)
    fine = np.arange(-10.0, 10.0001, 0.02)
    tru = analytic_density(spec_inj, fine)
    p_i = np.interp(fine, xg, pi, left=0.0, right=0.0)
    tot = p_i.sum() * 0.02
    l1 = float(np.abs(p_i / tot - tru).sum() * 0.02) if tot > 0 else np.nan

    # arms ------------------------------------------------------------------
    arms = {}
    arms["gaussian"] = analytic_nll({"kind": "gaussian",
                                     "sigma": a.assumed_error})
    arms["true"] = analytic_nll(spec_inj)

    gaussian_tails(cache, xg, pi, sd, Quiet(not quiet),
                   junction_frac=a.junction_frac)
    est_spec, nfixed = DY.to_spec(cache)
    est_bad = DY.check(est_spec)
    half = 12.0 * max(sd, spec_inj["sample_sigma"], a.assumed_error)
    if not est_bad:
        arms["estimated"] = spec_nll(est_spec, half)

    tf_spec, _ = DY.to_spec(oracle_cache(spec_inj))
    tf_bad = DY.check(tf_spec)
    if not tf_bad and not a.no_true_fmt:
        arms["true-fmt"] = spec_nll(tf_spec, half)

    # multistart solves -----------------------------------------------------
    starts = [xb, truth]
    for _ in range(a.extra_starts):
        starts.append(xb + Cs @ rng.standard_normal(NGRID))
    out = {"sd": sd, "ku": ku, "l1": l1, "nfixed": nfixed,
           "est_export_ok": not est_bad, "resolvability":
           float(cache.get("resolvability", np.nan))}
    sols = {}
    for name, (nll, dnll) in arms.items():
        x, j, nb, ok = multistart_map(Cinv, H, xb, y, nll, dnll, starts)
        sols[name] = x
        out[f"basins_{name}"] = nb
        out[f"converged_{name}"] = ok
    if "estimated" not in sols:
        return out, est_bad

    nll_t, _ = arms["true"]

    def j_true(x):
        r = x - xb
        return 0.5 * r @ Cinv @ r + float(np.sum(nll_t(y - H @ x)))

    xt = sols["true"]
    jt = j_true(xt)
    out["rmse_bg"] = float(np.sqrt(np.mean((xb - xt) ** 2)))
    for name in sols:
        if name == "true":
            continue
        d = sols[name] - xt
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(d ** 2)))
        out[f"regret_{name}"] = (j_true(sols[name]) - jt) / a.n_obs
    return out, []


# ---------------------------------------------------------------------------
# selftest: closed form and the null case
# ---------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(3)
    C = prior_cov()
    Cinv = np.linalg.inv(C)
    H = interp_operator(120)
    xb = rng.standard_normal(NGRID)
    truth = xb + np.linalg.cholesky(C) @ rng.standard_normal(NGRID)
    s = 0.4
    y = H @ truth + rng.normal(0, s, 120)
    nll, dnll = analytic_nll({"kind": "gaussian", "sigma": s})
    x, j, ok = solve_map(Cinv, H, xb, y, nll, dnll, xb)
    A = Cinv + H.T @ H / s ** 2
    x_cf = xb + np.linalg.solve(A, H.T @ (y - H @ xb) / s ** 2)
    err = float(np.max(np.abs(x - x_cf)))
    print(f"closed-form Gaussian MAP: max deviation {err:.2e} "
          f"(converged {ok})")
    assert ok and err < 1e-7, "solver does not reproduce the Gaussian MAP"

    # null case: injected Gaussian at the assumed sigma; the gaussian and
    # true arms must coincide to solver tolerance
    class A0:
        density, scale, assumed_error = "gaussian", 1.0, 0.4
        sigma_b, length_scale = 0.6, 1.0
        n_obs, members, lam, adaptive = 400, 50, 30.0, False
        junction_frac, extra_starts, no_true_fmt = 0.98, 2, False
    out, bad = replicate(A0, 11, quiet=True)
    assert not bad, f"null-case export rejected: {bad[:1]}"
    print(f"null case: rmse_gaussian {out['rmse_gaussian']:.2e} "
          f"(gaussian arm vs true arm; should be ~0), "
          f"rmse_estimated {out.get('rmse_estimated', np.nan):.4f}, "
          f"regret_gaussian {out['regret_gaussian']:.2e}")
    assert out["rmse_gaussian"] < 1e-6, \
        "gaussian and true arms differ in the null case"
    assert out["regret_gaussian"] < 1e-10, \
        "nonzero true-cost regret in the null case"
    print("selftest passed")
    return 0


# ---------------------------------------------------------------------------

def ci(vals, nboot=2000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, (nboot, v.size))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", default="heavy",
                    choices=["gaussian", "heavy", "laplace",
                             "mirrored_gamma"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=400)
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--assumed-error", type=float, default=0.4)
    ap.add_argument("--sigma-b", type=float, default=0.6)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--junction-frac", type=float, default=0.98)
    ap.add_argument("--extra-starts", type=int, default=3)
    ap.add_argument("--no-true-fmt", action="store_true",
                    help="skip the true-through-Format-A arm")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.adaptive:
        a.lam = None

    print(f"MAP reference: density {a.density} scale {a.scale}, "
          f"n_obs {a.n_obs}, members {a.members}, assumed "
          f"{a.assumed_error}, "
          f"{'adaptive' if a.adaptive else f'lam {a.lam:g}'}, "
          f"{a.replicates} replicates")

    rows, skipped = [], 0
    for r in range(a.replicates):
        out, bad = replicate(a, a.seed + 100 * r, quiet=True)
        if bad:
            skipped += 1
            print(f"  rep {r}: DOEE export rejected "
                  f"({bad[0]}); replicate skipped")
            continue
        rows.append(out)
        basins = "/".join(str(out.get(f"basins_{n}", "-"))
                          for n in ("gaussian", "estimated", "true",
                                    "true-fmt"))
        print(f"  rep {r}: doee sd {out['sd']:.3f} L1 {out['l1']:.3f}  "
              f"rmse bg {out['rmse_bg']:.4f}  "
              f"gauss {out['rmse_gaussian']:.4f}  "
              f"est {out['rmse_estimated']:.4f}  "
              + (f"true-fmt {out['rmse_true-fmt']:.4f}  "
                 if 'rmse_true-fmt' in out else "")
              + f"regret g {out['regret_gaussian']:.5f} "
              f"e {out['regret_estimated']:.5f}  basins {basins}")
        neg = [k for k in out if k.startswith("regret_")
               and out[k] < -1e-9]
        if neg:
            print(f"       WARNING negative regret {neg}: the true-arm "
                  "multistart likely missed a basin; add --extra-starts")
        conv = [k for k in out if k.startswith("converged_") and not out[k]]
        if conv:
            print(f"       WARNING not converged: {conv}")

    if not rows:
        print("no completed replicates")
        return 1
    print(f"\nsummary over {len(rows)} replicates"
          + (f" ({skipped} skipped on export gates)" if skipped else ""))
    for name in ("gaussian", "estimated", "true-fmt"):
        key = f"rmse_{name}"
        if key not in rows[0]:
            continue
        v = [o[key] for o in rows]
        lo, hi = ci(v)
        print(f"  rmse to true MAP, {name:9s} mean {np.mean(v):.4f}  "
              f"CI [{lo:.4f}, {hi:.4f}]")
    for name in ("gaussian", "estimated", "true-fmt"):
        key = f"regret_{name}"
        if key not in rows[0]:
            continue
        v = [o[key] for o in rows]
        lo, hi = ci(v)
        print(f"  true-cost regret, {name:9s} mean {np.mean(v):.5f} "
              f"nats/ob  CI [{lo:.5f}, {hi:.5f}]")
    d = [o["rmse_gaussian"] - o["rmse_estimated"] for o in rows]
    lo, hi = ci(d)
    print(f"  paired rmse margin (gaussian - estimated): mean "
          f"{np.mean(d):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({'estimated closer to the true MAP' if np.mean(d) > 0 else 'gaussian closer to the true MAP'})")
    dr = [o["regret_gaussian"] - o["regret_estimated"] for o in rows]
    lo, hi = ci(dr)
    print(f"  paired regret margin (gaussian - estimated): mean "
          f"{np.mean(dr):+.5f} nats/ob  CI [{lo:+.5f}, {hi:+.5f}]")
    if "rmse_true-fmt" in rows[0]:
        v = [o["rmse_true-fmt"] for o in rows]
        print(f"  Format A representation bound: true-fmt sits "
              f"{np.mean(v):.4f} rmse from the analytic true MAP; "
              "estimation error below this is invisible to the export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
