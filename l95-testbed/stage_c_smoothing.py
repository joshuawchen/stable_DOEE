#!/usr/bin/env python3
"""Stage C: the smoothing view and the ideal DOEE samples (single window).

NOTES_CONVERGENCE.md section 8 in executable form. For the observation
being scored, valid member samples must (i) not condition on that
observation and (ii) be calibrated draws given whatever they do condition
on. The optimum conditions maximally subject to (i): the LEAVE-ONE-OUT
smoothing posterior, whose spread is the narrowest valid kernel. This
module races the ladder at matched n:

    prior      members from the climatological prior N(0, C): valid,
               widest kernel (the Stage A construction)
    residual   full smoothing-posterior members used naively: INVALID --
               fitted residuals, shrunk toward the noise; included to
               measure the bias the influence identity predicts
               (recovered width ~ sigma sqrt(1 - d/n) in the
               linear-Gaussian case)
    loo        leave-one-out members by importance-reweighting the full
               smoothing ensemble per observation, w_k proportional to
               1/pi(r_k) (single-observation removal keeps weights mild;
               mean effective sample size is reported)

Setup: strong-constraint 4D in anomaly coordinates. Deterministic linear
evolution M = a R (R the one-point roll) inside the window, T observation
times, m locations per time, H_eff stacked, n = T m, prior N(0, C),
truth drawn from the prior, errors iid from the injected density. The
smoothing posterior is sampled exactly by per-member MALA chains under
the TRUE density (the fixed-point case: LOO minimizes the kernel GIVEN
calibration; miscalibration is the stage-B loop question). Nonlinear
in-window dynamics (l95 RK4 with finite-difference gradients) is a
straightforward extension left until the linear ladder is measured.

    python3 stage_c_smoothing.py --selftest
    python3 stage_c_smoothing.py --density heavy --T-list 3,10,25
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from map_reference import (NGRID, analytic_nll, draw_errors,  # noqa: E402
                           estimate_density, interp_operator, moments_on,
                           prior_cov, sample_sigma_of)
from stage_a_end_to_end import analytic_density  # noqa: E402
from stage_b_cycle import analyze_exact  # noqa: E402


def build_heff(m_per_time, T, a):
    """Stacked observation operator of the strong-constraint window:
    row block t observes x_t = (a R)^t x_0, t = 1..T."""
    H1 = interp_operator(m_per_time)
    P = np.roll(np.eye(NGRID), 1, axis=0)          # P @ x = roll(x, 1)
    blocks, M = [], np.eye(NGRID)
    for _ in range(T):
        M = a * (P @ M)
        blocks.append(H1 @ M)
    return np.vstack(blocks)


def l1_to_truth(xg, pi, spec_inj):
    fine = np.arange(-10.0, 10.0001, 0.02)
    tru = analytic_density(spec_inj, fine)
    p = np.interp(fine, xg, pi, left=0.0, right=0.0)
    tot = p.sum() * 0.02
    return float(np.abs(p / tot - tru).sum() * 0.02) if tot > 0 else np.nan


def estimate_arm(y, hofx, spec_inj, seed, lam, adaptive):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xg, pi, cache = estimate_density(y, hofx, seed, lam, adaptive)
    sd, _ = moments_on(xg, pi)
    return sd, l1_to_truth(xg, pi, spec_inj)


def loo_hofx(y, hofx_post, nll, K, rng):
    """Per-observation importance resampling of the full smoothing
    ensemble into approximate leave-one-out members: the LOO posterior
    differs from the full one by exactly the factor 1/pi(r_i), so
    w_k proportional to exp(nll(r_i^k)), softmax-normalized. Returns the
    (n, K) member matrix and the mean effective sample size."""
    n, Kref = hofx_post.shape
    out = np.empty((n, K))
    ess = np.empty(n)
    for i in range(n):
        lw = np.asarray(nll(y[i] - hofx_post[i]), float)
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
        ess[i] = 1.0 / np.sum(w * w)
        idx = rng.choice(Kref, size=K, replace=True, p=w)
        out[i] = hofx_post[i, idx]
    return out, float(ess.mean())


def one_window(a, T, seed, quiet=False):
    rng = np.random.default_rng(seed)
    C = prior_cov(a.sigma_b, a.length_scale)
    Cs = np.linalg.cholesky(C)
    Heff = build_heff(a.m_per_time, T, a.persistence)
    n = Heff.shape[0]
    z0 = Cs @ rng.standard_normal(NGRID)
    eps, spec_inj = draw_errors(a.density, rng, n, a.scale)
    spec_inj["sample_sigma"] = sample_sigma_of(spec_inj)
    y = Heff @ z0 + eps
    nll, dnll = analytic_nll(spec_inj)

    Xf = Cs @ rng.standard_normal((NGRID, a.kref))
    Zp, info = analyze_exact(Xf, y, Heff, C, nll, dnll, 1e-5, a.kref,
                             np.random.default_rng(seed + 13),
                             prior=(np.zeros(NGRID), C))
    hofx_post = Heff @ Zp

    arms = {}
    Zprior = Cs @ rng.standard_normal((NGRID, a.members))
    arms["prior"] = Heff @ Zprior
    arms["residual"] = hofx_post[:, :a.members].copy()
    arms["loo"], mean_ess = loo_hofx(y, hofx_post, nll, a.members,
                                     np.random.default_rng(seed + 29))

    row = {"n": n, "acc": info["acc_rate"], "ess": mean_ess,
           "sigma_true": spec_inj["sample_sigma"],
           "shrink": float(spec_inj["sample_sigma"]
                           * np.sqrt(max(1.0 - NGRID / n, 0.0)))}
    for name, hofx in arms.items():
        sd, l1 = estimate_arm(y, hofx, spec_inj, seed + 2, a.lam,
                              a.adaptive)
        row[f"sd_{name}"], row[f"l1_{name}"] = sd, l1
    if not quiet:
        print(f"    T {T:3d} n {n:5d} d/n {NGRID / n:.2f}  "
              f"prior sd {row['sd_prior']:.3f} L1 {row['l1_prior']:.3f}  "
              f"residual sd {row['sd_residual']:.3f} "
              f"L1 {row['l1_residual']:.3f} "
              f"(shrink predicts {row['shrink']:.3f})  "
              f"loo sd {row['sd_loo']:.3f} L1 {row['l1_loo']:.3f}  "
              f"ESS {mean_ess:.0f}/{a.kref}  acc {row['acc']:.2f}")
    return row


def selftest():
    class A0:
        density, scale = "gaussian", 1.0
        sigma_b, length_scale, persistence = 0.6, 1.0, 0.9
        m_per_time, members, kref = 40, 50, 400
        lam, adaptive = 3.0, False
    a = A0()
    rng = np.random.default_rng(3)
    C = prior_cov(a.sigma_b, a.length_scale)
    Cs = np.linalg.cholesky(C)
    Heff = build_heff(40, 5, a.persistence)
    z0 = Cs @ rng.standard_normal(NGRID)
    s = 0.4
    y = Heff @ z0 + rng.normal(0, s, Heff.shape[0])
    nll, dnll = analytic_nll({"kind": "gaussian", "sigma": s})
    Xf = Cs @ rng.standard_normal((NGRID, 200))
    Zp, info = analyze_exact(Xf, y, Heff, C, nll, dnll, 1e-5, 200,
                             np.random.default_rng(11),
                             prior=(np.zeros(NGRID), C))
    Cinv = np.linalg.inv(C)
    A = Cinv + Heff.T @ Heff / s ** 2
    kal = np.linalg.solve(A, Heff.T @ y / s ** 2)
    dev = float(np.sqrt(np.mean((Zp.mean(axis=1) - kal) ** 2)))
    print(f"stacked-window MALA vs Kalman: mean deviation {dev:.4f} "
          f"(acc {info['acc_rate']:.2f})")
    assert dev < 0.10, "smoothing sampler misses the stacked Kalman mean"
    assert 0.2 < info["acc_rate"] < 0.95, "MALA acceptance unhealthy"

    row = one_window(a, 5, 21, quiet=True)
    print(f"gaussian null (n {row['n']}): residual sd "
          f"{row['sd_residual']:.3f} (shrink predicts {row['shrink']:.3f} "
          f"of true {row['sigma_true']:.3f}); loo sd {row['sd_loo']:.3f}; "
          f"ESS {row['ess']:.0f}")
    assert row["sd_residual"] < row["sd_loo"], \
        "naive residuals should be narrower than loo (shrinkage)"
    assert abs(row["sd_loo"] - row["sigma_true"]) \
        < abs(row["sd_residual"] - row["sigma_true"]), \
        "loo should recover the width better than naive residuals"
    assert row["ess"] > 0.4 * a.kref, "loo weights degenerate"
    print("selftest passed")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", default="heavy",
                    choices=["gaussian", "heavy", "laplace",
                             "mirrored_gamma"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--T-list", default="3,10,25",
                    help="observation times per window; n = T x m")
    ap.add_argument("--m-per-time", type=int, default=40)
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--kref", type=int, default=400,
                    help="smoothing-ensemble size the loo arm resamples "
                         "from")
    ap.add_argument("--sigma-b", type=float, default=0.6)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--persistence", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.adaptive:
        a.lam = None

    Ts = [int(s) for s in a.T_list.split(",")]
    print(f"stage C ladder: density {a.density} scale {a.scale}, "
          f"m {a.m_per_time}/time, members {a.members}, kref {a.kref}, "
          f"{'adaptive' if a.adaptive else f'lam {a.lam:g}'}, "
          f"{a.replicates} replicates per T")
    for T in Ts:
        rows = [one_window(a, T, a.seed + 100 * r) for r in
                range(a.replicates)]
        line = f"  T {T:3d} means:"
        for name in ("prior", "residual", "loo"):
            l1 = np.mean([r[f"l1_{name}"] for r in rows])
            sd = np.mean([r[f"sd_{name}"] for r in rows])
            line += f"  {name} sd {sd:.3f} L1 {l1:.3f}"
        print(line)
    print("\nreading: prior = widest valid kernel (noisy), residual = "
          "invalid (width biased low toward sqrt(1 - d/n)), loo = the "
          "ideal samples; the loo column should win at small n and the "
          "three should converge as d/n -> 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
