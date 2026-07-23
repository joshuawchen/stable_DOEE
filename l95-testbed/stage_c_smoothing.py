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

from map_reference import (NGRID, Quiet, analytic_density_ext,  # noqa: E402
                           analytic_nll, draw_errors, estimate_density,
                           interp_operator, moments_on, multistart_map,
                           prior_cov, sample_sigma_of, spec_nll)
from jedi_export import doee_to_yaml as DY  # noqa: E402
from stage_a_end_to_end import analytic_density, gaussian_tails  # noqa: E402
from stage_b_cycle import analyze_exact, analyze_pff  # noqa: E402


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
    tru = analytic_density_ext(spec_inj, fine)
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
    w_k proportional to exp(nll(r_i^k)), softmax-normalized -- with
    Ionides-style truncation at mean(w) sqrt(Kref), because for densities
    that vanish at a support wall 1/pi(r) is UNBOUNDED and a few
    wall-violating members otherwise absorb all the weight (the known
    IS-LOO failure mode; PSIS or exact per-observation refits are the
    principled upgrades). Returns the (n, K) member matrix and the mean
    effective sample size AFTER truncation."""
    n, Kref = hofx_post.shape
    out = np.empty((n, K))
    ess = np.empty(n)
    for i in range(n):
        lw = np.asarray(nll(y[i] - hofx_post[i]), float)
        lw -= lw.max()
        w = np.exp(lw)
        w = np.minimum(w, w.mean() * np.sqrt(Kref))
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


def l1_between(a_pair, b_pair):
    """Truth-free convergence criterion: L1 between successive recovered
    densities on a common grid."""
    if a_pair is None:
        return float("inf")
    fine = np.arange(-10.0, 10.0001, 0.02)
    ps = []
    for xg, pi in (a_pair, b_pair):
        p = np.interp(fine, xg, pi, left=0.0, right=0.0)
        t = p.sum() * 0.02
        ps.append(p / t if t > 0 else p)
    return float(np.abs(ps[0] - ps[1]).sum() * 0.02)


def export_gap(spec, xg, pi, sd, assumed):
    """L1 between the raw recovered density and the density the export
    actually delivers to the analysis (exp of the integrated spec score,
    normalized). The pipeline's L1 column scores the raw estimate, but
    the MAP consumes the export -- unimodality enforcement and the tail
    retraction were built on symmetric cases, and this measures what
    they do to an asymmetric one."""
    if spec is None:
        return float("nan")
    half = 12.0 * max(sd, assumed)
    nll_e, _ = spec_nll(spec, half)
    fine = np.arange(-half, half + 1e-9, 0.02)
    q = np.exp(-np.asarray(nll_e(fine)))
    tq = q.sum() * 0.02
    q = q / tq if tq > 0 else q
    p = np.interp(fine, xg, pi, left=0.0, right=0.0)
    tp = p.sum() * 0.02
    p = p / tp if tp > 0 else p
    return float(np.abs(p - q).sum() * 0.02)


def density_to_spec(y, hofx, a, seed):
    """DOEE through the export the DA consumes; None if the gates refuse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xg, pi, cache = estimate_density(y, hofx, seed, a.lam, a.adaptive)
    sd, _ = moments_on(xg, pi)
    gaussian_tails(cache, xg, pi, sd, Quiet(False))
    spec, _ = DY.to_spec(cache)
    return (None if DY.check(spec) else spec), sd, (xg, pi)


def pipeline_window(a, T, seed, quiet=False):
    """End-to-end: every arm produces a density, the density produces a
    4D MAP, and the MAP is scored against the true-density MAP. Arms:
    gauss-matched and gauss-best (the strongest variance-only Gaussians,
    closed form), prior (wide-kernel no-iteration baseline; no EDA and no
    loop are needed for it), loo (DOEE from leave-one-out
    innovations of a smoothing posterior sampled under the assumed
    Gaussian: the honest bootstrap, no truth anywhere), and the shared
    fixed-point loop. Truth enters scoring only (the reference MAP, the
    oracle Gaussians, the L1 diagnostics). One documented second-order
    liberty: from iteration 2 onward the members scoring y_i depend on
    y_i THROUGH pi-hat (a density estimated from all n innovations), an
    O(1/n) influence -- the standard empirical-Bayes reuse, same order
    as terms already tolerated. Reference: the MAP under the analytic
    injected density."""
    rng = np.random.default_rng(seed)
    C = prior_cov(a.sigma_b, a.length_scale)
    Cs = np.linalg.cholesky(C)
    Cinv = np.linalg.inv(C)
    Heff = build_heff(a.m_per_time, T, a.persistence)
    n = Heff.shape[0]
    z0 = Cs @ rng.standard_normal(NGRID)
    eps, spec_inj = draw_errors(a.density, rng, n, a.scale)
    spec_inj["sample_sigma"] = sample_sigma_of(spec_inj)
    y = Heff @ z0 + eps
    nll_t, dnll_t = analytic_nll(spec_inj)

    # reference: the true-density MAP, multistart (the truth start is
    # legitimate HERE only -- we want the global reference MAP)
    zeros = np.zeros(NGRID)
    starts_ref = [zeros, z0] + [Cs @ rng.standard_normal(NGRID)
                                for _ in range(2)]
    xt, _, _, _ = multistart_map(Cinv, Heff, zeros, y, nll_t,
                                 dnll_t, starts_ref)

    def j_true(x):
        return 0.5 * x @ Cinv @ x + float(np.sum(nll_t(y - Heff @ x)))

    jt = j_true(xt)
    out = {"n": n}
    wall = (spec_inj["kind"] == "mirrored_gamma")
    wall_edge = (2.0 * spec_inj["rescaled_to_sigma"] / (2.0 * np.sqrt(2.0))
                 if wall else None)

    def score(tag, x):
        out[f"rmse_{tag}"] = float(np.sqrt(np.mean((x - xt) ** 2)))
        out[f"regret_{tag}"] = (j_true(x) - jt) / n
        if wall:
            # fraction of residuals past the hard support edge:
            # probability-zero events under the true density, so regret
            # is formally infinite there and its printed magnitude is an
            # artifact of the log-density continuation; rmse and this
            # fraction are the meaningful scores for wall densities
            out[f"viol_{tag}"] = float(np.mean((y - Heff @ x)
                                               > wall_edge))

    # Gaussian ladder, closed form
    HtH = Heff.T @ Heff
    Hty = Heff.T @ y

    def gauss_map(sig):
        return np.linalg.solve(Cinv + HtH / sig ** 2, Hty / sig ** 2)

    sm = spec_inj["sample_sigma"]
    score("gaussM", gauss_map(sm))
    sgrid = np.geomspace(0.3 * sm, 3.0 * sm, 40)
    regs = [j_true(gauss_map(float(s))) for s in sgrid]
    k = int(np.argmin(regs))
    fine_s = np.linspace(sgrid[max(k - 1, 0)],
                         sgrid[min(k + 1, sgrid.size - 1)], 25)
    sB = float(fine_s[int(np.argmin([j_true(gauss_map(float(s)))
                                     for s in fine_s]))])
    score("gaussB", gauss_map(sB))

    # honest starts for the honest arms: no truth anywhere
    starts_h = [zeros, gauss_map(a.assumed_error)] \
        + [Cs @ rng.standard_normal(NGRID) for _ in range(2)]

    def map_under(spec, sd_est):
        if spec is None:
            return None
        half = 12.0 * max(sd_est, a.assumed_error)
        nll_e, dnll_e = spec_nll(spec, half)
        x, _, _, _ = multistart_map(Cinv, Heff, zeros, y, nll_e, dnll_e,
                                    starts_h,
                                    h=0.5 * spec["grid spacing"])
        return x

    # prior arm: wide-kernel baseline. No data in the loop, no iteration,
    # and no EDA needed to reach its MAP -- a genuine EDA arm only exists
    # under cycling (stage B), where the ensemble comes from DA.
    Zp0 = Cs @ rng.standard_normal((NGRID, a.members))
    spec_e, sd_e, (xg_e, pi_e) = density_to_spec(y, Heff @ Zp0, a,
                                                 seed + 2)
    out["l1_prior"] = l1_to_truth(xg_e, pi_e, spec_inj)
    x_e = map_under(spec_e, sd_e)
    if x_e is not None:
        score("prior", x_e)

    # the two ITERATED pipelines share one loop: sample the smoothing
    # posterior under the current likelihood, leave-one-out reweight,
    # re-estimate, feed back; stop when successive estimates agree
    # (truth-free L1) or at max-iters. They differ in ONE thing: what the
    # loop and the final MAP retain. "gauss" keeps variance only (the
    # honest good-Gaussian pipeline); "shape" keeps the full density
    # (ours). Their margin is the value of shape, nothing else.
    Xf = Cs @ rng.standard_normal((NGRID, a.kref))

    def sample_posterior(cur_nll, cur_dnll, cur_h, sub, it):
        rk = np.random.default_rng(seed + 13 + 1000 * sub + it)
        if a.sampler == "pff":
            Zp, _ = analyze_pff(Xf, y, Heff, C, cur_nll, cur_dnll,
                                cur_h, a.kref, rk, prior=(zeros, C))
            m_ = Zp.mean(axis=1, keepdims=True)
            Zp = m_ + a.pff_inflation * (Zp - m_)
            if sub == 1 and it == 0:
                # FREE exact calibration check: iteration 0 samples under
                # the assumed Gaussian, whose posterior is analytic
                A0 = Cinv + HtH / a.assumed_error ** 2
                mean0 = np.linalg.solve(A0, Hty / a.assumed_error ** 2)
                sd0 = np.sqrt(np.diag(np.linalg.inv(A0)))
                out["pff_sdr"] = float(np.mean(
                    np.std(Zp, axis=1, ddof=1) / sd0))
                out["pff_dev"] = float(np.sqrt(np.mean(
                    (Zp.mean(axis=1) - mean0) ** 2)))
                if not (0.90 <= out["pff_sdr"] <= 1.08):
                    print(f"       PFF CALIBRATION WARNING: sd ratio "
                          f"{out['pff_sdr']:.2f} vs the analytic Gaussian "
                          f"posterior; tune kref (now {a.kref}), "
                          f"--pff-inflation (now {a.pff_inflation}), or "
                          f"the bandwidth (median heuristic here; the "
                          f"YAML sigma in oops)")
            return Zp
        Zp, _ = analyze_exact(Xf, y, Heff, C, cur_nll, cur_dnll, cur_h,
                              a.kref, rk, prior=(zeros, C))
        return Zp

    def iterate(mode, sub):
        cur_nll, cur_dnll = analytic_nll({"kind": "gaussian",
                                          "sigma": a.assumed_error})
        cur_h, prev, spec_f, sd_f = 1e-5, None, None, a.assumed_error
        ess = delta = float("nan")
        it_used = 0
        hist = []
        for it in range(a.max_iters):
            Zp = sample_posterior(cur_nll, cur_dnll, cur_h, sub, it)
            hofx_loo, ess = loo_hofx(
                y, Heff @ Zp, cur_nll, a.members,
                np.random.default_rng(seed + 29 + 1000 * sub + it))
            spec, sd, (xg, pi) = density_to_spec(y, hofx_loo, a, seed + 2)
            if spec is None:
                break
            it_used = it + 1
            delta = l1_between(prev, (xg, pi))
            prev, spec_f, sd_f = (xg, pi), spec, sd
            hist.append((delta, spec, sd, (xg, pi)))
            if mode == "gauss":
                cur_nll, cur_dnll = analytic_nll(
                    {"kind": "gaussian", "sigma": sd})
                cur_h = 1e-5
            else:
                cur_nll, cur_dnll = spec_nll(
                    spec, 12.0 * max(sd, a.assumed_error))
                cur_h = 0.5 * spec["grid spacing"]
            if delta < a.iter_tol:
                break
        # damped selection, truth-free: if the loop did not settle, ship
        # the MOST SELF-CONSISTENT iterate (minimum successive L1) rather
        # than the last one -- an unconverged fixed-point map can
        # oscillate, and the last iterate of an oscillation is arbitrary
        if len(hist) >= 2:
            d_, s_, sd_, e_ = min(hist[1:], key=lambda h: h[0])
            if d_ < delta:
                spec_f, sd_f, prev = s_, sd_, e_
        return spec_f, sd_f, prev, it_used, ess, delta

    spec_g, sd_g, est_g, it_g, ess_g, dl_g = iterate("gauss", 1)
    if est_g is not None:
        out["l1_gaussI"] = l1_to_truth(*est_g, spec_inj)
        out["sd_gaussI"], out["it_gaussI"] = sd_g, it_g
        out["dl_gaussI"] = dl_g
        score("gaussI", gauss_map(sd_g))

    spec_s, sd_s, est_s, it_s, ess_s, dl_s = iterate("shape", 2)
    if spec_s is not None:
        out["l1_loo"] = l1_to_truth(*est_s, spec_inj)
        out["ess"], out["it_loo"], out["dl_loo"] = ess_s, it_s, dl_s
        x_s = map_under(spec_s, sd_s)
        if x_s is not None:
            score("loo", x_s)

    if not quiet:
        key = "rmse" if wall else "regret"
        bits = [f"    T {T:3d} n {n:5d} [{key}]"]
        for tag in ("gaussM", "gaussB", "prior", "gaussI", "loo"):
            if f"{key}_{tag}" in out:
                bits.append(f"{tag} {out[f'{key}_{tag}']:.4f}"
                            + (f"({100 * out[f'viol_{tag}']:.0f}%)"
                               if wall and f"viol_{tag}" in out else ""))
        bits.append(f"L1 prior {out.get('l1_prior', float('nan')):.2f} "
                    f"gaussI {out.get('l1_gaussI', float('nan')):.2f} "
                    f"loo {out.get('l1_loo', float('nan')):.2f}")
        bits.append(f"iters {out.get('it_gaussI', 0)}/"
                    f"{out.get('it_loo', 0)} "
                    f"conv {out.get('dl_gaussI', float('nan')):.3f}/"
                    f"{out.get('dl_loo', float('nan')):.3f} "
                    f"ESS {out.get('ess', 0):.0f}"
                    + (f" pff-sdr {out['pff_sdr']:.2f}"
                       if "pff_sdr" in out else ""))
        print("  ".join(bits))
    return out


def _ci(v, nboot=2000, seed=0):
    v = np.asarray(v, float)
    r = np.random.default_rng(seed)
    m = v[r.integers(0, v.size, (nboot, v.size))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run_pipeline(a):
    Ts = [int(s) for s in a.T_list.split(",")]
    print(f"stage C pipeline: density {a.density} scale {a.scale}, "
          f"m {a.m_per_time}/time, members {a.members}, kref {a.kref}, "
          f"assumed {a.assumed_error}, "
          f"{'adaptive' if a.adaptive else f'lam {a.lam:g}'}, "
          f"sampler {a.sampler}"
          + (f" (inflation {a.pff_inflation})" if a.sampler == "pff"
             else "")
          + f", max-iters {a.max_iters}, {a.replicates} replicates "
          f"per T (regret in nats/ob vs the true-density MAP)")
    for T in Ts:
        rows = [pipeline_window(a, T, a.seed + 100 * r)
                for r in range(a.replicates)]
        wall = (a.density == "mirrored_gamma")
        key = "rmse" if wall else "regret"
        line = f"  T {T:3d} mean {key}:"
        for tag in ("gaussM", "gaussB", "prior", "gaussI", "loo"):
            v = [o[f"{key}_{tag}"] for o in rows if f"{key}_{tag}" in o]
            if v:
                line += f"  {tag} {np.mean(v):.4f}"
                if wall:
                    w = [o[f"viol_{tag}"] for o in rows
                         if f"viol_{tag}" in o]
                    line += f"({100 * np.mean(w):.0f}%)"
        print(line)
        if wall:
            print("        wall density: regret is formally infinite for "
                  "any wall-violating analysis, so scores are rmse to "
                  "the true MAP with the wall-violation fraction in "
                  "parentheses")
        for base, name in (("gaussI", "iterated Gaussian"),
                           ("gaussB", "oracle Gaussian")):
            v_b = [o[f"{key}_{base}"] for o in rows
                   if f"{key}_{base}" in o and f"{key}_loo" in o]
            v_l = [o[f"{key}_loo"] for o in rows
                   if f"{key}_{base}" in o and f"{key}_loo" in o]
            if v_l:
                d = np.array(v_b) - np.array(v_l)
                lo, hi = _ci(d)
                print(f"        paired margin ({name} - loo, {key}): "
                      f"{np.mean(d):+.4f} CI [{lo:+.4f}, "
                      f"{hi:+.4f}] "
                      f"({int((d > 0).sum())}/{len(d)} windows loo wins)")
    return 0


def run_windows(a):
    """The operational mode: W independent re-anchored windows sharing
    one pooled LOO archive, run PREQUENTIALLY -- window w is analyzed and
    posterior-sampled under the density estimated from windows 1..w-1
    only, then contributes its innovations. H1 therefore holds EXACTLY
    (no empirical-Bayes reuse at all), the across-window loop replaces
    the within-window iteration, and its kicks shrink with the archive,
    so the fixed-point map self-damps. The economics this measures: the
    Gaussian's total regret per window is a CONSTANT (its shape tax),
    while the shape pipeline's falls like 1/W -- the crossover window is
    the operational promise."""
    T = int(a.T_list.split(",")[0])
    rng0 = np.random.default_rng(a.seed)
    C = prior_cov(a.sigma_b, a.length_scale)
    Cs = np.linalg.cholesky(C)
    Cinv = np.linalg.inv(C)
    Heff = build_heff(a.m_per_time, T, a.persistence)
    n = Heff.shape[0]
    zeros = np.zeros(NGRID)
    print(f"stage C prequential windows: density {a.density} scale "
          f"{a.scale}, {a.windows} windows x (T {T} x m {a.m_per_time} "
          f"= {n} obs), members {a.members}, kref {a.kref}, assumed "
          f"{a.assumed_error}, "
          f"{'adaptive' if a.adaptive else f'lam {a.lam:g}'}, sampler "
          f"{a.sampler} (regret in nats/ob vs each window's true MAP)")
    g_nll0, g_dnll0 = analytic_nll({"kind": "gaussian",
                                    "sigma": a.assumed_error})
    pipes = {"gauss": {"nll": g_nll0, "dnll": g_dnll0, "h": 1e-5,
                       "spec": None, "sd": a.assumed_error,
                       "obs": [], "hofx": []},
             "shape": {"nll": g_nll0, "dnll": g_dnll0, "h": 1e-5,
                       "spec": None, "sd": a.assumed_error,
                       "obs": [], "hofx": []}}
    hist = {k: [] for k in ("gaussM", "gaussB", "gauss", "shape")}
    ys = []

    def sample_post(y_v, st, rk):
        Xf_v = Cs @ rk.standard_normal((NGRID, a.kref))
        if a.sampler == "pff":
            Zv, _ = analyze_pff(Xf_v, y_v, Heff, C, st["nll"],
                                st["dnll"], st["h"], a.kref, rk,
                                prior=(zeros, C))
            m_ = Zv.mean(axis=1, keepdims=True)
            return m_ + a.pff_inflation * (Zv - m_)
        Zv, _ = analyze_exact(Xf_v, y_v, Heff, C, st["nll"],
                              st["dnll"], st["h"], a.kref, rk,
                              prior=(zeros, C))
        return Zv

    def loo_row(y_v, st, rk):
        Zv = sample_post(y_v, st, rk)
        h_v, ess_v = loo_hofx(y_v, Heff @ Zv, st["nll"], a.members,
                              np.random.default_rng(rk.integers(2 ** 31)))
        if a.center_innovations:
            y_v = y_v - float(np.mean(y_v[:, None] - h_v))
        return y_v, h_v, ess_v

    for w in range(a.windows):
        rng = np.random.default_rng(a.seed + 100 * w)
        z0 = Cs @ rng.standard_normal(NGRID)
        eps, spec_inj = draw_errors(a.density, rng, n, a.scale)
        spec_inj["sample_sigma"] = sample_sigma_of(spec_inj)
        y = Heff @ z0 + eps
        ys.append(y.copy())
        nll_t, dnll_t = analytic_nll(spec_inj)
        starts_ref = [zeros, z0] + [Cs @ rng.standard_normal(NGRID)
                                    for _ in range(2)]
        xt, _, _, _ = multistart_map(Cinv, Heff, zeros, y, nll_t,
                                     dnll_t, starts_ref)

        def j_true(x):
            return 0.5 * x @ Cinv @ x \
                + float(np.sum(nll_t(y - Heff @ x)))

        jt = j_true(xt)
        HtH = Heff.T @ Heff
        Hty = Heff.T @ y

        def gauss_map(sig):
            return np.linalg.solve(Cinv + HtH / sig ** 2,
                                   Hty / sig ** 2)

        sm = spec_inj["sample_sigma"]
        row = {}

        def reg(x):
            return (j_true(x) - jt) / n

        row["gaussM"] = reg(gauss_map(sm))
        sgrid = np.geomspace(0.3 * sm, 3.0 * sm, 40)
        k_ = int(np.argmin([j_true(gauss_map(float(s)))
                            for s in sgrid]))
        fine_s = np.linspace(sgrid[max(k_ - 1, 0)],
                             sgrid[min(k_ + 1, sgrid.size - 1)], 25)
        sB = float(fine_s[int(np.argmin([j_true(gauss_map(float(s)))
                                         for s in fine_s]))])
        row["gaussB"] = reg(gauss_map(sB))

        starts_h = [zeros, gauss_map(a.assumed_error)] \
            + [Cs @ rng.standard_normal(NGRID) for _ in range(2)]
        l1s, ess_w = {}, float("nan")
        for name, st in pipes.items():
            # ANALYZE window w under the density from windows 1..w-1
            if name == "gauss" or st["spec"] is None:
                x_w = gauss_map(st["sd"])
            else:
                half = 12.0 * max(st["sd"], a.assumed_error)
                nll_e, dnll_e = spec_nll(st["spec"], half)
                x_w, _, _, _ = multistart_map(
                    Cinv, Heff, zeros, y, nll_e, dnll_e, starts_h,
                    h=0.5 * st["spec"]["grid spacing"])
            row[name] = reg(x_w)
            # build the estimation archive per --archive-mode: pooled
            # keeps every window's rows as generated (online EM, infinite
            # memory of stale E-steps -- measured to run away on skewed);
            # regen REGENERATES every window's LOO rows under the CURRENT
            # density (batch EM, self-consistent archive); recent keeps a
            # sliding memory so stale imprints age out
            off = 0 if name == "gauss" else 1
            if a.archive_mode == "regen":
                obs_l, hof_l = [], []
                ess_w = float("nan")
                for v in range(w + 1):
                    rk = np.random.default_rng(
                        a.seed + 11 + 1000 * v + 100000 * w + off)
                    y_u, h_v, ess_v = loo_row(ys[v], st, rk)
                    obs_l.append(y_u)
                    hof_l.append(h_v)
                    if v == w:
                        ess_w = ess_v
            else:
                rk = np.random.default_rng(a.seed + 7 + 1000 * w + off)
                y_u, h_v, ess_w = loo_row(y, st, rk)
                st["obs"].append(y_u)
                st["hofx"].append(h_v)
                if a.archive_mode == "recent":
                    obs_l = st["obs"][-a.archive_windows:]
                    hof_l = st["hofx"][-a.archive_windows:]
                else:
                    obs_l, hof_l = st["obs"], st["hofx"]
            spec, sd, (xg, pi) = density_to_spec(
                np.concatenate(obs_l), np.vstack(hof_l), a, a.seed + 2)
            l1s[name] = l1_to_truth(xg, pi, spec_inj)
            l1s[name + "_exp"] = export_gap(spec, xg, pi,
                                            sd if spec is not None
                                            else st["sd"],
                                            a.assumed_error)
            if spec is not None:
                st["spec"], st["sd"] = spec, sd
                if name == "gauss":
                    st["nll"], st["dnll"] = analytic_nll(
                        {"kind": "gaussian", "sigma": sd})
                    st["h"] = 1e-5
                else:
                    st["nll"], st["dnll"] = spec_nll(
                        spec, 12.0 * max(sd, a.assumed_error))
                    st["h"] = 0.5 * spec["grid spacing"]
        for k_ in hist:
            hist[k_].append(row[k_])
        print(f"    w {w:3d} N {n * (w + 1):5d}  "
              f"gaussM {row['gaussM']:.4f} gaussB {row['gaussB']:.4f} "
              f"gaussI {row['gauss']:.4f} loo {row['shape']:.4f}  "
              f"L1 gaussI {l1s.get('gauss', float('nan')):.2f} "
              f"loo {l1s.get('shape', float('nan')):.2f}  "
              f"exp {l1s.get('gauss_exp', float('nan')):.2f}/"
              f"{l1s.get('shape_exp', float('nan')):.2f}  "
              f"ESS {ess_w:.0f}")
    half_w = a.windows // 2
    print(f"  trailing {a.windows - half_w} windows, mean regret: "
          + "  ".join(f"{tag} {np.mean(hist[k][half_w:]):.4f}"
                      for tag, k in (("gaussM", "gaussM"),
                                     ("gaussB", "gaussB"),
                                     ("gaussI", "gauss"),
                                     ("loo", "shape"))))
    d = np.array(hist["gauss"][half_w:]) - np.array(hist["shape"][half_w:])
    if d.size >= 3:
        lo, hi = _ci(d)
        print(f"  trailing paired margin (prequential Gaussian - loo): "
              f"{np.mean(d):+.4f} nats/ob CI [{lo:+.4f}, {hi:+.4f}] "
              f"({int((d > 0).sum())}/{d.size} windows loo wins)")
    d_all = np.array(hist["gauss"]) - np.array(hist["shape"])
    suff = [float(np.mean(d_all[v:])) for v in range(a.windows)]
    stable = next((v for v in range(a.windows)
                   if all(s > 0 for s in suff[v:])), None)
    if stable is not None and stable <= a.windows - 3:
        print(f"  crossover: the margin stays positive in the mean from "
              f"window {stable} onward")
    else:
        print("  no stable crossover in this run")
    return 0


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

    class P0:
        density, scale = "gaussian", 1.0
        sigma_b, length_scale, persistence = 0.6, 1.0, 0.9
        m_per_time, members, kref = 40, 50, 400
        lam, adaptive = 3.0, False
        assumed_error, max_iters, iter_tol = 0.4, 3, 0.05
        sampler, pff_inflation = "mala", 1.05
    p = P0()
    prow = pipeline_window(p, 5, 33, quiet=True)
    print(f"pipeline gaussian null (n {prow['n']}): regret gaussM "
          f"{prow['regret_gaussM']:.2e} gaussI "
          f"{prow.get('regret_gaussI', float('nan')):.4f} "
          f"(sd {prow.get('sd_gaussI', float('nan')):.3f}, "
          f"{prow.get('it_gaussI', 0)} iters) loo "
          f"{prow.get('regret_loo', float('nan')):.4f}")
    assert prow["regret_gaussM"] < 1e-5, \
        "matched Gaussian must reproduce the true MAP in the null"
    assert "regret_gaussI" in prow and prow["regret_gaussI"] < 5e-3, \
        "iterated Gaussian pipeline fails the null"
    assert 0.32 < prow.get("sd_gaussI", 0.0) < 0.48, \
        "iterated Gaussian did not converge near the true sigma"
    assert "regret_loo" in prow and prow["regret_loo"] < 0.08, \
        "shape pipeline far from the floor in the null"
    print("selftest passed")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", default="heavy",
                    choices=["gaussian", "heavy", "skewed", "laplace"])
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
    ap.add_argument("--pipeline", action="store_true",
                    help="end-to-end mode: every arm's density feeds a 4D "
                         "MAP scored against the true-density MAP. Arms: "
                         "gaussM/gaussB (oracle Gaussians, closed form), "
                         "prior (wide-kernel no-iteration baseline), and "
                         "two ITERATED pipelines sharing one "
                         "posterior<->LOO<->estimate loop that differ "
                         "only in what the loop retains: gaussI keeps "
                         "variance, loo keeps the full shape")
    ap.add_argument("--max-iters", type=int, default=4,
                    help="cap on the fixed-point iterations per pipeline")
    ap.add_argument("--iter-tol", type=float, default=0.05,
                    help="stop when successive recovered densities agree "
                         "to this L1 (truth-free)")
    ap.add_argument("--sampler", default="mala",
                    choices=["mala", "pff"],
                    help="posterior sampler for the iterated pipelines; "
                         "pff = componentwise flow with anomaly "
                         "inflation, health-checked every window against "
                         "the analytic iteration-0 Gaussian posterior")
    ap.add_argument("--pff-inflation", type=float, default=1.05)
    ap.add_argument("--assumed-error", type=float, default=0.4)
    ap.add_argument("--windows", type=int, default=0,
                    help="prequential multi-window mode: this many "
                         "re-anchored windows share one LOO archive; "
                         "window w is analyzed under the density from "
                         "windows 1..w-1 (uses the first entry of "
                         "--T-list)")
    ap.add_argument("--archive-mode", default="pooled",
                    choices=["pooled", "regen", "recent"],
                    help="pooled = rows kept as generated (online EM, "
                         "runs away on soft-direction densities); regen "
                         "= regenerate all rows under the current "
                         "density each window (batch EM, O(W^2) "
                         "sampling); recent = sliding memory")
    ap.add_argument("--archive-windows", type=int, default=5,
                    help="memory length for --archive-mode recent")
    ap.add_argument("--center-innovations", action="store_true",
                    help="remove each window's mean innovation before "
                         "archiving (VarBC-style pinning of the offset "
                         "direction; location is only weakly identified "
                         "by DOEE anyway)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.adaptive:
        a.lam = None
    if a.windows:
        return run_windows(a)
    if a.pipeline:
        return run_pipeline(a)

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
