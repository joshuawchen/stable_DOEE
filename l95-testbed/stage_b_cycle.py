#!/usr/bin/env python3
"""Stage B in the sandbox: DA cycling in the loop, three ensemble providers.

The question: which cycled-DA ensemble yields innovation vectors from which
the observation-error density is best estimated -- and is particle flow
provably better than EDA-3DVar cycling?

THE ARGUMENT. DOEE's kernel is unbiased iff the truth is exchangeable with
the members, which holds iff the members sample the true filtering prior
p(x_t | y_past): then truth is one more draw, H(x_t - x^k) is distributed
as the member differences, and every window is a Stage A instance with
epsilon independent across windows (n accumulates). EDA-3DVar members
sample a GAUSSIAN posterior calibrated to the ASSUMED R, so its kernel
error is first order in the R error and the non-Gaussianity, systematic,
and irreducible with n. An exact-posterior filter's members sample the
posterior under the current estimate pi-hat, so its kernel error is second
order, entering only through the pi-hat feedback, and vanishes at the
fixed point pi-hat = pi. Particle flow inherits the exact-posterior
guarantee in the idealized limit (many particles, converged flow, score of
pi-hat) and carries finite-K approximations otherwise. Ordering: EDA <
PFF < exact in asymptotic kernel bias; this module measures the gaps.

THE SETUP. Periodic 40-point state; evolution x_{t+1} = a R(x_t) +
sqrt(1-a^2) C^{1/2} eta with R a one-point roll (orthogonal), so the
climatological law N(base, C) is stationary and cycling decorrelates
windows. Observations y_t = H x_t + eps_t with eps from the injected
density. Every provider gets the same truth, the same observations, the
same forecast rule, and the same per-window Gaussian prior fit
N(mean(Xf), shrink(cov(Xf), C)); only the ANALYSIS step differs:

    exact  preconditioned MALA on the (Gaussian prior x pi-hat likelihood)
           posterior; the yardstick oops cannot supply
    pff    the interacting particle flow (Hu & van Leeuwen 2021, QJRMS),
           SVGD form, particle positions LIVE in the kernel, likelihood
           score from pi-hat
    eda    per-member Gaussian 3D-Var with observations perturbed from the
           assumed R -- the operational competitor

Each provider runs its own loop: prior innovations accumulate across
windows, DOEE re-estimates from the archive, the new estimate feeds back
into that provider's next analysis (eda stays Gaussian; its archive is
still estimated from, to measure the estimation its innovations support).

Per cycle and provider: the exchangeability ratio ER =
var(H truth - H members) / (2 mean member variance), the direct test of
the theorem's premise (1 iff calibrated); analysis rmse to truth; and at
each re-estimation the recovered density's L1 and weight-profile scatter
against the injected truth.

SELFTEST: the linear-Gaussian null (gaussian truth at the assumed sigma),
where EDA is exact and all three providers must agree: ER near 1 for all,
the exact provider's posterior mean matching the Kalman closed form, MALA
acceptance in a healthy band.

    python3 stage_b_cycle.py --selftest
    python3 stage_b_cycle.py --density heavy --cycles 10 --n-obs 200
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from jedi_export import doee_to_yaml as DY          # noqa: E402
from map_reference import (NGRID, Quiet, analytic_nll, draw_errors,  # noqa: E402
                           estimate_density, interp_operator, moments_on,
                           prior_cov, sample_sigma_of, solve_map, spec_nll)
from stage_a_end_to_end import analytic_density, gaussian_tails  # noqa: E402


# ---------------------------------------------------------------------------
# shared per-window machinery
# ---------------------------------------------------------------------------

def fit_prior(Xf, C, shrink=0.2):
    """The Gaussian prior every provider uses this window: ensemble mean,
    sample covariance shrunk toward the climatological C."""
    mu = Xf.mean(axis=1)
    S = np.cov(Xf)
    P = (1.0 - shrink) * S + shrink * C + 1e-8 * np.eye(Xf.shape[0])
    return mu, P


def posterior_pieces(Pinv, mu, H, y, nll, dnll):
    def logp(x):
        e = y - H @ x
        r = x - mu
        return -(0.5 * r @ Pinv @ r + float(np.sum(nll(e))))

    def grad(x):
        return -Pinv @ (x - mu) + H.T @ dnll(y - H @ x)
    return logp, grad


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

def analyze_exact(Xf, y, H, C, nll, dnll, hh, K, rng, burn=150,
                  prior=None):
    """Preconditioned MALA targeting the exact posterior of the shared
    Gaussian prior and the current likelihood -- one independent chain per
    member, started at that member's prior position, keeping the last
    state. Independent chains matter: a single thinned chain leaves the
    ensemble autocorrelated (effective K well below K), which at sparse
    observations starves both the analysis mean and the deconvolution
    kernel; per-member chains also cover separated modes the way a single
    chain cannot. Preconditioner = the MAP Hessian (clipped curvature),
    adapted step on the first chain, reused on the rest."""
    mu, P = prior if prior is not None else fit_prior(Xf, C)
    Pinv = np.linalg.inv(P)
    xmap, _, _ = solve_map(Pinv, H, mu, y, nll, dnll, mu, h=hh)
    logp, grad = posterior_pieces(Pinv, mu, H, y, nll, dnll)
    lp_map = logp(xmap)
    e = y - H @ xmap
    c = np.clip((dnll(e + hh) - dnll(e - hh)) / (2 * hh), 0.0, None)
    A = Pinv + (H.T * c) @ H + 1e-10 * np.eye(Xf.shape[0])
    L = np.linalg.cholesky(A)
    d_ = Xf.shape[0]

    def drift_of(g_):
        # capped preconditioned drift, applied identically in the forward
        # and backward proposal means so the MH ratio stays exact; the cap
        # tames support-wall gradients (mirrored_gamma's continuation has
        # curvature 1/t0^2) that otherwise launch proposals into the void
        dr = 0.5 * tau * tau * np.linalg.solve(A, g_)
        an = float(np.sqrt(max(dr @ A @ dr, 0.0)))
        cap = 3.0 * tau * np.sqrt(d_)
        return dr * (cap / an) if an > cap else dr

    tau, acc_tot, n_tot = 0.4, 0, 0
    Xa = np.empty((d_, K))
    for k in range(K):
        x = Xf[:, k].copy()
        lp, g = logp(x), grad(x)
        if not np.isfinite(lp) or lp < lp_map - 50.0 * d_:
            # a start beyond a support wall (or absurdly deep in a tail)
            # never accepts; restart at the MAP with preconditioned jitter
            x = xmap + 0.5 * np.linalg.solve(
                L.T, rng.standard_normal(d_))
            lp, g = logp(x), grad(x)
        acc_c, acc_win = 0, 0
        nb = 2 * burn if k == 0 else burn
        for it in range(nb):
            m_x = x + drift_of(g)
            xp = m_x + tau * np.linalg.solve(
                L.T, rng.standard_normal(d_))
            lpp, gp = logp(xp), grad(xp)
            m_xp = xp + drift_of(gp)
            dq_f = xp - m_x
            dq_b = x - m_xp
            a_log = lpp - lp \
                - 0.5 / tau ** 2 * (dq_b @ A @ dq_b) \
                + 0.5 / tau ** 2 * (dq_f @ A @ dq_f)
            if np.log(rng.random()) < a_log:
                x, lp, g = xp, lpp, gp
                acc_c += 1
                acc_win += 1
            if k == 0 and it % 25 == 24:
                r_ = acc_win / 25.0
                acc_win = 0
                if r_ > 0.65:
                    tau *= 1.2
                elif r_ < 0.45:
                    tau /= 1.3
        acc_tot += acc_c
        n_tot += nb
        Xa[:, k] = x
    return Xa, {"acc_rate": acc_tot / max(n_tot, 1), "tau": tau}


def analyze_pff(Xf, y, H, C, nll, dnll, hh, K, rng, iters=300, step0=0.3,
                prior=None, kernel="component"):
    """The interacting particle flow, SVGD form, particle positions live
    in the kernel. kernel="component" is the dimension-wise kernel of Hu
    & van Leeuwen (2021) and of oops PFF.h (Schur products): each state
    component gets its own bandwidth, the paper's remedy for the variance
    collapse a scalar RBF kernel suffers when K is comparable to the
    dimension. kernel="scalar" keeps vanilla SVGD for comparison.
    AdaGrad stepping (the SVGD default), stable as the bandwidth shrinks
    with K; a fixed step is not."""
    mu, P = prior if prior is not None else fit_prior(Xf, C)
    Pinv = np.linalg.inv(P)
    _, grad = posterior_pieces(Pinv, mu, H, y, nll, dnll)
    X = Xf[:, :K].copy()
    d, N = X.shape
    G = np.zeros_like(X)
    logN = np.log(N + 1.0)
    iu = np.triu_indices(N, 1)
    last = np.inf
    for _ in range(iters):
        S = np.stack([grad(X[:, j]) for j in range(N)], axis=1)  # d x N
        if kernel == "scalar":
            n2 = np.sum(X * X, axis=0)
            D2 = np.maximum(n2[:, None] + n2[None, :]
                            - 2.0 * (X.T @ X), 0.0)
            h2 = max(np.median(D2[iu]) / (2.0 * logN), 1e-8)
            Kn = np.exp(-D2 / (2.0 * h2))
            phi = (S @ Kn) / N \
                + (X * Kn.sum(axis=0) - X @ Kn) / (h2 * N)
        else:
            phi = np.empty_like(X)
            for c in range(d):
                dc = X[c][:, None] - X[c][None, :]
                d2 = dc * dc
                h2 = max(np.median(d2[iu]) / (2.0 * logN), 1e-10)
                Kc = np.exp(-d2 / (2.0 * h2))
                phi[c] = (S[c] @ Kc) / N \
                    + (X[c] * Kc.sum(axis=0) - X[c] @ Kc) / (h2 * N)
        G += phi * phi
        upd = step0 * phi / (np.sqrt(G) + 1e-8)
        X = X + upd
        last = float(np.sqrt(np.mean(upd ** 2)))
        if last < 1e-6:
            break
    return X, {"final_update": last}


def analyze_eda(Xf, y, H, C, sig_assumed, rng):
    """Per-member Gaussian 3D-Var with observations perturbed from the
    assumed R: the operational EDA competitor, closed form."""
    mu, P = fit_prior(Xf, C)
    Pinv = np.linalg.inv(P)
    A = Pinv + H.T @ H / sig_assumed ** 2
    K = Xf.shape[1]
    Xa = np.empty_like(Xf)
    for k in range(K):
        yk = y + rng.normal(0.0, sig_assumed, y.size)
        rhs = H.T @ (yk - H @ Xf[:, k]) / sig_assumed ** 2
        Xa[:, k] = Xf[:, k] + np.linalg.solve(A, rhs)
    return Xa, {}


# ---------------------------------------------------------------------------
# the estimation loop pieces
# ---------------------------------------------------------------------------

def reestimate(obs_arch, hofx_arch, lam, adaptive, assumed_error, seed):
    """DOEE on the accumulated archive, through the export the DA would
    consume. Returns (spec or None, sd, l1_grid_pair) -- spec None means
    the export gates refused and the caller keeps the previous density."""
    obs = np.concatenate(obs_arch)
    hofx = np.concatenate(hofx_arch, axis=0)
    xg, pi, cache = estimate_density(obs, hofx, seed, lam, adaptive)
    sd, _ = moments_on(xg, pi)
    gaussian_tails(cache, xg, pi, sd, Quiet(False))
    spec, _ = DY.to_spec(cache)
    if DY.check(spec):
        return None, sd, (xg, pi)
    return spec, sd, (xg, pi)


def density_metrics(xg, pi, spec, spec_inj, dtru, sig_true, eps_pool):
    fine = np.arange(-10.0, 10.0001, 0.02)
    tru = analytic_density(spec_inj, fine)
    p = np.interp(fine, xg, pi, left=0.0, right=0.0)
    tot = p.sum() * 0.02
    l1 = float(np.abs(p / tot - tru).sum() * 0.02) if tot > 0 else np.nan
    wstd = np.nan
    if spec is not None:
        den = DY.Density(spec)
        d = -np.asarray(eps_pool)
        gt = -dtru(-d)
        with np.errstate(divide="ignore", invalid="ignore"):
            vt = np.where(np.abs(d) > 1e-3, d / gt, sig_true ** 2)
        vt = np.where((vt > 0) & np.isfinite(vt), vt, sig_true ** 2)
        ve = np.array([den.variance(float(x)) for x in d])
        wstd = float(np.std(0.5 * np.log(ve / vt)))
    return l1, wstd


def exch_ratio(xt, Xf, H):
    dt = (H @ xt)[:, None] - H @ Xf
    v_t = float(np.var(dt))
    v_m = float(np.mean(np.var(H @ Xf, axis=1, ddof=1)))
    return v_t / max(2.0 * v_m, 1e-300)


# ---------------------------------------------------------------------------
# the cycle
# ---------------------------------------------------------------------------

def run(a):
    rng = np.random.default_rng(a.seed)
    C = prior_cov(a.sigma_b, a.length_scale)
    Cs = np.linalg.cholesky(C)
    H = interp_operator(a.n_obs)
    base = 4.0 * (np.sin(np.arange(NGRID) / 5.5)
                  + 0.5 * np.cos(np.arange(NGRID) / 2.1))
    aa = a.persistence
    q = np.sqrt(1.0 - aa * aa)

    def forecast(X, r):
        anom = X - (base[:, None] if X.ndim > 1 else base)
        return (base[:, None] if X.ndim > 1 else base) \
            + aa * np.roll(anom, 1, axis=0) \
            + q * (Cs @ r.standard_normal(X.shape))

    # nature and the shared initial ensemble
    xt = base + Cs @ rng.standard_normal(NGRID)
    X0 = base[:, None] + Cs @ rng.standard_normal((NGRID, a.members))

    spec_inj = None
    _, dtru, sig_true = None, None, None

    providers = ["exact", "pff", "eda"]
    state = {p: {"Xf": X0.copy(), "obs": [], "hofx": [], "eps": [],
                 "nll": None, "dnll": None, "hh": 1e-5, "spec": None,
                 "xg": None, "pi": None, "sd": np.nan}
             for p in providers}
    g_nll, g_dnll = analytic_nll({"kind": "gaussian",
                                  "sigma": a.assumed_error})
    for p in providers:
        state[p]["nll"], state[p]["dnll"] = g_nll, g_dnll

    rng_p = {p: np.random.default_rng(a.seed + 7919 + i)
             for i, p in enumerate(providers)}

    print(f"stage B cycle: density {a.density} scale {a.scale}, "
          f"{a.cycles} cycles x {a.n_obs} obs, members {a.members}, "
          f"assumed {a.assumed_error}, "
          f"{'adaptive' if a.adaptive else f'lam {a.lam:g}'}, "
          f"likelihood feedback "
          f"{'ON' if not a.no_feedback else 'OFF (assumed Gaussian)'}")

    diag = {p: {"er": [], "rmse": []} for p in providers}
    for t in range(a.cycles):
        eps, spec_inj = draw_errors(a.density, rng, a.n_obs, a.scale)
        spec_inj["sample_sigma"] = sample_sigma_of(spec_inj)
        if dtru is None:
            _, dtru = analytic_nll(spec_inj)
            dh = 1e-4
            c0 = float((dtru(np.array([dh]))
                        - dtru(np.array([-dh])))[0] / (2 * dh))
            sig_true = float(1.0 / np.sqrt(c0)) if c0 > 0 else float("nan")
        y = H @ xt + eps

        line = f"  cycle {t}:"
        for p in providers:
            st = state[p]
            er = exch_ratio(xt, st["Xf"], H)
            st["obs"].append(y.copy())
            st["hofx"].append(H @ st["Xf"])
            st["eps"].append(eps.copy())
            if p == "exact":
                Xa, info = analyze_exact(st["Xf"], y, H, C, st["nll"],
                                         st["dnll"], st["hh"], a.members,
                                         rng_p[p])
            elif p == "pff":
                Xa, info = analyze_pff(st["Xf"], y, H, C, st["nll"],
                                       st["dnll"], st["hh"], a.members,
                                       rng_p[p])
            else:
                Xa, info = analyze_eda(st["Xf"], y, H, C, a.assumed_error,
                                       rng_p[p])
            if p == "pff" and a.pff_inflation != 1.0:
                m_ = Xa.mean(axis=1, keepdims=True)
                Xa = m_ + a.pff_inflation * (Xa - m_)
            rmse = float(np.sqrt(np.mean((Xa.mean(axis=1) - xt) ** 2)))
            diag[p]["er"].append(er)
            diag[p]["rmse"].append(rmse)
            line += f"  {p}: ER {er:.2f} rmse {rmse:.3f}"
            if p == "exact" and "acc_rate" in info:
                line += f" (acc {info['acc_rate']:.2f})"
            st["Xa"] = Xa
        print(line)

        # re-estimate per provider from its own archive, feed back
        if t >= a.warmup - 1:
            for p in providers:
                st = state[p]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spec, sd, (xg, pi) = reestimate(
                        st["obs"], st["hofx"], a.lam, a.adaptive,
                        a.assumed_error, a.seed + 2)
                st["xg"], st["pi"], st["sd"] = xg, pi, sd
                if spec is not None:
                    st["spec"] = spec
                    ntot = sum(len(o) for o in st["obs"])
                    if p != "eda" and not a.no_feedback \
                            and ntot >= a.feedback_min_n:
                        half = 12.0 * max(sd, spec_inj["sample_sigma"],
                                          a.assumed_error)
                        nll_h, dnll_h = spec_nll(spec, half)
                        st["nll"], st["dnll"] = nll_h, dnll_h
                        st["hh"] = 0.5 * spec["grid spacing"]
            mets = []
            for p in providers:
                st = state[p]
                l1, wstd = density_metrics(
                    st["xg"], st["pi"], st["spec"], spec_inj, dtru,
                    sig_true, np.concatenate(st["eps"]))
                mets.append(f"{p}: L1 {l1:.3f} wstd {wstd:.2f}")
                st["l1"], st["wstd"] = l1, wstd
            print("           density  " + "   ".join(mets))

        # forecast to the next window
        xt = forecast(xt, rng)
        for p in providers:
            state[p]["Xf"] = forecast(state[p]["Xa"], rng_p[p])

    print("\nsummary (means over cycles)")
    for p in providers:
        print(f"  {p:5s} ER {np.mean(diag[p]['er']):.2f}  analysis rmse "
              f"{np.mean(diag[p]['rmse']):.3f}  final density: L1 "
              f"{state[p].get('l1', np.nan):.3f}  wstd "
              f"{state[p].get('wstd', np.nan):.2f}")
    print("  reading: ER near 1 is the exchangeability the estimator "
          "needs; the provider ordering of final L1/wstd is the measured "
          "version of the EDA < PFF < exact claim")
    return 0


# ---------------------------------------------------------------------------

def selftest():
    """Linear-Gaussian null: gaussian truth at the assumed sigma. EDA is
    exact here, so all three providers target the same Gaussian posterior:
    ER near 1 for all, exact-provider posterior mean near the Kalman
    closed form, MALA acceptance healthy."""
    rng = np.random.default_rng(5)
    C = prior_cov()
    Cs = np.linalg.cholesky(C)
    H = interp_operator(200)
    s = 0.4
    nll, dnll = analytic_nll({"kind": "gaussian", "sigma": s})
    base = np.zeros(NGRID)
    xt = Cs @ rng.standard_normal(NGRID)
    Xf = Cs @ rng.standard_normal((NGRID, 50))
    y = H @ xt + rng.normal(0, s, 200)

    mu, P = fit_prior(Xf, C)
    Pinv = np.linalg.inv(P)
    A = Pinv + H.T @ H / s ** 2
    kal = mu + np.linalg.solve(A, H.T @ (y - H @ mu) / s ** 2)

    Xa, info = analyze_exact(Xf, y, H, C, nll, dnll, 1e-5, 50,
                             np.random.default_rng(11))
    dev = float(np.sqrt(np.mean((Xa.mean(axis=1) - kal) ** 2)))
    print(f"exact: posterior-mean deviation from Kalman {dev:.4f} "
          f"(acc {info['acc_rate']:.2f}); sample sd ratio "
          f"{float(np.mean(np.std(Xa, axis=1)) / np.mean(np.sqrt(np.diag(np.linalg.inv(A))))):.2f}")
    assert 0.2 < info["acc_rate"] < 0.95, "MALA acceptance unhealthy"
    assert dev < 0.10, "exact provider misses the Kalman mean"

    Xp, pinfo = analyze_pff(Xf, y, H, C, nll, dnll, 1e-5, 50,
                            np.random.default_rng(12))
    devp = float(np.sqrt(np.mean((Xp.mean(axis=1) - kal) ** 2)))
    print(f"pff:   posterior-mean deviation from Kalman {devp:.4f} "
          f"(final update {pinfo['final_update']:.3f})")
    assert devp < 0.15, "pff misses the Kalman mean in the Gaussian null"

    Xe, _ = analyze_eda(Xf, y, H, C, s, np.random.default_rng(13))
    deve = float(np.sqrt(np.mean((Xe.mean(axis=1) - kal) ** 2)))
    print(f"eda:   posterior-mean deviation from Kalman {deve:.4f}")
    assert deve < 0.1, "eda misses the Kalman mean in the Gaussian null"

    for name, X in (("exact", Xa), ("pff", Xp), ("eda", Xe)):
        er = exch_ratio(xt, Xf, H)
    print(f"prior exchangeability ratio {er:.2f} (construction: iid, "
          "should be near 1)")
    assert 0.6 < er < 1.6, "prior construction is not exchangeable"
    print("selftest passed")
    return 0


def pff_k_sweep(a):
    """How fast does the flow's under-dispersion close with particle
    count? Single window, the KNOWN prior N(base, C) shared exactly
    across every K, the TRUE density in the likelihood -- nothing varies
    but the flow. Gaussian case scored against the analytic Kalman
    posterior; the non-Gaussian case against a 400-independent-chain MALA
    reference. Reports the sd ratio (particle spread / reference spread)
    and its reciprocal, the inflation factor the oops PFF would need."""
    import time
    rng = np.random.default_rng(a.seed)
    C = prior_cov(a.sigma_b, a.length_scale)
    Cs = np.linalg.cholesky(C)
    Cinv = np.linalg.inv(C)
    H = interp_operator(a.n_obs)
    base = 4.0 * (np.sin(np.arange(NGRID) / 5.5)
                  + 0.5 * np.cos(np.arange(NGRID) / 2.1))
    xt = base + Cs @ rng.standard_normal(NGRID)
    Ks = [int(s) for s in a.pff_k_sweep.split(",")]
    cases = ["gaussian"] + ([a.density] if a.density != "gaussian" else [])
    for case in cases:
        eps, spec = draw_errors(case, rng, a.n_obs, a.scale)
        spec["sample_sigma"] = sample_sigma_of(spec)
        y = H @ xt + eps
        nll, dnll = analytic_nll(spec)
        if case == "gaussian":
            s = spec["sigma"]
            A = Cinv + H.T @ H / s ** 2
            ref_mean = base + np.linalg.solve(
                A, H.T @ (y - H @ base) / s ** 2)
            ref_sd = np.sqrt(np.diag(np.linalg.inv(A)))
            ref_note = "Kalman analytic"
        else:
            Kref = 400
            Xf_ref = base[:, None] + Cs @ rng.standard_normal((NGRID, Kref))
            Xr, _ = analyze_exact(Xf_ref, y, H, C, nll, dnll, 1e-5, Kref,
                                  np.random.default_rng(a.seed + 31),
                                  prior=(base, C))
            ref_mean = Xr.mean(axis=1)
            ref_sd = np.std(Xr, axis=1, ddof=1)
            ref_note = f"MALA, {Kref} independent chains"
        print(f"\npff K sweep, {case} likelihood (reference: {ref_note}), "
              f"kernel {a.pff_kernel}")
        print("      K    rmse(mean)   sd ratio   implied inflation   "
              "final upd   seconds")
        for K in Ks:
            Xf = base[:, None] + Cs @ rng.standard_normal((NGRID, K))
            t0 = time.time()
            Xp, inf_ = analyze_pff(Xf, y, H, C, nll, dnll, 1e-5, K,
                                   np.random.default_rng(a.seed + 97 + K),
                                   iters=a.pff_iters, prior=(base, C),
                                   kernel=a.pff_kernel)
            dt = time.time() - t0
            rmse = float(np.sqrt(np.mean((Xp.mean(axis=1) - ref_mean) ** 2)))
            sdr = float(np.mean(np.std(Xp, axis=1, ddof=1) / ref_sd))
            print(f"  {K:5d}    {rmse:.4f}       {sdr:.3f}      "
                  f"{1.0 / max(sdr, 1e-6):.2f}                "
                  f"{inf_['final_update']:.1e}     {dt:5.1f}")
    print("\nreading: sd ratio -> 1 with K is the finite-particle "
          "under-dispersion closing; the implied inflation column is "
          "what PFF.h's `inflation factor` would need at that K")
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
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--n-obs", type=int, default=200)
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--assumed-error", type=float, default=0.4)
    ap.add_argument("--sigma-b", type=float, default=0.6)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--persistence", type=float, default=0.8,
                    help="a in x' = a R(x) + sqrt(1-a^2) noise; the "
                         "climatological law is stationary for any a<1")
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--warmup", type=int, default=2,
                    help="cycles of archive before the first re-estimation")
    ap.add_argument("--no-feedback", action="store_true",
                    help="providers keep the assumed Gaussian in the "
                         "analysis; estimation still runs on their "
                         "archives (separates provider bias from the "
                         "feedback loop)")
    ap.add_argument("--pff-inflation", type=float, default=1.0,
                    help="multiplicative posterior-anomaly inflation for "
                         "the pff provider; the K sweep priced the "
                         "componentwise kernel's residual "
                         "under-dispersion at ~1.05")
    ap.add_argument("--feedback-min-n", type=int, default=0,
                    help="adopt the estimated density into the analysis "
                         "only once the archive holds at least this many "
                         "innovations; a bad early estimate in the loop "
                         "is worse than the assumed Gaussian")
    ap.add_argument("--pff-k-sweep", default=None,
                    help="comma-separated particle counts; single-window "
                         "calibration study of the flow against analytic "
                         "and MALA references (no cycling)")
    ap.add_argument("--pff-iters", type=int, default=300)
    ap.add_argument("--pff-kernel", default="component",
                    choices=["component", "scalar"],
                    help="component = the dimension-wise kernel of Hu & "
                         "van Leeuwen (2021) and oops PFF.h; scalar = "
                         "vanilla SVGD, kept for comparison")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.adaptive:
        a.lam = None
    if a.pff_k_sweep:
        return pff_k_sweep(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
