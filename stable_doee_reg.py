"""Regularized variant of the DOEE estimator.

Same deconvolution as `estimate_noise_pmf` in stable_doee: match the innovation
histogram to the convolution of the noise density with the empirical difference
histogram, subject to non-negativity and unit mass. Three changes.

1. PENALTY ON THE THIRD DIFFERENCE OF log pi, not on differences of pi.

   The penalty is sum_i w_i [(L pi)_i]^2 with L the THIRD difference and
   w_i ~ 1/pihat^2 refreshed each solve, which penalizes pi'''/pi, the
   leading term of d^3 log pi / dx^3. It is scale free in pi, so it keeps
   acting in the tails, and its null space contains every log-quadratic
   density: a Gaussian costs nothing, and so does an exponential tail.

   That last property is the point, and it was learned in two steps. The
   original penalizes differences of pi, which prefers flat compact
   densities: measured against known truths it returns an excess kurtosis
   near -0.6 for every case tried, including truths at +7 and +22. The first
   regularized variant penalized the SECOND difference of log pi, which
   vanishes on log-linear (exponential) tails but charges -1/sigma^2 per bin
   on Gaussian ones: on a Gaussian truth with exact analytic histograms its
   kurtosis bias grew monotonically with lambda, +0.3 at 1e-4 to +14.5 at
   1e0 -- a prior, not an artifact. Neither penalty is neutral. The third
   difference vanishes for log-quadratic AND log-linear tails, so it
   privileges neither: same test, +1.0 at lambda 1e-2, +0.8 at 1e-1,
   +1.0 at 1e0.

2. THE SMOOTHING STRENGTH IS CROSS-VALIDATED, not fixed.

   Innovations are split into folds; the density is fitted on the training folds
   and scored by the predictive log-likelihood of the held-out innovations under
   the fitted innovation density Eta @ pi -- the quantity the model actually
   predicts, so no separate criterion is needed.

   The criterion is nearly flat in lambda, because the innovation-space
   likelihood is insensitive to the deconvolved density: the ill-posedness of
   the inverse problem reappearing in the selection step. Selection is therefore
   gated on how well posed the problem is (see below); where it is well posed
   the argmax is taken, and where it is not the usual one-standard-error rule
   picks the smoothest lambda indistinguishable from the best, which stops the
   estimator inventing structure it cannot resolve.

3. RESOLVABILITY IS REPORTED, and a warning is raised when the problem cannot
   be solved by anyone.

   sigma_o^2 = var(innovation) - var(perturbation). When sigma_o is much smaller
   than sigma_b the deconvolution is severely ill posed -- recovering a narrow
   density from a much wider kernel has logarithmic convergence rates -- and no
   choice of penalty rescues it. Measured at sigma_o/sigma_b = 0.4, neither this
   variant nor the original recovers the shape. Better to say so than to return
   a confident wrong answer.

Measured against the original on 8000 observations, 20 members, three seeds per
case, scoring L1 distance to the true density (third-difference penalty):

    case            original   regularized   resolvability   chosen lambda
    gaussian 1.0      0.095       0.099          1.00             1e+01
    heavy 85/15       0.341       0.186          1.02             1e+01
    laplace 0.8       0.302       0.174          1.15             1e+02
    very heavy 95/5   0.386       0.143          0.84             1e+01
    gaussian 0.4      0.428       0.439          0.41  (flagged)  1e+03
    total             1.552       1.041

Every non-Gaussian case improves by roughly a factor of two, and the Gaussian
regression of the second-difference variant (0.134 against 0.093) is gone: on
a Gaussian truth the two estimators are now equivalent and the recovered
excess kurtosis is +0.4 rather than a floor of about +4. Recovered kurtosis is
CONSERVATIVE under this penalty -- +2.8 against a true +7.2, +4.1 against
+21.6, +1.8 against +2.8 -- because the cross-validation picks strong
smoothing; but what it reports now clears the null floor (see
l95-testbed/null_calibration.py): at the laplace configuration (ratio 1.13)
the floor p95 is +0.96 against the recovered +1.8, where the
second-difference floor at ratio 0.8 was +4.3 and Laplace-level tails could
not be claimed below a ratio of about one.

Requires numpy and quadprog. `validate_doee_variants.py` reproduces the table.
"""

import numpy as np
import quadprog


def _grid_and_histograms(X, Y, p_lo=1, p_hi=99, pad_frac=0.05, rng=None):
    """Support, grid and the two histograms, following the original."""
    rng = np.random.default_rng() if rng is None else rng
    x_lo, x_hi = np.percentile(X, [p_lo, p_hi])
    y_lo, y_hi = np.percentile(Y, [p_lo, p_hi])
    n_lo, n_hi = y_lo - x_hi, y_hi - x_lo
    pad = pad_frac * (n_hi - n_lo)
    x_min, x_max = n_lo - pad, n_hi + pad

    _, eY = np.histogram(Y, bins="sqrt")
    _, eX = np.histogram(X, bins="sqrt")
    n = min(len(eY), len(eX))
    if n % 2 == 0:
        n += 1

    x_grid = np.linspace(x_min, x_max, n)
    dx = x_grid[1] - x_grid[0]

    iy = rng.integers(0, len(Y), len(Y))
    ix = rng.integers(0, len(X), len(Y))
    innov = Y[iy] - X[ix]
    f_d, _ = np.histogram(innov, bins=n, range=(x_min, x_max), density=True)

    i1 = rng.integers(0, len(X), len(X))
    i2 = rng.integers(0, len(X), len(X))
    f_k, _ = np.histogram(X[i1] - X[i2], bins=n, range=(x_min, x_max),
                          density=True)
    return x_grid, dx, n, f_d, f_k, innov, (x_min, x_max)


def _conv_matrix(f_k, n, dx, x_min):
    k0 = int(round((0.0 - x_min) / dx))
    Eta = np.zeros((n, n))
    for j in range(n):
        idxs = np.arange(n) - j + k0
        ok = (0 <= idxs) & (idxs < n)
        Eta[ok, j] = f_k[idxs[ok]]
    return Eta


def _third_difference(n):
    L = np.zeros((n - 3, n))
    for i in range(n - 3):
        L[i, i], L[i, i + 1] = -1.0, 3.0
        L[i, i + 2], L[i, i + 3] = -3.0, 1.0
    return L


def histograms_from_ensemble(obs, hofx, n_bins=None, pad_frac=0.05,
                             n_kernel=None, seed=0):
    """Build the innovation histogram and the kernel from a background ensemble,
    pairing at the same location.

    obs  : (n,)   observed values
    hofx : (n, K) member values in observation space

    Returns (grid, f_d, f_k, innov). The innovations are returned as well
    because estimate_from_histograms needs them to cross-validate the smoothing
    strength.

    Two identities are available and only one of them is well conditioned.

    The estimator's own is f_{Y-X} = f_{X1-X2} * f_N with Y the observations and
    X the member values, formed by pairing indices AT RANDOM. That is correct
    when the truth is exchangeable with the members, but random pairing across
    locations leaves the variability of the field in both histograms, and the
    field is far wider than the errors. Measured, that put 98 per cent of the
    variance into the part to be removed and the recovered width was several
    times the truth.

    The identity used here pairs at the same location instead:

        d = obs - member          = eps_o - (member - truth)
        f_d = f_eps_o * f_{-(member - truth)}
        kernel: member_i - member_j at the SAME location, which under
        exchangeability has the law of member - truth, and is symmetric so the
        sign does not matter

    Both histograms are then error-scale and the field cancels before the
    deconvolution sees anything.

    What does NOT work is handing the innovations and the perturbations to
    estimate_noise_pmf as its X and Y. The innovation is already a difference,
    so the estimator differences again and deconvolves d_i - p_j, whose variance
    is sigma_o^2 + 3 sigma_b^2, with a kernel of 2 sigma_b^2: one sigma_b^2 is
    left behind and no amount of data removes it.
    """
    obs = np.asarray(obs, float)
    hofx = np.asarray(hofx, float)
    if hofx.ndim != 2:
        raise ValueError("hofx must be (n_obs, n_members)")
    n, K = hofx.shape
    if K < 2:
        raise ValueError("need at least two members to form the kernel")
    if obs.shape[0] != n:
        raise ValueError("obs and hofx disagree on the number of observations")

    rng = np.random.default_rng(seed)
    innov = (obs[:, None] - hofx).ravel(order="F")

    m = n_kernel or innov.size
    loc = rng.integers(0, n, size=m)
    i = rng.integers(0, K, size=m)
    j = rng.integers(0, K, size=m)
    ok = i != j
    kern = hofx[loc[ok], i[ok]] - hofx[loc[ok], j[ok]]

    lo = min(innov.min(), kern.min())
    hi = max(innov.max(), kern.max())
    pad = pad_frac * (hi - lo)
    lo, hi = lo - pad, hi + pad
    nb = n_bins or (int(np.sqrt(innov.size)) | 1)
    if nb % 2 == 0:
        nb += 1

    grid = np.linspace(lo, hi, nb)
    f_d, _ = np.histogram(innov, bins=nb, range=(lo, hi), density=True)
    f_k, _ = np.histogram(kern, bins=nb, range=(lo, hi), density=True)
    return grid, f_d, f_k, innov


def innovation_groups(n_obs, n_members):
    """Group labels for innovations stacked the way histograms_from_ensemble
    stacks them (member-major, ravel order 'F'): sample i belongs to
    observation i mod n_obs. Pass to estimate_from_histograms as `groups`."""
    return np.tile(np.arange(n_obs), n_members)


def estimate_from_histograms(grid, f_d, f_k, lam=None, lam_grid=None,
                             n_irls=3, trim_log=-30.0, well_posed=0.7,
                             innov=None, folds=4, seed=0, verbose=False,
                             groups=None, lam_flat=1.0e-1):
    """Deconvolve f_k out of f_d on `grid`, returning (x_grid, pi, cache).

    This is the entry point to use when the two histograms have already been
    formed, which is the case whenever the pairing carries information worth
    keeping. `histograms_from_ensemble` builds them from a background ensemble.

    lam is cross-validated when None, which needs `innov`, the innovation
    samples the histogram was built from. When those samples are stacked over
    ensemble members, ALSO pass `groups` (see innovation_groups): each
    observation's error draw appears in every member's innovation, so folds
    split at random leak the draw across the split and understate the fold
    standard error (by 2.6x, measured on the l95 testbed at n=2000, K=20).

    Selection follows three rules in order. When lam_flat cannot be
    distinguished from the best lambda -- its criterion mean within one
    standard error of the maximum -- lam_flat is used: on the l95 testbed the
    held-out innovation likelihood is flat over four decades of lambda (the
    kernel convolution maps very different densities to nearly the same
    innovation fit, the ill-posedness reappearing in the selection step), and
    the argmax is then a noise tilt that lands on the spiky end of the grid.
    lam_flat's default 1e-1 is the value null_calibration measures the
    artifact floor at, so the estimate and the floor it is judged against
    stay one instrument. When the data genuinely favor another lambda by
    more than one standard error, the data win: the argmax is taken where the
    deconvolution is well posed, the one-standard-error fallback where it is
    not.
    """
    grid = np.asarray(grid, float)
    n = grid.size
    dx = grid[1] - grid[0]
    x_min, x_max = grid[0], grid[-1] + dx
    Eta = _conv_matrix(np.asarray(f_k, float), n, dx, x_min)

    ratio = np.nan
    v_d, v_k = _hist_var(grid, f_d), _hist_var(grid, f_k)
    if v_k > 0:
        ratio = float(np.sqrt(max(v_d - v_k, 0.0)) / np.sqrt(v_k / 2.0))

    if lam is None:
        if innov is None:
            lam = 1.0e-1
        else:
            if lam_grid is None:
                lam_grid = np.logspace(-3, 3, 7)
            means, ses = [], []
            for lm in lam_grid:
                try:
                    fs = _cv_fold_scores(Eta, dx, n, x_min, x_max,
                                         np.asarray(innov, float), lm,
                                         folds=folds, seed=seed, n_irls=n_irls,
                                         groups=groups)
                    means.append(float(fs.mean()))
                    ses.append(float(fs.std(ddof=1) / np.sqrt(len(fs))))
                except Exception:
                    means.append(-np.inf)
                    ses.append(0.0)
            means, ses = np.asarray(means), np.asarray(ses)
            best = int(np.argmax(means))
            # Three rules, in order. CALIBRATED DEFAULT: when lam_flat cannot
            # be distinguished from the best lambda -- its criterion mean
            # within one standard error of the maximum -- lam_flat is used.
            # On the l95 testbed the held-out innovation likelihood is flat
            # over four decades of lambda (the kernel convolution maps very
            # different densities to nearly the same innovation fit, the
            # ill-posedness reappearing in the selection step) and the argmax
            # is then a noise tilt that lands on the spiky end of the grid.
            # lam_flat's default 1e-1 is the value null_calibration measures
            # the artifact floor at, so the estimate and the floor it is
            # judged against stay one instrument. When the data genuinely
            # favor another lambda by more than one standard error, the data
            # win: argmax where the deconvolution is well posed, the
            # one-standard-error fallback where it is not.
            i_flat = int(np.argmin(np.abs(np.log10(np.asarray(lam_grid))
                                          - np.log10(lam_flat))))
            if np.isfinite(means[i_flat]) \
                    and means[i_flat] >= means[best] - ses[best]:
                lam, rule = float(lam_flat), "calibrated default"
            elif not np.isfinite(ratio) or ratio >= well_posed:
                lam, rule = float(lam_grid[best]), "argmax"
            else:
                ok = np.where(means >= means[best] - ses[best])[0]
                lam = float(lam_grid[int(ok.max())]) if ok.size \
                    else float(lam_grid[best])
                rule = "one-SE"
            if verbose:
                print(f"    cv best {means[best]:+.4f} (se {ses[best]:.4f}), "
                      f"at lam_flat {means[i_flat]:+.4f}; chosen ({rule}) "
                      f"lambda = {lam:.3e} (resolvability {ratio:.2f})")

    pi = _solve(Eta, np.asarray(f_d, float), dx, n, lam, n_irls=n_irls)
    return _make_cache(grid, pi, dx, lam, ratio, trim_log)


# ---------------------------------------------------------------------------
# Adaptive estimation: whitened data term, resolution-invariant penalty,
# smoothing chosen by the discrepancy principle against a measured noise
# level. The design, derived:
#
#   DATA MODEL. The innovation histogram f_d is regression data for the
#   deconvolution, not a density estimate in its own right, so the bin
#   count is numerics, not statistics: with a correct per-bin noise model
#   nothing is lost by binning finely. Per-bin noise cannot be modeled
#   cleanly (each observation's error draw appears in every member's
#   innovation, so bins are correlated through the kernel width and the
#   effective count sits between n and n*K), so it is MEASURED: for each
#   of `splits` grouped half-splits, ((fA - fB)/2)^2 is an unbiased
#   per-bin sample of Var(f_d) including all correlation effects, and the
#   average over splits is the variance estimate. Structurally empty bins
#   are floored at a small fraction of the median positive variance.
#
#   OBJECTIVE. chi2(pi) = sum_i (dx*(Eta pi)_i - f_d_i)^2 / sig_i^2
#   plus mu * dx^-5 * sum_j w_j (L pi)_j^2 with the same reweighted
#   third-difference-of-log penalty as _solve. The dx^-5 makes the
#   penalty's continuum strength independent of the bin count (the third
#   difference contributes dx^6 per bin over 1/dx bins), so mu means the
#   same thing at every resolution -- the unnormalized form is why more
#   members silently collapsed the smoothing 32x on the record ensemble.
#
#   SELECTION. E[chi2 at the true density] = nb exactly, by construction
#   of the empirical variances, so the discrepancy principle has a
#   parameter-free target: the LARGEST mu with chi2(mu) <= nb. This is
#   the adaptivity the problem demands: nothing about the truth's
#   smoothness is assumed; more data shrinks sig, shrinks the feasible
#   mu, and lets the estimate become as rough as the data can support.
#   If no mu is feasible the model cannot represent the data within its
#   own noise (the over-dispersion signature) and the minimum-chi2 mu is
#   used with chi2_min/nb reported as a consistency ratio.
# ---------------------------------------------------------------------------

def measure_bin_noise(grid, innov, groups, seed=0, splits=8, floor_frac=0.05):
    """Per-bin noise sd of the innovation histogram, measured from grouped
    half-splits. Returns sig of shape (n,)."""
    innov = np.asarray(innov, float)
    grid = np.asarray(grid, float)
    n = grid.size
    dx = grid[1] - grid[0]
    lo, hi = grid[0], grid[-1] + dx
    if groups is None:
        groups = np.arange(innov.size)
    groups = np.asarray(groups)
    uniq0 = np.unique(groups)
    acc = np.zeros(n)
    for m in range(splits):
        rng = np.random.default_rng(seed + 7919 * (m + 1))
        uniq = rng.permutation(uniq0)
        selA = np.isin(groups, uniq[:uniq.size // 2])
        fA, _ = np.histogram(innov[selA], bins=n, range=(lo, hi),
                             density=True)
        fB, _ = np.histogram(innov[~selA], bins=n, range=(lo, hi),
                             density=True)
        acc += 0.25 * (fA - fB) ** 2
    sig2 = acc / splits
    # Sparse-bin floor. With a handful of counts per bin the split
    # estimate can come out near zero by luck (each split difference is a
    # single chi-square draw on discrete counts), and an under-estimated
    # sigma over-weights exactly the bins the model cannot and should not
    # fit -- measured at n=500 obs and 263 bins, the unfittable
    # single-count spikes ate the whole chi-square budget and forced the
    # selection to the smallest mu on the grid. Any bin's variance is at
    # least the all-samples-independent Poisson rate f/(N dx) (computed on
    # a box-smoothed histogram with an additive half count so it stays
    # positive in empty regions), so that is the floor; the measured value
    # keeps the correlation surplus wherever the counts support measuring
    # it. The group-limited rate f/(n_groups dx) was tried as a
    # conservative envelope and is wrong at fine bins -- it assumes a
    # group's members share a bin, over-states the noise 20-50x, and
    # smoothed every recovery flat.
    f_loc, _ = np.histogram(innov, bins=n, range=(lo, hi), density=True)
    box = np.ones(5) / 5.0
    f_s = np.convolve(f_loc, box, mode="same") + 0.5 / (innov.size * dx)
    sig2 = np.maximum(sig2, f_s / (innov.size * dx))
    pos = sig2[sig2 > 0]
    if pos.size:
        sig2 = np.maximum(sig2, floor_frac * np.median(pos))
    else:
        sig2 = np.full(n, 1.0)
    return np.sqrt(sig2)


def _solve_whitened(Eta, f_d, sig, dx, n, mu, n_irls=3, floor=1e-6,
                    ridge=1e-10):
    """The _solve QP with a whitened data term and the dx^-5-normalized
    penalty; see the section comment above. Returns (pi, chi2)."""
    L = _third_difference(n)
    W2 = 1.0 / np.asarray(sig, float) ** 2
    H_data = 2.0 * dx ** 2 * (Eta.T @ (W2[:, None] * Eta))
    a = 2.0 * dx * (Eta.T @ (W2 * f_d))

    C_eq = dx * np.ones((n, 1))
    C = np.hstack([C_eq, np.eye(n)])
    b = np.hstack([1.0, np.zeros(n)])

    pi = np.full(n, 1.0 / (n * dx))
    for _ in range(max(1, n_irls)):
        w = 1.0 / np.maximum(pi, floor) ** 2
        wr = np.exp(np.log(w[:-3] * w[1:-2] * w[2:-1] * w[3:]) / 4.0)
        H = H_data + 2.0 * (mu / dx ** 5) * (L.T @ (wr[:, None] * L)) \
            + ridge * np.eye(n)
        H = 0.5 * (H + H.T)
        ev = np.linalg.eigvalsh(H)
        if ev[0] <= 0:
            H += (abs(ev[0]) + 1e-8) * np.eye(n)
        pi = quadprog.solve_qp(H, a, C, b, meq=1)[0]
        pi = np.maximum(pi, 0.0)
    r = dx * (Eta @ pi) - f_d
    chi2 = float(((r / sig) ** 2).sum())
    return pi, chi2


def adaptive_bin_count(innov, per_scale=30, max_bins=1201):
    """Bin count from the data scale, not the sample count: dx is a robust
    innovation scale over per_scale, so resolution reflects the features
    the density can have rather than how many duplicated samples exist."""
    innov = np.asarray(innov, float)
    s = 1.4826 * np.median(np.abs(innov - np.median(innov)))
    if not s > 0:
        s = max(float(np.std(innov)), 1e-6)
    nb = int(np.ceil((innov.max() - innov.min()) / (s / per_scale))) | 1
    return min(nb, max_bins)


def estimate_adaptive(grid, f_d, f_k, innov, groups, seed=0, n_irls=3,
                      mu_grid=None, splits=8, trim_log=-30.0, verbose=False):
    """Deconvolve with the whitened objective and discrepancy-selected mu.
    Same inputs and cache contract as estimate_from_histograms, plus the
    raw innovations and groups (both required: the noise level is
    measured, not assumed). The cache carries mu in cache['lambda'] and
    the consistency ratio chi2/nb in cache['chi2_ratio']."""
    grid = np.asarray(grid, float)
    f_d = np.asarray(f_d, float)
    n = grid.size
    dx = grid[1] - grid[0]
    Eta = _conv_matrix(np.asarray(f_k, float), n, dx, grid[0])

    ratio = np.nan
    v_d, v_k = _hist_var(grid, f_d), _hist_var(grid, f_k)
    if v_k > 0:
        ratio = float(np.sqrt(max(v_d - v_k, 0.0)) / np.sqrt(v_k / 2.0))

    sig = measure_bin_noise(grid, innov, groups, seed=seed, splits=splits)
    if mu_grid is None:
        mu_grid = np.logspace(-10.0, 2.0, 17)
    chi2 = {}
    best_pi = None
    chosen = None
    for mu in mu_grid:                       # ascending
        try:
            pi, c2 = _solve_whitened(Eta, f_d, sig, dx, n, float(mu),
                                     n_irls=n_irls)
        except Exception:
            continue
        chi2[float(mu)] = c2
        if c2 <= n:
            chosen, best_pi = float(mu), pi   # largest feasible so far
    rule = "discrepancy"
    if chosen is None:
        if not chi2:
            raise RuntimeError("no mu on the grid produced a solution")
        chosen = min(chi2, key=chi2.get)
        best_pi, _ = _solve_whitened(Eta, f_d, sig, dx, n, chosen,
                                     n_irls=n_irls)
        rule = "min-chi2 (model cannot reach the noise level)"
    if verbose:
        pts = "  ".join(f"{m:.0e}:{chi2[m] / n:.2f}"
                        for m in sorted(chi2))
        print(f"    chi2/nb over mu {pts}; chosen ({rule}) mu = "
              f"{chosen:.3e} (nb {n}, resolvability {ratio:.2f})")
    # The cache keeps _make_cache's contract (the contiguous main run, which
    # is what the unimodal DA export can use, with kept_mass honestly
    # reporting that run's fraction), but the RETURNED arrays are the full
    # estimate: a genuinely multimodal recovery must not be silently
    # truncated to its tallest mode -- measured, the truncation cost a
    # bimodal truth its entire second mode and 70% of its variance.
    _, _, cache = _make_cache(grid, best_pi, dx, chosen, ratio, trim_log)
    cache["chi2_ratio"] = chi2[chosen] / n
    return grid, best_pi, cache


def estimate_adaptive_from_ensemble(obs, hofx, seed=0, **kw):
    """Convenience wrapper: histograms at the data-scale bin count, then
    estimate_adaptive. Two histogram passes; the first only supplies the
    innovations for the bin-count rule."""
    _, _, _, innov = histograms_from_ensemble(obs, hofx, seed=seed)
    nb = adaptive_bin_count(innov)
    grid, f_d, f_k, innov = histograms_from_ensemble(obs, hofx, seed=seed,
                                                     n_bins=nb)
    groups = innovation_groups(np.asarray(obs).size,
                               np.asarray(hofx).shape[1])
    return estimate_adaptive(grid, f_d, f_k, innov, groups, seed=seed, **kw)


def _hist_var(grid, f):
    dx = grid[1] - grid[0]
    f = np.asarray(f, float)
    m = float((f * grid).sum() * dx)
    return float((f * (grid - m) ** 2).sum() * dx)


def _make_cache(x_grid, pi, dx, lam, ratio, trim_log):
    """Trim to a stable interior and differentiate the log density.

    Among the contiguous runs of bins with log density above trim_log, keep
    the one carrying the most probability MASS, not the longest. The
    difference matters when the solve fragments: deconvolving with an
    over-dispersed kernel (reliability ratio 1.24, the first full-size l95
    run) demands a sharper solution than any density can provide, the QP
    answers with a picket fence of spikes separated by gaps below the
    threshold, and the longest run is then a smooth, nearly massless
    penalty ramp out in a tail -- the run selected held 2.7% of the mass
    and put the argmax at -2.6 for a truth with mode zero. The most-mass
    rule keeps the bulk; kept_mass in the cache says how fragmented the
    solution was, and anything well below one means the estimate should
    not be used regardless of which run was kept."""
    logp = np.log(pi + 1e-300)
    mask = logp >= trim_log
    padded = np.r_[0, mask, 0]
    d = np.diff(padded.astype(int))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    k = int(np.argmax([pi[s:e].sum() for s, e in zip(starts, ends)]))
    xg, pin = x_grid[starts[k]:ends[k]], pi[starts[k]:ends[k]]

    logp = np.log(pin + 1e-300)
    slopes = np.empty_like(logp)
    slopes[0] = (logp[1] - logp[0]) / dx
    slopes[-1] = (logp[-1] - logp[-2]) / dx
    slopes[1:-1] = (logp[2:] - logp[:-2]) / (2 * dx)
    intercepts = logp - slopes * xg
    cache = {"dx": dx, "stable_min": xg[0], "stable_max": xg[-1],
             "slopes_log": slopes, "intercepts_log": intercepts,
             "left_log_slope": slopes[0], "left_log_int": intercepts[0],
             "left_dd": (slopes[1] - slopes[0]) / dx,
             "right_log_slope": slopes[-1], "right_log_int": intercepts[-1],
             "right_dd": (slopes[-1] - slopes[-2]) / dx,
             "lambda": lam, "resolvability": ratio,
             "kept_mass": float(pin.sum() * dx)}
    return xg, pin, cache


def _solve(Eta, f_d, dx, n, lam, n_irls=3, floor=1e-6, ridge=1e-10):
    """QP with a reweighted third-difference-of-log penalty.

    The penalty is

        lam * sum_i w_i [(L pi)_i]^2,
        w_i = (pihat_i pihat_i+1 pihat_i+2 pihat_i+3)^(-1/2)

    with L the THIRD difference and the weights refreshed from the previous
    solve. This penalizes pi''' / pi, the leading term of

        d^3 log pi / dx^3 = pi'''/pi - 3 (pi''/pi)(pi'/pi) + 2 (pi'/pi)^3

    so it is scale free in pi (it keeps acting in the tails, where a penalty on
    differences of pi has almost no effect), and its null space contains every
    log-QUADRATIC density. That is the point of the third difference: the
    second-difference version of this penalty vanishes on log-linear
    (exponential) tails but charges -1/sigma^2 per bin on Gaussian ones, so
    minimizing it pushed every recovery toward exponential tails -- measured on
    a Gaussian truth with exact analytic histograms its excess-kurtosis bias
    grew monotonically with lambda (+0.3 at 1e-4 up to +14.5 at 1e0), the
    signature of a prior rather than a numerical artifact. The third difference
    vanishes for log-quadratic AND log-linear tails, so it privileges neither:
    same exact-histogram test, +1.0 at lambda 1e-2, +0.8 at 1e-1, +1.0 at 1e0,
    while still detecting a genuine heavy tail. It degrades below about
    lambda = 1e-3 (under-regularized and spiky), which the cross-validation
    grid respects by starting at 1e-3.

    The dropped lower-order terms make this an approximation to the third
    derivative of log pi rather than that derivative exactly. Linearizing
    log pi about the previous iterate gives the exact quadratic form, but
    diag(1/pihat) then carries entries as large as 1/floor and the constant
    term carries log(pihat), and the resulting QP is ill conditioned: tested
    on the second-difference form, it was worse on a Gaussian truth (L1 0.23
    against 0.13) and quadprog reported inconsistent constraints on a narrow
    one. The approximation is used because it works.
    """
    L = _third_difference(n)
    H_data = 2.0 * dx ** 2 * (Eta.T @ Eta)
    f_data = -2.0 * dx * (Eta.T @ f_d)
    a = -f_data

    C_eq = dx * np.ones((n, 1))
    C = np.hstack([C_eq, np.eye(n)])
    b = np.hstack([1.0, np.zeros(n)])

    pi = np.full(n, 1.0 / (n * dx))            # flat start
    for _ in range(max(1, n_irls)):
        w = 1.0 / np.maximum(pi, floor) ** 2
        # geometric mean over the four bins each row of L touches, so the
        # weight itself is smooth
        wr = np.exp(np.log(w[:-3] * w[1:-2] * w[2:-1] * w[3:]) / 4.0)
        H = H_data + 2.0 * lam * (L.T @ (wr[:, None] * L)) + ridge * np.eye(n)
        H = 0.5 * (H + H.T)                    # symmetrize for quadprog
        ev = np.linalg.eigvalsh(H)
        if ev[0] <= 0:
            H += (abs(ev[0]) + 1e-8) * np.eye(n)
        pi = quadprog.solve_qp(H, a, C, b, meq=1)[0]
        pi = np.maximum(pi, 0.0)
    return pi


def _cv_fold_scores(Eta, dx, n, x_min, x_max, innov, lam, folds=4, seed=0,
                    n_irls=3, groups=None):
    """Per-fold predictive log-likelihood of held-out innovations under
    Eta @ pi, returned separately so a standard error can be formed.

    groups assigns each innovation sample to an observation. When innovations
    are stacked over ensemble members, every observation's error draw appears
    in K samples; random folds then place the same draw on both sides of the
    split, and the criterion rewards a spiky fit that memorizes the draws.
    Grouped folds keep all of an observation's samples in one fold, so the
    held-out likelihood measures generalization to unseen observations."""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.permutation(len(innov))
        parts = np.array_split(idx, folds)
    else:
        groups = np.asarray(groups)
        if groups.shape[0] != len(innov):
            raise ValueError("groups must have one entry per innovation "
                             f"sample ({groups.shape[0]} != {len(innov)})")
        uniq = rng.permutation(np.unique(groups))
        gparts = np.array_split(uniq, folds)
        parts = [np.where(np.isin(groups, gp))[0] for gp in gparts]
    out = []
    for f in range(folds):
        test = parts[f]
        train = np.concatenate([parts[g] for g in range(folds) if g != f])
        f_tr, _ = np.histogram(innov[train], bins=n, range=(x_min, x_max),
                               density=True)
        pi = _solve(Eta, f_tr, dx, n, lam, n_irls=n_irls)
        pred = np.maximum(Eta @ pi, 1e-300)         # fitted innovation density
        pred /= pred.sum() * dx
        bins = np.clip(((innov[test] - x_min) / dx).astype(int), 0, n - 1)
        out.append(float(np.mean(np.log(pred[bins]))))
    return np.asarray(out)


def resolvability(X, Y, n_members):
    """How much of the innovation is observation error rather than background
    spread. Returns (sigma_o / sigma_b, sigma_o).

    Perturbations are taken about the ensemble mean, so their variance is
    sigma_b^2 (K-1)/K, not sigma_b^2. Without the correction sigma_b comes out
    low and sigma_o high, by about 2.5% at K=20 and more for smaller ensembles.

        sigma_o^2 = var(Y) - K/(K-1) var(X)

    Below about 0.5 the deconvolution is ill posed and the recovered shape
    should not be trusted, whichever estimator is used: no amount of data or
    regularization resolves a narrow density from a much wider kernel.
    """
    if n_members < 2:
        raise ValueError("need at least two members to estimate the spread")
    v_innov = float(np.var(Y))                       # sigma_o^2 + sigma_b^2
    v_b = float(np.var(X)) * n_members / (n_members - 1.0)
    v_noise = max(v_innov - v_b, 0.0)
    sb = np.sqrt(max(v_b, 1e-300))
    return float(np.sqrt(v_noise) / sb), float(np.sqrt(v_noise))


def estimate_noise_pmf_reg(X, Y, n_members, lam=None, lam_grid=None, folds=4,
                           n_irls=3, p_lo=1, p_hi=99, pad_frac=0.05,
                           trim_log=-30.0, seed=0, verbose=False,
                           well_posed=0.7):
    """Regularized DOEE. Returns (x_grid, pi_N, cache) like the original, with
    'lambda' and 'resolvability' added to the cache.

    X   : ensemble perturbations   H(x_k) - H(x_bar), stacked over members
    Y   : ensemble innovations     y - H(x_k), stacked over members
    n_members : K, needed to correct the perturbation variance for the mean
    lam : smoothing strength; cross-validated when None
    trim_log : log-density threshold for the stable interior, far below the
               original -20 because the log-curvature penalty keeps the tails
               populated and trimming them is what removed them before
    well_posed : resolvability above which the cross-validation argmax is
               trusted rather than the one-standard-error fallback
    """
    rng = np.random.default_rng(seed)
    x_grid, dx, n, f_d, f_k, innov, (x_min, x_max) = _grid_and_histograms(
        X, Y, p_lo, p_hi, pad_frac, rng)
    Eta = _conv_matrix(f_k, n, dx, x_min)

    ratio, sig_o_hat = resolvability(X, Y, n_members)
    if verbose:
        print(f"    resolvability sigma_o/sigma_b = {ratio:.2f} "
              f"(sigma_o ~ {sig_o_hat:.3f})")

    if lam is None:
        if lam_grid is None:
            lam_grid = np.logspace(-3, 3, 7)
        means, ses = [], []
        for lm in lam_grid:
            try:
                fs = _cv_fold_scores(Eta, dx, n, x_min, x_max, innov, lm,
                                     folds=folds, seed=seed, n_irls=n_irls)
                means.append(float(fs.mean()))
                ses.append(float(fs.std(ddof=1) / np.sqrt(len(fs))))
            except Exception:
                means.append(-np.inf)
                ses.append(0.0)
            if verbose:
                print(f"    lambda={lm:9.2e}  cv loglik={means[-1]:+.6f}"
                      f"  se={ses[-1]:.6f}")
        means = np.asarray(means)
        best = int(np.argmax(means))
        # Selection is gated on how well posed the deconvolution is. The
        # criterion is nearly flat in lambda because the innovation-space
        # likelihood is insensitive to the deconvolved density. Where the noise
        # is an appreciable part of the innovation the argmax is informative and
        # taking it preserves the tails; where it is not, the argmax
        # under-smooths and invents structure, so fall back to the smoothest
        # value statistically indistinguishable from the best.
        if ratio >= well_posed:
            lam = float(lam_grid[best])
            rule = "argmax"
        else:
            thresh = means[best] - ses[best]
            ok = np.where(means >= thresh)[0]
            lam = float(lam_grid[int(ok.max())]) if ok.size \
                else float(lam_grid[best])
            rule = "one-SE"
        if verbose:
            print(f"    argmax lambda = {lam_grid[best]:.3e}; "
                  f"chosen ({rule}) = {lam:.3e}")

    pi = _solve(Eta, f_d, dx, n, lam, n_irls=n_irls)

    # trim to a numerically stable interior, far less aggressively than before
    logp = np.log(pi + 1e-300)
    mask = logp >= trim_log
    padded = np.r_[0, mask, 0]
    d = np.diff(padded.astype(int))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    # most-mass run, matching _make_cache: see its docstring for the measured
    # failure the longest-run rule produced
    k = int(np.argmax([pi[s:e].sum() for s, e in zip(starts, ends)]))
    i_lo, i_hi = starts[k], ends[k]
    xg, pin = x_grid[i_lo:i_hi], pi[i_lo:i_hi]

    logp = np.log(pin + 1e-300)
    slopes = np.empty_like(logp)
    slopes[0] = (logp[1] - logp[0]) / dx
    slopes[-1] = (logp[-1] - logp[-2]) / dx
    slopes[1:-1] = (logp[2:] - logp[:-2]) / (2 * dx)
    intercepts = logp - slopes * xg
    cache = {"dx": dx, "stable_min": xg[0], "stable_max": xg[-1],
             "slopes_log": slopes, "intercepts_log": intercepts,
             "left_log_slope": slopes[0], "left_log_int": intercepts[0],
             "left_dd": (slopes[1] - slopes[0]) / dx,
             "right_log_slope": slopes[-1], "right_log_int": intercepts[-1],
             "right_dd": (slopes[-1] - slopes[-2]) / dx,
             "lambda": lam, "resolvability": ratio,
             "kept_mass": float(pin.sum() * dx)}
    if ratio < 0.5:
        import warnings as _w
        _w.warn(f"observation error is only {ratio:.2f} of the background "
                "spread; the deconvolution is ill posed in this regime and the "
                "recovered shape should not be trusted. Stratify to reduce the "
                "background spread, or take the variance estimate only.",
                RuntimeWarning)
    return xg, pin, cache
