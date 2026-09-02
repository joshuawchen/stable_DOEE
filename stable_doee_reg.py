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
    pi, n_filled = _fill_zero_bins(pi, dx)
    xg, pin, cache = _make_cache(grid, pi, dx, lam, ratio, trim_log)
    cache["zero_bins_filled"] = n_filled
    return xg, pin, cache


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
#   SELECTION. The default is the CROSS-SPLIT one-SE rule: fit on half
#   the observation groups, score the whitened residual against the
#   other half's histogram, symmetrize, and take the largest mu within
#   one standard error of the argmin. The halves carry independent
#   error draws, so the criterion's expectation at the truth is fixed
#   regardless of the effective dof the fit spends and regardless of
#   cross-bin noise correlation -- the two things that break any
#   fitted-residual target. The discrepancy principle against
#   chi2 <= nb - dof(mu) survives as select="discrepancy"; measured, it
#   is dominated (its target is loose by the correlated-block
#   multiplicity under marginal whitening, and under GLS whitening it
#   inherits the covariance estimation noise; see the whiten note in
#   estimate_adaptive). An argmin criterion also removes the
#   over-dispersion failure mode outright: nothing compares chi2 to an
#   absolute height. More data still shrinks the chosen mu -- the
#   adaptivity the problem demands -- but through the criterion's
#   minimum moving, not through a noise-budget bookkeeping.
#
#   IDENTIFIABLE SUPPORT. Finite supports add under convolution, so the
#   support of pi identifiable from data is the observed innovation
#   range ERODED by the kernel reach (Minkowski difference). Bins
#   outside it are uninformed by construction -- the data term cannot
#   distinguish their values and only the penalty fills them -- and they
#   are exactly where spurious tail lobes grew at low n. When enabled
#   they are fixed at zero (columns removed from the QP); the export's
#   tail extrapolation from interior log-slopes carries on unchanged.
#   OFF by default: at finite n the sample range understates supp f_d
#   and full-reach erosion cut genuine tail support (measured L1 0.281
#   -> 0.449 on a heavy truth at identical selection), while the
#   whitened data term already leaves uninformed bins near zero.
#
#   SELECTION SEARCH. The cv criterion is scanned on a coarse log grid
#   and locally refined at the argmin; the discrepancy option uses
#   bisection on log mu for its feasibility crossing. Both replace the
#   former fixed 10^0.75-step grid, which could not express the
#   optimum's motion with n -- theory puts the optimal smoothing at
#   ~ 1/(n log^3 n), which moves by LESS than one grid step over a
#   fourfold change in n, and mu duly pinned at one grid value across
#   every configuration measured.
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


def _split_histograms(grid, innov, groups, seed):
    """One grouped half-split of the innovation histogram: all of an
    observation's samples land on one side, so the two halves carry
    independent error draws. Returns (f_A, f_B)."""
    grid = np.asarray(grid, float)
    n = grid.size
    dx = grid[1] - grid[0]
    lo, hi = grid[0], grid[-1] + dx
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(np.asarray(groups)))
    selA = np.isin(np.asarray(groups), uniq[:uniq.size // 2])
    f_A, _ = np.histogram(innov[selA], bins=n, range=(lo, hi), density=True)
    f_B, _ = np.histogram(innov[~selA], bins=n, range=(lo, hi), density=True)
    return f_A, f_B


def _cross_split_crit(Eta, grid, innov, groups, sig, dx, mu, seed=0,
                      reps=2, n_irls=3, row_W=None):
    """Cross-split whitened predictive chi2 at smoothing mu.

    Fit on one grouped half, score the whitened residual against the
    OTHER half's histogram, symmetrize, average over reps splits. The
    halves carry independent error draws (grouped splits: an
    observation's draw never appears on both sides), so the criterion's
    expectation at the true density is fixed regardless of how many
    effective degrees of freedom the fit spent and regardless of
    cross-bin noise correlation -- the two things that break any
    fitted-residual target like chi2 <= nb (measured: the nb target's
    slack at the good mu ranged 0.55-0.9 across cases, so no fixed
    height works). Under-smoothing memorizes the training half's noise,
    which is wrong for the test half; over-smoothing misfits both:
    the criterion is U-shaped in mu and only its ARGMIN is used, so
    its absolute height never needs calibrating. sig*sqrt(2) whitens
    the half-sized histograms. Returns the 2*reps per-fold scores so the
    caller can form a standard error for the one-SE rule."""
    out = []
    sig_h = np.asarray(sig, float) * np.sqrt(2.0)
    for m in range(reps):
        f_A, f_B = _split_histograms(grid, innov, groups,
                                     seed + 104729 * (m + 1))
        if row_W is not None:
            # GLS: Eta arrives row-whitened by the full-data W, so the
            # halves must live in the same space; W/sqrt(2) whitens the
            # doubled half-data covariance, which is what sig_h does
            f_A, f_B = row_W @ f_A, row_W @ f_B
        for f_tr, f_te in ((f_A, f_B), (f_B, f_A)):
            pi, _, _ = _solve_whitened(Eta, f_tr, sig_h, dx, mu,
                                       n_irls=n_irls)
            r = dx * (Eta @ pi) - f_te
            out.append(float(((r / sig_h) ** 2).sum()))
    return np.asarray(out)


def measure_bin_cov(grid, innov, groups, sig, seed=0, n_boot=192):
    """Per-bin COVARIANCE of the innovation histogram, measured by a
    grouped bootstrap, shrunk toward diag(sig^2). Returns W, the
    symmetric inverse square root of the shrunk covariance, so that
    W @ (f_hat - E f) has identity covariance to estimation accuracy.

    Why the covariance and not just the variances: each observation's
    error draw appears in every member's innovation, so histogram bins
    are strongly CORRELATED, and a chi-square whitened only marginally
    lets one fitted degree of freedom absorb a whole correlated block of
    nominal residual. Measured on heavy 2000x20: the roughest fit spent
    9.6 nominal dof and removed 108 units of marginal chi2 -- about 11
    per dof, the duplication multiplicity -- so the discrepancy target
    nb - dof was loose by ~100 and mu inflated into oversmoothing.
    After GLS whitening the identity E[chi2 at fit] = nb - dof holds by
    construction and the discrepancy criterion means what it says.

    Shrinkage: C = (1-a) C_boot + a diag(sig^2) with the
    Schaefer-Strimmer intensity for the off-diagonal (closed form, no
    tuning); sig carries the sparse-bin Poisson floor, so shrinking
    toward it also floors the covariance where counts are too thin to
    measure it. Replicates scale with the bin count and eigenvalues are
    floored at 0.5 in correlation scale, so no direction is claimed
    more than sqrt(2)-fold more certain than the marginal model.

    STATUS: correct but not default. The GLS route makes the
    discrepancy target exact, yet measured end-to-end it loses to
    marginal whitening under the cross-split selection on every case
    (see the note in estimate_adaptive): the dense rotation injects
    covariance-estimation noise where the tails are decided, and the
    cross-split criterion is correlation-proof without it."""
    innov = np.asarray(innov, float)
    grid = np.asarray(grid, float)
    n = grid.size
    dx = grid[1] - grid[0]
    lo, hi = grid[0], grid[-1] + dx
    if groups is None:
        groups = np.arange(innov.size)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    G = uniq.size
    # index innovation samples by group once
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    starts = np.searchsorted(g_sorted, uniq, side="left")
    ends = np.searchsorted(g_sorted, uniq, side="right")
    rng = np.random.default_rng(seed + 424243)
    sizes = ends - starts
    eq = int(sizes.min()) == int(sizes.max())
    OM = order.reshape(G, int(sizes[0])) if eq else None
    # replicates must outnumber bins, or the sample covariance is rank
    # deficient and the whitener makes confident claims about directions
    # it never measured -- at 435 bins and 192 replicates the unmeasured
    # subspace was over half the space and a well-specified case read as
    # over-dispersed (chi2 1.33x its target at the roughest fit)
    B = max(int(n_boot), int(np.ceil(1.25 * n)) + 1)
    reps = np.empty((B, n))
    for b in range(B):
        pick = rng.integers(0, G, size=G)
        if eq:
            sel = OM[pick].ravel()
        else:
            sel = np.concatenate([order[starts[p]:ends[p]] for p in pick])
        reps[b], _ = np.histogram(innov[sel], bins=n, range=(lo, hi),
                                  density=True)
    Xc = reps - reps.mean(axis=0)
    C = (Xc.T @ Xc) / (B - 1)
    # Schaefer-Strimmer intensity toward the diagonal target, in matmuls:
    # with w_bij = Xc_bi Xc_bj, sum_b (w - wbar)^2 = S2 - B wbar^2 where
    # S2 = (Xc^2)' (Xc^2) and wbar = S1/B, S1 = Xc' Xc. No (B,n,n) tensor.
    S1 = Xc.T @ Xc
    S2 = (Xc ** 2).T @ (Xc ** 2)
    wbar = S1 / B
    var_s = (B / float(B - 1) ** 3) * (S2 - B * wbar ** 2)
    off = ~np.eye(n, dtype=bool)
    denom = float((C[off] ** 2).sum())
    a = 1.0 if denom <= 0 else float(np.clip(var_s[off].sum() / denom,
                                             0.0, 1.0))
    D = np.asarray(sig, float) ** 2
    C_sh = (1.0 - a) * C
    np.fill_diagonal(C_sh, (1.0 - a) * np.diag(C) + a * D)
    # keep the marginal floor: no bin less certain than sig says
    dcl = np.maximum(np.diag(C_sh), D)
    np.fill_diagonal(C_sh, dcl)
    # STATISTICAL eigenvalue floor, in correlation scale: no direction
    # may be claimed more certain than ev_floor times the independent-
    # bins marginal model. Duplication (the reason for GLS) only
    # INFLATES directions; the genuinely deflated one is the fixed-mass
    # sum, where the model's equality constraint zeroes the residual
    # anyway, so flooring it costs nothing. Without this floor,
    # estimation noise in the small eigenvalues of C amplifies residual
    # components 1/sqrt(ev)-fold and chi2 reads over-dispersion on
    # well-specified data.
    ev_floor = 0.5
    s = np.sqrt(D)
    M = C_sh / np.outer(s, s)
    evl, U = np.linalg.eigh(M)
    evl = np.maximum(evl, ev_floor)
    Wc = (U / np.sqrt(evl)) @ U.T
    # W must satisfy W C_sh W' = I: with M = S^-1 C_sh S^-1 (S = diag(s))
    # and Wc = M^{-1/2}, W = Wc S^-1 does, and acts on residuals via W r
    W = Wc * (1.0 / s)[None, :]
    return W, a


def _qp(H, a, C, b):
    """quadprog with a conditioning-failure retry ladder.

    quadprog reports "constraints are inconsistent" on numerical
    breakdown even though pi >= 0 with unit mass is always feasible;
    measured on the d800 ensemble the failures are PATCHY IN mu
    (1.2e-2 failed between two successes), and since the cross-split
    criterion maps a failed half-fit to +inf, the selection was left
    choosing over a mu axis with holes exactly where the smooth
    candidates lived -- the one-SE rule degraded to "roughest that
    solves" and picked 1.1e-3 with satellites. The first attempt is
    bit-identical to the direct call; retries rescale the objective to
    O(1) (solution-invariant) and add a relative ridge escalating from
    1e-12 to 1e-4 of the Hessian scale, negligible against the penalty
    curvature but enough to carry the factorization through."""
    try:
        return quadprog.solve_qp(H, a, C, b, meq=1)[0]
    except ValueError:
        pass
    n = H.shape[0]
    s = max(float(np.abs(np.diag(H)).max()), 1.0)
    last = None
    for rel in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return quadprog.solve_qp((H + rel * s * np.eye(n)) / s, a / s,
                                     C, b, meq=1)[0]
        except ValueError as e:
            last = e
    raise last


def _solve_whitened(Eta, f_d, sig, dx, mu, n_irls=3, floor=1e-6,
                    ridge=1e-10):
    """The _solve QP with a whitened data term and the dx^-5-normalized
    penalty; see the section comment above. Returns (pi, chi2).

    Eta may be RECTANGULAR, (n_rows, n_cols) with n_cols <= n_rows: the
    support mask removes columns the data cannot inform (see
    _eroded_support) while every innovation bin keeps its row, so mass
    the kernel spreads outward from interior bins still meets its data.
    pi has length n_cols; the caller embeds it into the full grid."""
    Eta = np.asarray(Eta, float)
    n = Eta.shape[1]
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
        pi = _qp(H, a, C, b)
        pi = np.maximum(pi, 0.0)
    r = dx * (Eta @ pi) - f_d
    chi2 = float(((r / sig) ** 2).sum())
    # Effective degrees of freedom of the fit: trace of the hat matrix of
    # the whitened penalized LS, restricted to the free (pi > 0) bins,
    # minus one for the mass constraint. This is what the discrepancy
    # target must be corrected by: E[chi2 at the FITTED solution] is
    # n_rows - dof, not n_rows, and targeting n_rows buys decades of
    # extra smoothing (measured: it flattened a +6-kurtosis truth to +3
    # while chi2/nb still read 0.9).
    F = pi > 0.0
    if F.sum() >= 2:
        Hd_F = H_data[np.ix_(F, F)]
        H_F = H[np.ix_(F, F)]
        try:
            dof = float(np.trace(np.linalg.solve(H_F, Hd_F))) - 1.0
        except np.linalg.LinAlgError:
            dof = float(F.sum()) - 1.0
        dof = float(min(max(dof, 0.0), F.sum()))
    else:
        dof = 0.0
    return pi, chi2, dof


def _kernel_reach(grid, f_k, q=0.01):
    """Effective half-support of the kernel: the smallest r such that
    [-r, r] carries at least 1-q of the kernel's histogram mass. Finite
    supports add under convolution, so this is the erosion radius for the
    identifiable support of pi (see _eroded_support)."""
    grid = np.asarray(grid, float)
    f_k = np.asarray(f_k, float)
    dx = grid[1] - grid[0]
    mass = f_k * dx
    tot = mass.sum()
    if not tot > 0:
        return float(grid[-1] - grid[0])
    c = np.cumsum(mass) / tot
    i_lo = int(np.searchsorted(c, q / 2.0))
    i_hi = int(np.searchsorted(c, 1.0 - q / 2.0))
    i_hi = min(i_hi, grid.size - 1)
    return float(max(abs(grid[i_lo]), abs(grid[i_hi]), dx))


def _eroded_support(grid, innov, reach, min_bins=24):
    """Column mask for the identifiable support of pi.

    supp f_d = supp pi + supp kernel, so supp pi is the OBSERVED
    innovation range eroded by the kernel reach (Minkowski difference):
    a bin of pi outside [d_min + r, d_max - r] would place kernel mass
    where no innovation was ever seen, and the data term cannot inform
    it -- those bins are where the estimator grew its spurious tail
    lobes at low n (the +-4.7 lobes at n=600 in the obs-density sweep
    lived entirely in the uninformed region). Masked bins are fixed at
    zero and the export's tail extrapolation carries on from the
    interior slopes, which is the existing contract. Returns (j0, j1)
    slice bounds into grid; the full grid if erosion would leave fewer
    than min_bins bins."""
    grid = np.asarray(grid, float)
    innov = np.asarray(innov, float)
    lo = float(innov.min()) + reach
    hi = float(innov.max()) - reach
    j0 = int(np.searchsorted(grid, lo, side="left"))
    j1 = int(np.searchsorted(grid, hi, side="right"))
    if j1 - j0 < min_bins:
        return 0, grid.size
    return j0, j1


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
                      mu_grid=None, splits=8, trim_log=-30.0, verbose=False,
                      sig=None, support_mask=False, mask_q=0.01,
                      mu_lo=1e-10, mu_hi=1e2, bisect_tol=0.05,
                      whiten="marginal", n_boot=192,
                      select="cv", cv_reps=2):
    """Deconvolve with the whitened objective and discrepancy-selected mu.
    Same inputs and cache contract as estimate_from_histograms, plus the
    raw innovations and groups (both required: the noise level is
    measured, not assumed). The cache carries mu in cache['lambda'] and
    the consistency ratio chi2/nb in cache['chi2_ratio'].

    sig, when given, replaces the measured per-bin noise and is treated
    as INDEPENDENT bin noise (analytic noise for synthetic tests;
    production always measures). Otherwise whiten="gls" measures the
    full bin covariance (see measure_bin_cov) and whitens with its
    inverse square root; whiten="diag" restores marginal whitening.

    support_mask (OFF by default) fixes pi to zero outside the observed
    innovation range eroded by the kernel reach: see _eroded_support.
    Measured, the whitened data term already leaves uninformed tail bins
    near zero (their sig is honestly large), and the sample range
    understates supp f_d at finite n, so full-reach erosion cut genuine
    tail support on a heavy truth (L1 0.281 -> 0.449 at identical
    selection). Reserve the mask for genuinely compact-support cases.

    select="cv" (default): mu by the cross-split one-SE rule, an argmin
    criterion with no absolute chi2 height (see the section comment).
    select="discrepancy": largest mu with chi2 <= nb - dof(mu), found
    by bisection on log mu; bisect_tol is the terminal bracket width in
    decades (0.05 ~ 12% in mu). Passing an explicit mu_grid bypasses
    both and restores the legacy grid scan (reproducibility only)."""
    grid = np.asarray(grid, float)
    f_d = np.asarray(f_d, float)
    n = grid.size
    dx = grid[1] - grid[0]
    f_k = np.asarray(f_k, float)
    Eta = _conv_matrix(f_k, n, dx, grid[0])

    ratio = np.nan
    v_d, v_k = _hist_var(grid, f_d), _hist_var(grid, f_k)
    if v_k > 0:
        ratio = float(np.sqrt(max(v_d - v_k, 0.0)) / np.sqrt(v_k / 2.0))

    shrink = np.nan
    W_gls = None
    if sig is None:
        sig = measure_bin_noise(grid, innov, groups, seed=seed,
                                splits=splits)
        # whiten='gls' is available, not default. First measured with a
        # rank-deficient covariance (192 replicates for up to 435 bins,
        # numerical-only eigenvalue floor), where its losses were partly
        # artifact: unmeasured directions were claimed (1-a)-fold too
        # certain and a well-specified case even drove the discrepancy
        # option into its min-chi2 fallback (L1 0.988). Remeasured after
        # the fix (replicates >= 1.25*nb, eigenvalue floor 0.5 in
        # correlation scale): the fallback failure is gone (heavy2
        # gls+discrepancy 0.988 -> 0.087) but gls+cv still loses every
        # case to marginal+cv -- gauss 0.183 vs 0.115, heavy 0.200 vs
        # 0.193, heavy2 0.116 vs 0.060, bimodal 0.123 vs 0.099 (seed 9).
        # The dense rotation spreads covariance-estimation noise into
        # the tail-sensitive directions, and the cross-split criterion
        # never needed the covariance: its expectation at the truth is
        # split-invariant under any correlation.
        if whiten == "gls":
            W_gls, shrink = measure_bin_cov(grid, innov, groups, sig,
                                            seed=seed, n_boot=n_boot)
            Eta = W_gls @ Eta
            f_d = W_gls @ f_d
            sig = np.ones(n)
    else:
        # analytic noise supplied: independent bins by assumption
        sig = np.asarray(sig, float)

    j0, j1 = 0, n
    if support_mask:
        reach = _kernel_reach(grid, f_k, q=mask_q)
        j0, j1 = _eroded_support(grid, np.asarray(innov, float), reach)
    Eta_s = Eta[:, j0:j1]

    chi2 = {}
    dofs = {}

    def ev(mu):
        pi_s, c2, df = _solve_whitened(Eta_s, f_d, sig, dx, float(mu),
                                       n_irls=n_irls)
        pi = np.zeros(n)
        pi[j0:j1] = pi_s
        chi2[float(mu)] = c2
        dofs[float(mu)] = df
        # feasibility is judged against the dof-corrected target: at the
        # FITTED solution the whitened residual runs n - dof(mu) in
        # expectation, so chi2 <= n is loose by exactly the flexibility
        # the fit spent, and mu inflates until that slack is consumed
        return pi, c2 - (n - df)

    if mu_grid is not None:
        # explicit grid: the legacy scan, kept for reproducibility
        best_pi, chosen = None, None
        for mu in mu_grid:                   # ascending
            try:
                pi, g = ev(mu)
            except Exception:
                continue
            if g <= 0.0:
                chosen, best_pi = float(mu), pi
        rule = "discrepancy (grid)"
        if chosen is None:
            if not chi2:
                raise RuntimeError("no mu on the grid produced a solution")
            chosen = min(chi2, key=chi2.get)
            best_pi, _ = ev(chosen)
            rule = "min-chi2 (model cannot reach the noise level)"
    elif select == "cv":
        # CROSS-SPLIT ONE-SE SELECTION, the default by measurement.
        # Argmin alone under-smooths where the criterion goes flat below
        # the optimum (a spiky pi convolves to nearly the same innovation
        # fit -- the ill-posedness), so the standard one-SE rule is
        # applied toward smoothness, the same rule and for the same
        # reason as the legacy cross-validation. Scoreboard against the
        # discrepancy selection and the shipped coarse grid, L1 to truth
        # (marginal whitening, seed 9):
        #     case              cv+1SE   old grid   discrepancy
        #     gauss 2000x20      0.115     0.067       0.106
        #     heavy 2000x20      0.193     0.281       0.330
        #     heavy2 1200x100    0.060     0.070       0.988 (fallback)
        #     bimodal 2400x20    0.099     0.147       0.217
        #     overdisp x1.24     0.205     0.189       (recovers)
        # Chosen mu spreads over two decades across cases instead of
        # pinning at one grid value, and the over-dispersion failure
        # mode of any fitted-residual target disappears outright: an
        # argmin criterion never compares chi2 to an absolute height.
        innov_a = np.asarray(innov, float)
        vals, ses = {}, {}

        def crit(mu):
            mu = float(mu)
            if mu in vals:
                return vals[mu]
            try:
                s = _cross_split_crit(Eta_s, grid, innov_a, groups, sig,
                                      dx, mu, seed=seed, reps=cv_reps,
                                      n_irls=n_irls, row_W=W_gls)
                vals[mu] = float(s.mean())
                ses[mu] = float(s.std(ddof=1) / np.sqrt(s.size))
            except Exception:
                vals[mu], ses[mu] = np.inf, 0.0
            return vals[mu]

        for mu in np.logspace(-6.0, 0.5, 9):
            crit(mu)
        for _ in range(2):                       # local log refinement
            ks = sorted(vals)
            i = int(np.argmin([vals[k] for k in ks]))
            if i > 0:
                crit(np.sqrt(ks[i - 1] * ks[i]))
            if i < len(ks) - 1:
                crit(np.sqrt(ks[i] * ks[i + 1]))
        best = min(vals, key=vals.get)
        if not np.isfinite(vals[best]):
            raise RuntimeError("cross-split criterion failed at every mu")
        thresh = vals[best] + ses[best]
        accept = sorted((m for m in vals if np.isfinite(vals[m])
                         and vals[m] <= thresh), reverse=True)
        chosen = best_pi = None
        for mu in accept:
            try:
                best_pi, _ = ev(mu)          # final fit on ALL data
                chosen = mu
                break
            except Exception:
                continue                     # half-fits solved, full did not
        if chosen is None:
            raise RuntimeError("no accepted mu solved on the full data")
        rule = "cross-split one-SE"
        mu_argmin = best
        if verbose:
            pts = "  ".join(f"{m:.0e}:{vals[m]:.0f}" for m in sorted(vals)
                            if np.isfinite(vals[m]))
            print(f"    cross-split crit over mu {pts}; argmin "
                  f"{best:.1e}, accepted {chosen:.3e}")
    else:
        # a failed solve counts as infeasible: at large mu the reweighted
        # Hessian can lose positive definiteness, which is the oversmoothing
        # pathology, not a reason to abort the selection
        def ev_safe(mu):
            try:
                return ev(mu)
            except Exception:
                return None, np.inf

        lo = float(mu_lo)
        pi_lo, g_lo = ev_safe(lo)
        while pi_lo is None and lo < mu_hi:
            lo *= 10.0
            pi_lo, g_lo = ev_safe(lo)
        if pi_lo is None:
            raise RuntimeError("no mu in the bracket produced a solution")
        if g_lo > 0.0:
            # infeasible at the roughest end: the over-dispersion signature
            chosen, best_pi = lo, pi_lo
            rule = "min-chi2 (model cannot reach the noise level)"
        else:
            pi_hi, g_hi = ev_safe(mu_hi)
            if g_hi <= 0.0:
                chosen, best_pi = float(mu_hi), pi_hi
                rule = "discrepancy (feasible at mu_hi)"
            else:
                hi = float(mu_hi)
                chosen, best_pi = lo, pi_lo
                while np.log10(hi / lo) > bisect_tol:
                    mid = float(np.sqrt(lo * hi))
                    pi_m, g_m = ev_safe(mid)
                    if pi_m is not None and g_m <= 0.0:
                        lo, chosen, best_pi = mid, mid, pi_m
                    else:
                        hi = mid
                rule = "discrepancy (bisection)"
    if verbose:
        pts = "  ".join(f"{m:.0e}:{chi2[m] / max(n - dofs[m], 1.0):.2f}"
                        for m in sorted(chi2))
        print(f"    chi2/(nb-dof) over mu {pts}; chosen ({rule}) mu = "
              f"{chosen:.3e} (nb {n}, dof {dofs[chosen]:.0f}, "
              f"mask [{j0}:{j1}] of {n}, resolvability {ratio:.2f})")
    # The cache keeps _make_cache's contract (the contiguous main run, which
    # is what the unimodal DA export can use, with kept_mass honestly
    # reporting that run's fraction), but the RETURNED arrays are the full
    # estimate: a genuinely multimodal recovery must not be silently
    # truncated to its tallest mode -- measured, the truncation cost a
    # bimodal truth its entire second mode and 70% of its variance.
    best_pi, n_filled = _fill_zero_bins(best_pi, dx)
    _, _, cache = _make_cache(grid, best_pi, dx, chosen, ratio, trim_log)
    cache["zero_bins_filled"] = n_filled
    if select == "cv" and mu_grid is None:
        cache["mu_argmin"] = mu_argmin
    cache["chi2_ratio"] = chi2[chosen] / max(n - dofs[chosen], 1.0)
    cache["dof"] = dofs[chosen]
    cache["shrink"] = shrink
    cache["support_mask"] = (int(j0), int(j1))
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


def _fill_zero_bins(pi, dx, max_gap=3):
    """Fill EXACT-ZERO bins inside the support of pi by linear interpolation
    of log pi from the nearest positive bins on each side, for gaps of at
    most max_gap bins, then renormalize to unit mass. Returns (pi, n_filled).

    The QP is constrained pi >= 0 and quadprog's active set puts exact
    zeros where a noisy data term pushes a bin negative. The reweighted
    third-difference penalty on log pi cannot itself reach zero, but the
    active set can, and three IRLS passes do not undo it. An exact zero
    inside the support is a solver boundary artifact, not a claim that the
    density vanishes: measured on cycled-filter archives (2026-09-02),
    isolated zero bins at 1.5 to 6 widths from the center inside the
    mass-carrying run caught 0.5 to 1.5 percent of the true errors of a
    channel, each at log(1e-300) = -690 nats once the export interpolated
    log pi across the bin, adding 2 to 6 nats per channel to the KL of a
    density whose width and kept mass both passed. Leading and trailing
    zeros, outside the outermost positive bins, are the support edge and
    are left alone; the export's tails take over there. Gaps longer than
    max_gap are left alone too: a body separated from far outlier lobes by
    a long run of zeros is a genuine gap, and bridging it in log space
    moves mass into it (measured: the self-test's outlier case, T sd 0.82).
    """
    pi = np.asarray(pi, float).copy()
    pos = np.flatnonzero(pi > 0.0)
    if pos.size < 2:
        return pi, 0
    gaps = np.diff(pos) - 1
    hole = np.zeros(pi.size, bool)
    for a, g in zip(pos[:-1], gaps):
        if 0 < g <= max_gap:
            hole[a + 1:a + 1 + g] = True
    n_filled = int(hole.sum())
    if n_filled:
        ii = np.arange(pi.size)
        pi[hole] = np.exp(np.interp(ii[hole], ii[pi > 0], np.log(pi[pi > 0])))
        pi /= pi.sum() * dx
    return pi, n_filled


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
    s0, e0 = int(starts[k]), int(ends[k])
    if e0 - s0 < 2:
        # a single-bin main run is a fully fragmented solve (min-chi2
        # fallback on undecodable data); widen to the two nearest bins so
        # slopes exist. kept_mass will be honest about the fragmentation
        # and the export contract already says such estimates are unusable.
        s0, e0 = max(0, s0 - 1), min(pi.size, e0 + 1)
        if e0 - s0 < 2:
            s0, e0 = 0, min(pi.size, 2)
    xg, pin = x_grid[s0:e0], pi[s0:e0]

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
    pi, n_filled = _fill_zero_bins(pi, dx)

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
             "kept_mass": float(pin.sum() * dx),
             "zero_bins_filled": n_filled}
    if ratio < 0.5:
        import warnings as _w
        _w.warn(f"observation error is only {ratio:.2f} of the background "
                "spread; the deconvolution is ill posed in this regime and the "
                "recovered shape should not be trusted. Stratify to reduce the "
                "background spread, or take the variance estimate only.",
                RuntimeWarning)
    return xg, pin, cache
