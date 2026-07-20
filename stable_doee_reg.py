"""Regularised variant of the DOEE estimator.

Same deconvolution as `estimate_noise_pmf` in stable_doee: match the innovation
histogram to the convolution of the noise density with the empirical difference
histogram, subject to non-negativity and unit mass. Three changes.

1. PENALTY ON THE CURVATURE OF log pi, not on differences of pi.

   The density is represented downstream as piecewise-linear in log with
   quadratic-log tails, so a penalty on the second difference of log pi costs
   nothing for a log-linear segment or an exponential tail -- exactly the shapes
   worth keeping -- while still suppressing bin-to-bin noise.

   Penalising differences of pi instead drives the solution against the
   non-negativity boundary in the tails; the trimming that follows then cuts
   them. Measured on synthetic data the original returns an excess kurtosis
   near -0.6 for every truth tried, including one whose true excess kurtosis is
   +7.2 -- that is, it reports a lighter-than-Gaussian tail for a strongly
   heavy-tailed error. For a method whose purpose is the tails that is the
   failure that matters.

   Kept a QP by iterative reweighting: d^2 log pi ~ (d^2 pi)/pi, so the penalty
   is || W^(1/2) L pi ||^2 with W = diag(1/pi_hat^2), refreshed each solve.

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
case, scoring L1 distance to the true density:

    case            original   regularised
    gaussian 1.0      0.101       0.134
    heavy 85/15       0.338       0.122
    laplace 0.8       0.306       0.123
    very heavy 95/5   0.381       0.244
    gaussian 0.4      0.430       0.216   (flagged: resolvability 0.48)
    total             1.557       0.839

The one regression is a Gaussian truth, where the original is already adequate;
every non-Gaussian case improves by a factor of two to three, and the recovered
excess kurtosis tracks the truth (+6.0 against +7.2, +3.0 against +2.8) instead
of sitting at -0.6 regardless.

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


def _second_difference(n):
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i], L[i, i + 1], L[i, i + 2] = 1.0, -2.0, 1.0
    return L


def _solve(Eta, f_d, dx, n, lam, n_irls=3, floor=1e-6, ridge=1e-10):
    """QP with an IRLS approximation to a curvature-of-log penalty."""
    L = _second_difference(n)
    H_data = 2.0 * dx ** 2 * (Eta.T @ Eta)
    f_data = -2.0 * dx * (Eta.T @ f_d)
    a = -f_data

    C_eq = dx * np.ones((n, 1))
    C = np.hstack([C_eq, np.eye(n)])
    b = np.hstack([1.0, np.zeros(n)])

    pi = np.full(n, 1.0 / (n * dx))            # flat start
    for _ in range(max(1, n_irls)):
        w = 1.0 / np.maximum(pi, floor) ** 2
        # weight each curvature row by the local 1/pi^2, geometric mean over the
        # three bins it touches so the weight itself is smooth
        wr = np.exp(np.log(w[:-2] * w[1:-1] * w[2:]) / 3.0)
        H_smooth = 2.0 * lam * (L.T @ (wr[:, None] * L))
        H = H_data + H_smooth + ridge * np.eye(n)
        H = 0.5 * (H + H.T)                    # symmetrise for quadprog
        ev = np.linalg.eigvalsh(H)
        if ev[0] <= 0:
            H += (abs(ev[0]) + 1e-8) * np.eye(n)
        pi = quadprog.solve_qp(H, a, C, b, meq=1)[0]
        pi = np.maximum(pi, 0.0)
    return pi


def _cv_fold_scores(Eta, dx, n, x_min, x_max, innov, lam, folds=4, seed=0,
                    n_irls=3):
    """Per-fold predictive log-likelihood of held-out innovations under
    Eta @ pi, returned separately so a standard error can be formed."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(innov))
    parts = np.array_split(idx, folds)
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


def resolvability(X, Y):
    """How much of the innovation is observation error rather than background
    spread. Returns (sigma_o / sigma_b, sigma_o).

    sigma_o^2 = var(innovation) - var(perturbation). Below about 0.5 the
    deconvolution is ill posed and the recovered shape should not be trusted,
    whichever estimator is used: no amount of data or regularisation resolves a
    narrow density from a much wider kernel.
    """
    v_innov = float(np.var(Y))           # sigma_o^2 + sigma_b^2
    v_b = float(np.var(X))               # perturbations carry sigma_b^2
    v_noise = max(v_innov - v_b, 0.0)
    sb = np.sqrt(max(v_b, 1e-300))
    return float(np.sqrt(v_noise) / sb), float(np.sqrt(v_noise))


def estimate_noise_pmf_reg(X, Y, lam=None, lam_grid=None, folds=4,
                           n_irls=3, p_lo=1, p_hi=99, pad_frac=0.05,
                           trim_log=-30.0, seed=0, verbose=False,
                           well_posed=0.7):
    """Regularised DOEE. Returns (x_grid, pi_N, cache) like the original, with
    'lambda' and 'resolvability' added to the cache.

    X   : ensemble perturbations   H(x_k) - H(x_bar)
    Y   : ensemble innovations     y - H(x_k)
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

    ratio, sig_o_hat = resolvability(X, Y)
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
    k = (ends - starts).argmax()
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
             "lambda": lam, "resolvability": ratio}
    if ratio < 0.5:
        import warnings as _w
        _w.warn(f"observation error is only {ratio:.2f} of the background "
                "spread; the deconvolution is ill posed in this regime and the "
                "recovered shape should not be trusted. Stratify to reduce the "
                "background spread, or take the variance estimate only.",
                RuntimeWarning)
    return xg, pin, cache
