#!/usr/bin/env python3
"""What tail heaviness does the estimator report when there is none?

Deconvolution under a non-negativity constraint cannot perform the
cancellations that exact inversion requires, so the recovered density can carry
tail weight that is not in the data. A recovered excess kurtosis is therefore
evidence of a heavy tail only if it exceeds what the same estimator returns on
Gaussian data at the same configuration. That number is the FLOOR, it depends
on sigma_o, sigma_b, the number of observations and the ensemble size, and it
has to be computed per stratum rather than taken from a table.

    floor = null_floor(sigma_o=0.4, sigma_b=0.5, n=8000, K=20)
    if recovered_kurtosis > floor["kurtosis_p95"]:
        ... the tail is supported at this configuration

The floor is also a measurement of the regularizer, which the second-difference
log penalty demonstrated: its floor sat near +4 at sigma_o/sigma_b = 0.8 and
+29 at 0.5, did not fall with more observations or members (2.5 times the data
moved it slightly UP), and was insensitive to lambda and bin count -- the
signature of the penalty's own exponential-tail prior contaminating the
answer, not of a law of deconvolution. Measured floors under the
third-difference log penalty, Gaussian truth, trials=6:

    sigma_o/sigma_b   floor mean   floor p95   width bias    (2nd-diff p95)
        2.00             +0.18       +0.26        +1.9%          +0.88
        0.80             +0.84       +1.54        +4.2%          +4.28
        0.80 2.5x data   +0.52       +0.55        +2.9%
        0.50             +1.04       +1.24       +11.5%         +28.99

Three changes of regime. The floor now FALLS with more data, variance-like
rather than bias-like. It stays near +1 even at ratio 0.5, where the old
penalty made the shape unusable -- though the width bias there (+11.5%) says
width recovery is still degraded, so the resolvability gate stays. And
Laplace-level tails (about +3) now clear the floor at every ratio measured --
at the five-density laplace configuration (ratio 1.13, 12 trials) the floor
p95 is +0.96 against a recovered +1.8 -- where under the old penalty they were
undetectable below a ratio of about one. Trials are few (6 to 12), so p95 is a
coarse quantile: compute the floor at your own configuration rather than
reading it off this table.
"""

import sys
from pathlib import Path

import numpy as np

# the estimator lives one level up, at the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_doee_reg as R


def _moments(grid, pi, ref=None):
    dx = grid[1] - grid[0]
    p = np.asarray(pi, float)
    tot = p.sum() * dx
    if tot <= 0:
        return np.nan, np.nan, np.nan
    p = p / tot
    mu = (p * grid).sum() * dx
    sd = np.sqrt((p * (grid - mu) ** 2).sum() * dx)
    ku = (p * (grid - mu) ** 4).sum() * dx / sd ** 4 - 3.0
    l1 = float(np.abs(p - ref).sum() * dx) if ref is not None else np.nan
    return float(sd), float(ku), l1


def _one_draw(sigma_o, sigma_b, n, K, seed, lam, n_bins, field_amp):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    field = field_amp * (np.sin(x / 37.0) + 0.5 * np.cos(x / 11.0))
    truth = field + rng.normal(0.0, sigma_b, n)        # exchangeable with members
    hofx = field[:, None] + rng.normal(0.0, sigma_b, (n, K))
    obs = truth + rng.normal(0.0, sigma_o, n)          # GAUSSIAN by construction
    grid, f_d, f_k, _iv = R.histograms_from_ensemble(obs, hofx, seed=seed + 1,
                                                     n_bins=n_bins)
    return R.estimate_from_histograms(grid, f_d, f_k, lam=lam)


def null_floor(sigma_o, sigma_b, n, K, trials=12, lam=1e-1, n_bins=161,
               field_amp=4.0, seed=0, common_grid=None):
    """Run the estimator on Gaussian data matching a configuration.

    Returns the distribution of recovered excess kurtosis and width. Compare a
    real estimate against `kurtosis_p95`: below it, the shape is consistent with
    a Gaussian error seen through this estimator, whatever it looks like.
    """
    grid = np.arange(-6.0, 6.0001, 0.02) if common_grid is None else common_grid
    ref = np.exp(-grid ** 2 / (2 * sigma_o ** 2)) \
        / (sigma_o * np.sqrt(2 * np.pi))
    kurt, width, l1 = [], [], []
    for t in range(trials):
        try:
            g, pi, _c = _one_draw(sigma_o, sigma_b, n, K, seed + 100 * t,
                                  lam, n_bins, field_amp)
            p = np.interp(grid, g, pi, left=0.0, right=0.0)
            sd, ku, d1 = _moments(grid, p, ref)
        except Exception:
            sd = ku = d1 = np.nan
        if np.isfinite(ku):
            kurt.append(ku)
            width.append(sd)
            l1.append(d1)
    if not kurt:
        raise RuntimeError("every null trial failed")
    kurt, width, l1 = np.array(kurt), np.array(width), np.array(l1)
    return {"trials": len(kurt),
            "sigma_o": sigma_o, "sigma_b": sigma_b, "n": n, "K": K,
            "kurtosis_mean": float(kurt.mean()),
            "kurtosis_sd": float(kurt.std(ddof=1)) if kurt.size > 1 else 0.0,
            "kurtosis_p95": float(np.percentile(kurt, 95)),
            "width_mean": float(width.mean()),
            "width_bias": float(width.mean() / sigma_o - 1.0),
            "l1_mean": float(np.nanmean(l1))}


def verdict(recovered_kurtosis, floor):
    """Is a recovered tail distinguishable from the artifact?"""
    if recovered_kurtosis > floor["kurtosis_p95"]:
        return ("heavier than the estimator produces on Gaussian data at this "
                "configuration: the tail is supported")
    return ("within what the estimator produces on Gaussian data at this "
            f"configuration (floor p95 = {floor['kurtosis_p95']:+.2f}): the "
            "tail is NOT supported, and a non-Gaussian density should not be "
            "claimed from it")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print("null calibration: excess kurtosis reported on Gaussian truth\n")
    print(f"{'sig_o':>7}{'sig_b':>7}{'ratio':>7}{'n':>8}{'K':>4}"
          f"{'floor mean':>12}{'floor p95':>11}{'width bias':>12}")
    for sig_o, sig_b, n, K in ((0.4, 0.5, 8000, 20),
                               (0.4, 0.5, 20000, 40),
                               (1.0, 0.5, 8000, 20),
                               (0.25, 0.5, 8000, 20)):
        f = null_floor(sig_o, sig_b, n, K, trials=6)
        print(f"{sig_o:7.2f}{sig_b:7.2f}{sig_o / sig_b:7.2f}{n:8d}{K:4d}"
              f"{f['kurtosis_mean']:+12.2f}{f['kurtosis_p95']:+11.2f}"
              f"{100 * f['width_bias']:+11.1f}%")
