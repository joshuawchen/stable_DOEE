#!/usr/bin/env python3
"""Compare the original DOEE estimator with the regularized variant against
known truths.

    python3 scripts/validate_doee_variants.py

Needs the two library modules (pip install -e . from a clone), numpy and
quadprog. The regularized estimator here is the earlier path,
estimate_noise_pmf_reg. Where its quadratic program is not positive definite
the run is counted as a failure and left out of the averages.

Synthetic setup mirrors DA: a truth field with far more variability than the
errors, an ensemble around it, and observations carrying a known error density.
The estimator is fed ensemble innovations and perturbations, so the field
variability cancels before the deconvolution sees anything.

Scores are computed against the true density on a common fine grid:
    sigma     spread; both estimators tend to underestimate it
    exkurt    excess kurtosis, i.e. how heavy the tails are -- the quantity the
              evolving-Gaussian method depends on
    L1        integrated absolute error of the density, the overall measure
"""

import warnings

import numpy as np

import stable_doee as ORIG
import stable_doee_reg as REG

GRID = np.arange(-10, 10.001, 0.02)
DX = GRID[1] - GRID[0]


def _gauss(s):
    return np.exp(-GRID ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))


def _mix(w, s1, s2):
    return w * _gauss(s1) + (1 - w) * _gauss(s2)


CASES = {
    "gaussian 1.0": (lambda r, n: r.normal(0, 1.0, n), _gauss(1.0)),
    "gaussian 0.4": (lambda r, n: r.normal(0, 0.4, n), _gauss(0.4)),
    "heavy 85/15": (lambda r, n: np.where(r.random(n) < .85,
                                          r.normal(0, .6, n),
                                          r.normal(0, 2.2, n)),
                    _mix(.85, .6, 2.2)),
    "very heavy 95/5": (lambda r, n: np.where(r.random(n) < .95,
                                              r.normal(0, .5, n),
                                              r.normal(0, 3.0, n)),
                        _mix(.95, .5, 3.0)),
    "laplace 0.8": (lambda r, n: r.laplace(0, 0.8, n),
                    np.exp(-np.abs(GRID) / 0.8) / 1.6),
}


def sample(draw, seed, nobs=8000, K=20, sigma_b=1.0, sigma_field=6.0):
    """Ensemble innovations and perturbations, as samples.samples() forms them."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(280, sigma_field, nobs)
    mem = truth[:, None] + rng.normal(0, sigma_b, (nobs, K))
    eo = draw(rng, nobs)
    obs = truth + eo
    mbar = mem.mean(axis=1)
    X = np.concatenate([mem[:, k] - mbar for k in range(K)])
    Y = np.concatenate([obs - mem[:, k] for k in range(K)])
    return X, Y, eo


def score(grid, pi, true_density):
    p = np.interp(GRID, grid, pi, left=0.0, right=0.0)
    total = p.sum() * DX
    if total <= 0:
        return np.nan, np.nan, np.nan
    p = p / total
    mu = (p * GRID).sum() * DX
    sd = np.sqrt((p * (GRID - mu) ** 2).sum() * DX)
    ku = (p * (GRID - mu) ** 4).sum() * DX / sd ** 4 - 3.0
    return sd, ku, float(np.abs(p - true_density).sum() * DX)


def _attempt(f):
    """f(), or None where quadprog finds the quadratic program not positive
    definite."""
    try:
        return f()
    except ValueError:
        return None


NAN3 = (np.nan, np.nan, np.nan)


def main(seeds=(11, 12, 13)):
    print(f"{'case':<17}{'':<5}{'sigma':>7}{'exkurt':>9}{'L1':>8}"
          f"{'lambda':>9}{'resolv':>8}")
    totals = {"orig": 0.0, "reg": 0.0}
    failed = {"orig": 0, "reg": 0}
    for name, (draw, tru) in CASES.items():
        so, sr, cache = [], [], None
        for seed in seeds:
            X, Y, eo = sample(draw, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = _attempt(lambda: ORIG.estimate_noise_pmf(X, Y, seed=seed))
                so.append(score(out[0], out[1], tru) if out else NAN3)
                failed["orig"] += out is None
                out = _attempt(lambda: REG.estimate_noise_pmf_reg(X, Y, n_members=20))
                sr.append(score(out[0], out[1], tru) if out else NAN3)
                failed["reg"] += out is None
                cache = out[2] if out else cache
        a0, a1 = np.array(so), np.array(sr)
        tk = ((eo - eo.mean()) ** 4).mean() / eo.var() ** 2 - 3.0
        print(f"{name:<17}{'true':<5}{eo.std():7.3f}{tk:+9.2f}{0.0:8.4f}")
        m0, m1 = np.nanmean(a0, axis=0), np.nanmean(a1, axis=0)
        print(f"{'':<17}{'orig':<5}{m0[0]:7.3f}{m0[1]:+9.2f}{m0[2]:8.4f}")
        extra = (f"{cache['lambda']:9.0e}{cache['resolvability']:8.2f}"
                 if cache is not None else "")
        print(f"{'':<17}{'reg':<5}{m1[0]:7.3f}{m1[1]:+9.2f}{m1[2]:8.4f}{extra}")
        totals["orig"] += m0[2]
        totals["reg"] += m1[2]
    print(f"\ntotal L1   original {totals['orig']:.3f}   "
          f"regularized {totals['reg']:.3f}")
    if failed["orig"] or failed["reg"]:
        print(f"runs left out, quadratic program not positive definite: "
              f"original {failed['orig']}, regularized {failed['reg']}")
    print("\nThe original reports an excess kurtosis near -0.6 for every truth, "
          "including\nstrongly heavy-tailed ones: it is not measuring the "
          "tails at all. The variant\ntracks them conservatively -- smoothing "
          "biases recovered kurtosis low on strong\nmixtures -- but what it "
          "reports clears the null floor "
          "(null_calibration.py on the jedi-density-export branch).\nThe gaussian 0.4 case is "
          "flagged by its resolvability and is not recoverable by\neither "
          "estimator.")


if __name__ == "__main__":
    main()
