#!/usr/bin/env python3
"""Read l95 EDA member observation files and form what DOEE needs.

Each EDA member runs a full 3D-Var and writes its own observation file. CostJo
saves the first-guess departure into it as the `ombg` column,

    ombg_k = y - H(x_b^k)

so the member values in observation space follow as

    H(x_b^k) = ObsValue - ombg_k

and nothing extra has to be written by the model.

WHAT TO DO WITH THEM. Hand the observations and the member values to
`histograms_from_ensemble`, which pairs at the SAME LOCATION to build the
innovation histogram and the kernel.

Do NOT form innovations and perturbations and pass them to estimate_noise_pmf as
its X and Y. That estimator forms Y - X itself by pairing indices at random, so
an innovation handed to it is differenced a second time:

    d = obs - member    has variance sigma_o^2 + 2 sigma_b^2
    p = member - mean   has variance sigma_b^2
    d_i - p_j           has variance sigma_o^2 + 3 sigma_b^2
    kernel p_i - p_j    has variance 2 sigma_b^2
    recovered           sigma_o^2 + sigma_b^2

One background variance is left behind, and no amount of data removes it. The
symptom is a recovered width of sqrt(sigma_o^2 + sigma_b^2) that does not
improve with sample size or ensemble size.

THE MEMBERS MUST HAVE DIFFERENT BACKGROUNDS. The EDA configuration shipped with
l95 gives every member the same background file and varies only the observation
perturbation seed, so H(x_b^k) is identical across members, the perturbations
are exactly zero, and the deconvolution kernel collapses to a delta. That
ensemble samples analysis uncertainty arising from observation error, not
background error, and is not usable here. Build the members from a background
ensemble: genenspert, or the previous cycle's analyses. `collect` checks for
this and refuses rather than returning a degenerate kernel.
"""

import glob
import os

import numpy as np


def read_obt(path):
    """Return {column name: array} plus index, time and location."""
    with open(path) as f:
        lines = f.read().splitlines()
    ncol = int(lines[0].strip())
    names = [lines[1 + i].strip() for i in range(ncol)]
    nobs = int(lines[1 + ncol].strip())
    rows = [ln.split() for ln in lines[2 + ncol: 2 + ncol + nobs]]
    if len(rows) != nobs:
        raise ValueError(f"{path}: header says {nobs} rows, found {len(rows)}")
    out = {"index": np.array([int(r[0]) for r in rows]),
           "time": np.array([r[1] for r in rows]),
           "location": np.array([float(r[2]) for r in rows])}
    for i, nm in enumerate(names):
        out[nm] = np.array([float(r[3 + i]) for r in rows])
    return out


def collect(pattern, departure="ombg", min_spread=1e-8):
    """Gather the observations and the member values from member files.

    Returns (obs, hofx, meta): obs of shape (n,), hofx of shape (n, K). Pass
    both to histograms_from_ensemble rather than forming innovations and
    perturbations here -- see the module docstring for why that fails.

    pattern : glob matching one file per member, e.g. 'Data/mem*.eda.*.obt'
    """
    paths = sorted(glob.glob(pattern)) if isinstance(pattern, str) \
        else sorted(pattern)
    if len(paths) < 2:
        raise ValueError(f"found {len(paths)} member files matching "
                         f"{pattern!r}; at least two are needed, since the "
                         "kernel is built from differences between members")

    recs = [read_obt(p) for p in paths]
    for p, r in zip(paths, recs):
        for col in ("ObsValue", departure):
            if col not in r:
                cols = [k for k in r
                        if k not in ("index", "time", "location")]
                raise KeyError(f"{p}: no '{col}' column (found {cols}). The "
                               "departure is written by the cost function, so "
                               "the member must have run an analysis.")
    n = len(recs[0]["ObsValue"])
    for p, r in zip(paths, recs):
        if len(r["ObsValue"]) != n:
            raise ValueError(f"{p}: {len(r['ObsValue'])} observations but "
                             f"{paths[0]} has {n}; members must cover the same "
                             "observations")

    obs = recs[0]["ObsValue"]
    innov = np.column_stack([r[departure] for r in recs])      # (n, K)
    hofx = obs[:, None] - innov                                # H(x_b^k)

    spread = float(np.mean(np.std(hofx, axis=1)))
    if spread < min_spread:
        raise ValueError(
            f"the member backgrounds are identical (mean spread {spread:.2e}). "
            "Every member is reading the same background file, so the kernel "
            "collapses to a delta and there is nothing to deconvolve. Build the "
            "ensemble from genenspert or from the previous cycle's analyses.")

    keep = np.all(np.isfinite(innov), axis=1) & np.isfinite(obs)
    meta = {"members": len(paths), "n_obs": int(keep.sum()),
            "files": [os.path.basename(p) for p in paths],
            "mean_background_spread": spread,
            "innovation_sigma": float(np.std(innov[keep]))}
    return obs[keep], hofx[keep, :], meta


def reliability(pattern, truth_obs, departure="ombg"):
    """Ensemble reliability, which can only be checked when the truth is known.

    If members and truth are exchangeable draws from the same distribution then

        RMSE(ensemble mean) = spread * sqrt((K+1)/K)

    Under-dispersion means the deconvolution attributes the missing background
    spread to observation error and sigma_o comes out biased high; over-
    dispersion biases it low. Worth reporting every cycle rather than assuming,
    because with real observations it cannot be checked at all.
    """
    paths = sorted(glob.glob(pattern)) if isinstance(pattern, str) \
        else sorted(pattern)
    recs = [read_obt(p) for p in paths]
    obs = recs[0]["ObsValue"]
    hofx = obs[:, None] - np.column_stack([r[departure] for r in recs])
    K = hofx.shape[1]
    spread = np.sqrt(np.mean(np.var(hofx, axis=1, ddof=1)))
    err = np.sqrt(np.mean((hofx.mean(axis=1) - truth_obs) ** 2))
    # The mean bias is reported separately because it is the deconvolution's
    # blind direction: the member-difference kernel is symmetric about zero
    # BY CONSTRUCTION (any ensemble-mean bias cancels in member_i - member_j),
    # so a bias mu in the members shifts the recovered noise density by -mu
    # and nothing on this data path can tell the two apart. The recovered
    # density's location is therefore identified only up to this number, and
    # |bias| <= rmse_of_mean bounds it even when the truth is not known.
    bias = float(np.mean(hofx.mean(axis=1) - truth_obs))
    expected = spread * np.sqrt((K + 1.0) / K)
    ratio = expected / err if err > 0 else np.nan
    return {"spread": float(spread), "rmse_of_mean": float(err),
            "bias_of_mean": bias,
            "expected_spread_for_reliability": float(expected),
            "ratio": float(ratio),
            "note": ("reliable" if 0.8 < ratio < 1.25 else
                     "UNDER-DISPERSIVE, sigma_o will come out biased high"
                     if ratio < 0.8 else
                     "OVER-DISPERSIVE, sigma_o will come out biased low")}
