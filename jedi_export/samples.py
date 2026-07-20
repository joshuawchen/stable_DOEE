#!/usr/bin/env python3
"""Collect samples for DOEE from JEDI/IODA diagnostic files, with the metadata
needed to stratify them, pooled over cycles.

WHAT TO FEED THE ESTIMATOR

estimate_noise_pmf(X, Y) solves Y = X + N by matching f_{Y-X} to
f_{X1-X2} * pi_N. Setting

    Y : ENSEMBLE INNOVATIONS      y - H(x_k)          = eps_o - eps_b
    X : ENSEMBLE PERTURBATIONS    H(x_k) - H(x_bar)   ~ -eps_b

gives f_{Y-X} = f_{eps_o} * f_{-eps_b} * f_{eps_b} and
f_{X1-X2} = f_{-eps_b} * f_{eps_b}, so the deconvolution returns f_{eps_o}
exactly. This is DOEE on ensemble innovations. The estimator still treats its
two arrays as unpaired samples; the pairing is used only to FORM the
innovations and perturbations, which is ordinary DA bookkeeping.

Do NOT pass raw ObsValue and raw H(x). Both carry the full variability of the
field, which is far larger than the errors -- for temperature, several K of
weather against about 1 K of observation error. The deconvolution is then asked
to extract a narrow density from a histogram dominated by weather, and in
testing it either returned a width roughly twice the truth or failed outright
with a non-positive-definite QP. Forming innovations and perturbations first
cancels the field exactly, and the recovered width becomes insensitive to how
variable the field is.

WHY STRATIFICATION MATTERS MORE THAN SAMPLE SIZE

Pooling observations with genuinely different error characteristics creates
apparent non-Gaussianity out of nothing: a mixture of Gaussians with different
widths is heavy-tailed. The reference warns about exactly this
(heteroskedasticity from grouping different populations). DOEE will faithfully
recover the mixture and the evolving-Gaussian method will appear to help by
down-weighting the wings, but the correct response is to separate the
populations, not to model their mixture.

So: stratify as finely as the counts allow, and recover sample size by pooling
over TIME, which costs only the assumption that the error density is stationary
over the window.

    always split on : variable, and ObsType (BUFR type; 181, 187, 120, ...)
    split if you can: pressure level or channel, and region if regimes differ
    pool across    : cycles
    or stratify by : a state-dependent predictor, as the reference does with
                     symmetric cloud, estimating one density per bin

A rough guide to counts: the estimator uses sqrt(N) bins, so N ~ 1e4 gives about
100 bins, which resolves a density well. Below about 1e3 the tails are not
usable, and the tails are the entire point of the method.
"""

import glob
import os
from collections import defaultdict

import numpy as np

try:
    import netCDF4
except ImportError:                                        # pragma: no cover
    netCDF4 = None

SENTINEL = 1.0e30
# metadata read when present; missing ones are skipped without complaint
META_WANTED = ("stationIdentification", "latitude", "longitude", "pressure",
               "height", "dateTime", "stationElevation")


def _clean(a):
    a = np.asarray(a, dtype=float)
    a[np.abs(a) > SENTINEL] = np.nan
    return a


def _get(ds, group, var, numeric=True):
    if group in ds.groups and var in ds.groups[group].variables:
        v = ds.groups[group].variables[var][:]
        return _clean(v) if numeric else np.asarray(v)
    return None


def read_file(path, variable, hofx_group="hofx0", qc_group="EffectiveQC0"):
    """Read one diag file. Returns a dict of equal-length arrays: obs, hofx, qc,
    obstype, plus whatever MetaData fields are present."""
    if netCDF4 is None:
        raise ImportError("netCDF4 is required to read IODA diagnostic files")
    rec = {}
    with netCDF4.Dataset(path) as ds:
        obs = _get(ds, "ObsValue", variable)
        if obs is None:
            raise KeyError(f"{path}: no ObsValue/{variable}")
        hofx = _get(ds, hofx_group, variable)
        if hofx is None:
            avail = sorted(g for g in ds.groups if g.lower().startswith("hofx"))
            raise KeyError(f"{path}: no {hofx_group}/{variable}. "
                           f"H(x) groups present: {avail}")
        qc = _get(ds, qc_group, variable)
        rec["obs"] = obs
        rec["hofx"] = hofx
        rec["qc"] = np.zeros_like(obs) if qc is None else qc
        ot = _get(ds, "ObsType", variable)
        rec["obstype"] = np.full(obs.shape, np.nan) if ot is None else ot
        for m in META_WANTED:
            v = _get(ds, "MetaData", m, numeric=(m != "stationIdentification"))
            if v is not None and len(v) == len(obs):
                rec[m] = v
        rec["file"] = np.array([os.path.basename(path)] * len(obs))
    return rec


def load(paths, variable, hofx_groups=("hofx0",), qc_group="EffectiveQC0"):
    """Read many files (cycles) and many H(x) groups (ensemble members).

    Returns (rec, meta). rec["hofx"] has shape (n_obs, n_members) so the
    ensemble mean can be formed per observation; every other column is
    per-observation. Observations are dropped unless they pass QC and every
    member is finite, so the perturbations are complete.
    """
    if isinstance(paths, str):
        paths = sorted(glob.glob(paths))
    paths = list(paths)
    if not paths:
        raise FileNotFoundError("no diagnostic files matched")
    if len(hofx_groups) < 2:
        raise ValueError(
            "at least two ensemble members are needed: the perturbations "
            "H(x_k) - H(x_bar) are what characterise the background error that "
            "gets deconvolved away. With one member there is nothing to "
            "deconvolve.")

    cols, hofx = None, []
    for g in hofx_groups:
        chunks = [read_file(p, variable, hofx_group=g, qc_group=qc_group)
                  for p in paths]
        merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        hofx.append(merged.pop("hofx"))
        if cols is None:
            cols = merged
        elif len(merged["obs"]) != len(cols["obs"]):
            raise ValueError(f"member {g} has {len(merged['obs'])} rows but "
                             f"{hofx_groups[0]} has {len(cols['obs'])}; the "
                             "members must cover the same observations")
    rec = dict(cols)
    rec["hofx"] = np.column_stack(hofx)

    keep = (np.isfinite(rec["obs"]) & (np.nan_to_num(rec["qc"]) == 0)
            & np.all(np.isfinite(rec["hofx"]), axis=1))
    rec = {k: (v[keep] if v.ndim == 1 else v[keep, :]) for k, v in rec.items()}

    meta = {"variable": variable, "files": len(paths),
            "members": len(hofx_groups), "n_after_qc": int(keep.sum()),
            "n_before_qc": int(keep.size),
            "span": (os.path.basename(paths[0]), os.path.basename(paths[-1]))}
    return rec, meta


# --------------------------------------------------------------------------
def group(rec, by=("obstype",), level_edges=None):
    """Split the records into strata.

    by : any of 'obstype', 'stationIdentification', 'level' (needs
         level_edges), or any metadata field present in the records.

    Note rec["hofx"] is 2-D (observations x members); the slicing below keeps
    that shape intact.
    """
    keys = []
    for name in by:
        if name == "level":
            if level_edges is None:
                raise ValueError("grouping by level needs level_edges")
            p = rec.get("pressure", rec.get("height"))
            if p is None:
                raise KeyError("no pressure or height metadata to bin by level")
            keys.append(np.digitize(p, np.asarray(level_edges)))
        elif name in rec:
            keys.append(rec[name])
        else:
            raise KeyError(f"'{name}' not present; available: "
                           f"{sorted(k for k in rec)}")

    tags = list(zip(*[np.asarray(k).tolist() for k in keys]))
    idx = defaultdict(list)
    for i, t in enumerate(tags):
        idx[t].append(i)
    out = {}
    for t, ii in idx.items():
        ii = np.asarray(ii)
        out[t] = {k: (v[ii] if v.ndim == 1 else v[ii, :])
                  for k, v in rec.items()}
    return out


def samples(sub):
    """Return (Y, X) for the estimator:

        Y = ensemble innovations   y - H(x_k),  stacked over members
        X = ensemble perturbations H(x_k) - H(x_bar), stacked over members

    Both are error-scale, so the variability of the field cancels before the
    deconvolution ever sees the data.
    """
    hofx = sub["hofx"]                       # (n_obs, K)
    obs = sub["obs"][:, None]
    mbar = hofx.mean(axis=1, keepdims=True)
    Y = (obs - hofx).ravel(order="F")
    X = (hofx - mbar).ravel(order="F")
    return Y, X


def report(sub, name="", min_n=1000, good_n=10000):
    """Print a usability summary for one stratum."""
    Y, _X = samples(sub)
    n = int(sub["obs"].size)
    verdict = ("ample" if n >= good_n else
               "usable, tails will be noisy" if n >= min_n else
               "TOO FEW -- pool more cycles or widen the stratum")
    print(f"  {str(name):<28} obs={n:>7d}  innov={Y.size:>8d}  "
          f"bins~{int(np.sqrt(Y.size)):>4d}  {verdict}")
    return n


def homogeneity(sub, split_on="file"):
    """Cheap check that a stratum is not a mixture of populations: split it in
    two by `split_on` and compare the spread of the innovations. A large
    difference means the stratum still mixes distinct error populations, and the
    apparent non-Gaussianity may be that mixture rather than a real heavy tail.
    """
    d = sub["obs"] - sub["hofx"].mean(axis=1)
    u = np.unique(sub[split_on])
    if u.size < 2:
        return None
    half = u[:u.size // 2]
    a = d[np.isin(sub[split_on], half)]
    b = d[~np.isin(sub[split_on], half)]
    if a.size < 50 or b.size < 50:
        return None
    ratio = float(np.nanstd(a) / np.nanstd(b))
    return {"std_a": float(np.nanstd(a)), "std_b": float(np.nanstd(b)),
            "ratio": ratio,
            "note": ("consistent" if 0.8 < ratio < 1.25 else
                     "INHOMOGENEOUS -- stratify further before trusting a "
                     "heavy-tailed result")}
