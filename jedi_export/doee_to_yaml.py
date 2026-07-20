#!/usr/bin/env python3
"""
Export a stable_DOEE observation-error density as a JEDI `non gaussian cost`
YAML block for CostJoEvolvingGaussian.

The density format is the stable_DOEE cache: a piecewise-linear log density on
a stable interior, with quadratic-log tails. Only the log-density SLOPES are
needed, because the cost function uses the score
    g(d) = d/dd[-log f](d)
and the effective variance of Hu, Geer & van Leeuwen (2025) Eq. 9
    sigma_o^2(d) = (d - m) / g(d).
Normalisation and intercepts do not enter, so logZ is not exported.

Three things this does that a naive dump would not:

1. MODE. Eq. 9 is 0/0 at the mode and the surrogate Gaussian is centred there,
   so the mode must be identified, not assumed to be zero. Taken as the argmax
   of the reconstructed log density.

2. SIGMA AT MODE. From the paper's Appendix A: fit a quadratic in a window
   about the mode and take sigma = sqrt(1/(2 p2)). This is the value used
   inside the mode neighbourhood, where Eq. 9 is unusable.

3. UNIMODALITY. The cost function REJECTS a density whose log slope is
   negative below the mode or positive above it, because the effective variance
   would be negative there. A raw deconvolved estimate is noisy and can violate
   this where samples are few. `enforce='monotone'` projects the slopes onto
   the nearest sign-correct sequence; `enforce='strict'` raises instead. The
   paper likewise fits a smooth sigma_o rather than using the raw bin values,
   so this is part of the method rather than a fudge.
"""

import numpy as np

REQUIRED = ["dx", "stable_min", "stable_max", "slopes_log",
            "left_log_slope", "left_dd", "right_log_slope", "right_dd"]


# --------------------------------------------------------------------------
# evaluator mirroring oops::NonGaussianDensity, for cross-checking the export
# --------------------------------------------------------------------------
class Density:
    def __init__(self, spec):
        self.m = spec["mode"]
        self.dx = spec["grid spacing"]
        self.lo = spec["stable min"]
        self.hi = spec["stable max"]
        self.sl = np.asarray(spec["log slopes"], float)
        self.lS = spec["left log slope"]
        self.lDD = spec["left curvature"]
        self.rS = spec["right log slope"]
        self.rDD = spec["right curvature"]
        self.sig = spec["sigma at mode"]
        self.win = spec.get("mode window", self.dx)
        self.floor = spec.get("sigma floor", 1.0e-3)

    def score(self, d):
        if d <= self.lo:
            return -(self.lS + self.lDD * (d - self.lo))
        if d >= self.hi:
            return -(self.rS + self.rDD * (d - self.hi))
        j = int((d - self.lo) / self.dx)
        j = max(0, min(j, self.sl.size - 1))
        return -self.sl[j]

    def variance(self, d):
        num = d - self.m
        if abs(num) <= self.win:
            return self.sig ** 2
        g = self.score(d)
        var = num / g if g != 0.0 else float("nan")
        if not (var > 0.0) or not np.isfinite(var):
            return self.sig ** 2
        return max(var, self.floor ** 2)


# --------------------------------------------------------------------------
def _centres(cache):
    n = len(cache["slopes_log"])
    return cache["stable_min"] + (np.arange(n) + 0.5) * cache["dx"]


def find_mode(cache):
    """Mode = argmax of the reconstructed log density.

    NOT the last positive slope: a single ragged bin (which deconvolution
    readily produces where it is ill-posed) would move that indicator far from
    the true mode, and every downstream quantity inherits the error. The argmax
    of the cumulative slope integral is insensitive to isolated bad bins.
    Refined by interpolating the slope zero crossing around the argmax."""
    s = np.asarray(cache["slopes_log"], float)
    c = _centres(cache)
    if s.size < 3:
        raise ValueError("need at least three interior bins to locate a mode")
    logf = np.concatenate([[0.0], np.cumsum(s[:-1] * np.diff(c))])
    j = int(np.argmax(logf))
    if j == 0 or j == s.size - 1:
        raise ValueError("mode is at the edge of the stable interior; widen it")
    s0, s1 = s[j - 1], s[j]
    if s0 > 0 > s1:                      # clean crossing: interpolate
        return float(c[j - 1] + (c[j] - c[j - 1]) * s0 / (s0 - s1))
    return float(c[j])


def sigma_at_mode(cache, mode, window=None):
    """Paper Appendix A: quadratic fit p2 (x-mode)^2 + p0 to -log f near the
    mode; sigma = sqrt(1/(2 p2)). Reconstructs -log f from the slopes, since
    the intercepts are not needed elsewhere."""
    c = _centres(cache)
    s = np.asarray(cache["slopes_log"], float)
    window = window if window is not None else 3.0 * cache["dx"]
    sel = np.abs(c - mode) <= window
    if sel.sum() < 3:
        sel = np.abs(c - mode) <= 5.0 * cache["dx"]
    if sel.sum() < 3:
        raise ValueError("too few bins near the mode for a quadratic fit")
    # cumulative integral of the slopes gives log f up to a constant
    logf = np.concatenate([[0.0], np.cumsum(s[:-1] * np.diff(c))])
    y = -logf[sel]
    x = c[sel] - mode
    # least squares on [x^2, 1]; the linear term vanishes at the mode
    A = np.vstack([x ** 2, np.ones_like(x)]).T
    p2, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    if not (p2 > 0):
        raise ValueError("quadratic fit at the mode is not convex (p2 <= 0); "
                         "the density may be multimodal or too noisy")
    return float(np.sqrt(1.0 / (2.0 * p2)))


def enforce_unimodal(slopes, centres, mode, how="monotone"):
    """The cost function requires slope > 0 below the mode and < 0 above it.
    A noisy estimate can violate this where samples are few."""
    s = np.asarray(slopes, float).copy()
    bad = ((centres < mode) & (s < 0)) | ((centres > mode) & (s > 0))
    if not bad.any():
        return s, 0
    if how == "strict":
        raise ValueError(f"{bad.sum()} log slopes violate unimodality about "
                         f"mode={mode:.4g}; pass enforce='monotone' to project them")
    # project onto the nearest sign-correct value: clear the bad entries, then
    # adopt the neighbouring valid slope so the score does not vanish (which
    # would make Eq. 9 undefined).
    s[bad] = 0.0
    for i in np.where(bad)[0]:
        nb = [s[k] for k in (i - 1, i + 1) if 0 <= k < s.size and not bad[k]]
        if nb:
            s[i] = min(nb, key=abs)
    return s, int(bad.sum())


# --------------------------------------------------------------------------
def to_spec(cache, *, mode=None, enforce="monotone", sigma_floor=1.0e-3,
            mode_window=None, save_sigma=False, sigma_group="EvolvingSigma"):
    """Build the `non gaussian cost` mapping from a finalized stable_DOEE cache.

    Returns (spec, nfixed). nfixed is how many log slopes had to be projected to
    satisfy unimodality; a large value means the estimate is too noisy to use
    as-is."""
    for k in REQUIRED:
        if k not in cache:
            raise KeyError(f"cache missing '{k}' (did you run finalize_pdf_cache?)")
    if cache["left_dd"] > 0 or cache["right_dd"] > 0:
        raise ValueError("tail curvatures must be <= 0; the density would grow "
                         "without bound")

    m = find_mode(cache) if mode is None else float(mode)
    c = _centres(cache)
    slopes, nfixed = enforce_unimodal(cache["slopes_log"], c, m, enforce)
    # sigma is fitted to the CLEANED slopes: fitting the raw ones would let a
    # single ragged bin set the weight given to every near-mode observation.
    sig = sigma_at_mode({**cache, "slopes_log": slopes}, m)
    win = mode_window if mode_window is not None else cache["dx"]

    spec = {
        "mode": m,
        "grid spacing": float(cache["dx"]),
        "stable min": float(cache["stable_min"]),
        "stable max": float(cache["stable_max"]),
        "log slopes": [float(x) for x in slopes],
        "left log slope": float(cache["left_log_slope"]),
        "left curvature": float(cache["left_dd"]),
        "right log slope": float(cache["right_log_slope"]),
        "right curvature": float(cache["right_dd"]),
        "sigma at mode": sig,
        "mode window": float(win),
        "sigma floor": float(sigma_floor),
    }
    if save_sigma:
        spec["save evolving sigma"] = True
        spec["evolving sigma group"] = sigma_group
    return spec, nfixed


def to_yaml(spec, indent=8, per_line=6):
    """Render as the `non gaussian cost` block, indented to sit under an
    observer entry in an RDASApp obtype template."""
    pad = " " * indent
    out = [f"{pad}non gaussian cost:"]
    p2 = pad + "  "
    for k in ["mode", "grid spacing", "stable min", "stable max"]:
        out.append(f"{p2}{k}: {spec[k]!r}")
    sl = spec["log slopes"]
    out.append(f"{p2}log slopes: [")
    for i in range(0, len(sl), per_line):
        chunk = ", ".join(f"{v:.8f}" for v in sl[i:i + per_line])
        tail = "," if i + per_line < len(sl) else ""
        out.append(f"{p2}  {chunk}{tail}")
    out.append(f"{p2}]")
    for k in ["left log slope", "left curvature", "right log slope",
              "right curvature", "sigma at mode", "mode window", "sigma floor"]:
        out.append(f"{p2}{k}: {spec[k]!r}")
    for k in ["save evolving sigma", "evolving sigma group"]:
        if k in spec:
            v = "true" if spec[k] is True else spec[k]
            out.append(f"{p2}{k}: {v}")
    return "\n".join(out)


def check(spec, dmax=None, n=2001):
    """Cross-check: the exported block must give a finite positive effective
    variance everywhere, using the same logic as the C++ evaluator. Returns a
    list of offending (d, variance) pairs, empty if the density is usable."""
    f = Density(spec)
    hi = dmax if dmax is not None else 3.0 * (spec["stable max"] - spec["mode"]) + 1.0
    lo = -3.0 * (spec["mode"] - spec["stable min"]) - 1.0
    bad = []
    for d in np.linspace(lo, hi, n):
        v = f.variance(float(d))
        if not (v > 0) or not np.isfinite(v):
            bad.append((float(d), v))
    return bad
