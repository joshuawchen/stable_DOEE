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
Normalization and intercepts do not enter, so logZ is not exported.

Three things this does that a naive dump would not:

1. MODE. Eq. 9 is 0/0 at the mode and the surrogate Gaussian is centered there,
   so the mode must be identified, not assumed to be zero. Taken as the argmax
   of the reconstructed log density.

2. SIGMA AT MODE. From the paper's Appendix A: fit a quadratic in a window
   about the mode and take sigma = sqrt(1/(2 p2)). This is the value used
   inside the mode neighborhood, where Eq. 9 is unusable.

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
def _centers(cache):
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
    c = _centers(cache)
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
    c = _centers(cache)
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


def enforce_unimodal(slopes, centers, mode, how="monotone"):
    """The cost function requires slope > 0 below the mode and < 0 above it.
    A noisy estimate can violate this where samples are few.

    Runs of wrong-sign slopes (a noise lobe in a tail) are projected by
    LINEAR INTERPOLATION between the flanking valid slopes, not by copying
    a neighbor: at a lobe boundary the neighbors are the small slopes, and
    copying them leaves a near-zero-slope shelf on which the effective
    variance (d - m)/g(d) explodes -- measured on the record ensemble,
    two 1%-mass lobes at |d| ~ 2.5 produced EvolvingSigma up to 10.8
    against a true ceiling of 2, and the treatment lost its edge exactly
    on the moderate-tail observations the method exists for."""
    s = np.asarray(slopes, float).copy()
    bad = ((centers < mode) & (s < 0)) | ((centers > mode) & (s > 0))
    if not bad.any():
        return s, 0
    if how == "strict":
        raise ValueError(f"{bad.sum()} log slopes violate unimodality about "
                         f"mode={mode:.4g}; pass enforce='monotone' to project them")
    good = np.where(~bad)[0]
    for i in np.where(bad)[0]:
        left = good[good < i]
        right = good[good > i]
        if left.size and right.size:
            l, r = int(left[-1]), int(right[0])
            t = (centers[i] - centers[l]) / (centers[r] - centers[l])
            s[i] = (1 - t) * s[l] + t * s[r]
        elif left.size:
            s[i] = s[int(left[-1])]
        elif right.size:
            s[i] = s[int(right[0])]
        else:
            s[i] = 0.0
        # interpolation between valid flanks can still cross zero right at
        # the mode; clamp to the correct sign with a tiny magnitude
        if centers[i] < mode and s[i] < 0:
            s[i] = 1e-12
        elif centers[i] > mode and s[i] > 0:
            s[i] = -1e-12
    return s, int(bad.sum())


# --------------------------------------------------------------------------
def to_spec(cache, *, mode=None, enforce="monotone", sigma_floor=1.0e-3,
            mode_window=None, save_sigma=False, sigma_group="EvolvingSigma",
            reflect=True):
    """Build the `non gaussian cost` mapping from a finalized stable_DOEE cache.

    Returns (spec, nfixed). nfixed is how many log slopes had to be projected to
    satisfy unimodality; a large value means the estimate is too noisy to use
    as-is.

    VARIABLE REFLECTION. The estimator works in the innovation y - H(x); the
    C++ evaluates the density at H(x) - y. The exported spec is therefore the
    reflection: mode negated, interval negated and swapped, slopes reversed
    and negated, tails swapped with slopes negated and curvatures kept.
    Symmetric densities are unchanged up to roundoff. Pinned empirically by
    the l95 runs: a density at mode +0.3 in the code's variable matches a
    Gaussian control with obs bias -0.3 and differs from +0.3. reflect=False
    exports the innovation-space density as-is.

    CONVENTION TRANSLATION. The C++ NonGaussianDensity treats stable min and
    stable max as bin EDGES and requires exactly (max-min)/dx log slopes. The
    estimator caches (stable_doee.finalize_pdf_cache and
    stable_doee_reg._make_cache) instead store the FIRST AND LAST GRID POINT
    of the kept interior -- bin centers -- so their extent is (n-1)*dx for n
    slopes, and the C++ rejected the first estimated density to reach it with
    "log slopes has 34 entries but the grid implies 33 bins" (the hand-written
    test configurations had always obeyed the edge convention). Both
    conventions are accepted here and detected from the extent: (n-1)*dx is
    padded by half a bin per side, which also makes _centers reconstruct the
    original grid points exactly, removing a half-bin shift the mode and
    sigma fits carried for estimator caches; n*dx is kept as-is. Either way
    the upper edge is rebuilt as lo + n*dx so the count identity holds in the
    exported floats, not just mathematically, and any other extent is
    refused."""
    for k in REQUIRED:
        if k not in cache:
            raise KeyError(f"cache missing '{k}' (did you run finalize_pdf_cache?)")
    if cache["left_dd"] > 0 or cache["right_dd"] > 0:
        raise ValueError("tail curvatures must be <= 0; the density would grow "
                         "without bound")

    dxv = float(cache["dx"])
    nsl = len(cache["slopes_log"])
    nb = (float(cache["stable_max"]) - float(cache["stable_min"])) / dxv
    if abs(nb - (nsl - 1)) < 1e-6:        # grid-point (bin center) convention
        lo = float(cache["stable_min"]) - 0.5 * dxv
    elif abs(nb - nsl) < 1e-6:            # already the edge convention
        lo = float(cache["stable_min"])
    else:
        raise ValueError(f"cache is inconsistent: {nsl} log slopes over an "
                         f"extent of {nb:.6f} bins; expected {nsl} (edges) "
                         f"or {nsl - 1} (grid points)")
    hi = lo + nsl * dxv
    cache = {**cache, "stable_min": lo, "stable_max": hi}

    m = find_mode(cache) if mode is None else float(mode)
    c = _centers(cache)
    slopes, nfixed = enforce_unimodal(cache["slopes_log"], c, m, enforce)
    # sigma is fitted to the CLEANED slopes: fitting the raw ones would let a
    # single ragged bin set the weight given to every near-mode observation.
    sig = sigma_at_mode({**cache, "slopes_log": slopes}, m)
    win = mode_window if mode_window is not None else cache["dx"]

    lS, lDD = cache["left_log_slope"], cache["left_dd"]
    rS, rDD = cache["right_log_slope"], cache["right_dd"]
    lo_out, hi_out = cache["stable_min"], cache["stable_max"]
    if reflect:
        m = -m
        lo_out, hi_out = -cache["stable_max"], -cache["stable_min"]
        slopes = -np.asarray(slopes)[::-1]
        lS, lDD, rS, rDD = -rS, rDD, -lS, lDD

    spec = {
        "mode": m,
        "grid spacing": float(cache["dx"]),
        "stable min": float(lo_out),
        "stable max": float(hi_out),
        "log slopes": [float(x) for x in slopes],
        "left log slope": float(lS),
        "left curvature": float(lDD),
        "right log slope": float(rS),
        "right curvature": float(rDD),
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


def validate_spec(spec):
    """Mirror of the C++ NonGaussianDensity::validate(). Returns a list of
    problems, empty if the C++ will accept the spec. Kept in lockstep with
    the C++ via jedi_export/fixtures.json, which both sides consume."""
    out = []
    dx = spec["grid spacing"]
    lo, hi = spec["stable min"], spec["stable max"]
    sl = spec["log slopes"]
    if not dx > 0:
        out.append("grid spacing must be > 0")
    if not hi > lo:
        out.append("stable max must exceed stable min")
    if not spec["sigma at mode"] > 0:
        out.append("sigma at mode must be > 0")
    if spec.get("sigma floor", 0.0) < 0:
        out.append("sigma floor must be >= 0")
    if len(sl) == 0:
        out.append("log slopes must not be empty")
        return out
    nb = (hi - lo) / dx
    if abs(nb - len(sl)) > 1e-6:
        out.append(f"log slopes has {len(sl)} entries but the grid implies "
                   f"{nb:.6f} bins")
    m = spec["mode"]
    if m < lo or m > hi:
        out.append("mode lies outside [stable min, stable max]")
    if spec["left curvature"] > 0 or spec["right curvature"] > 0:
        out.append("tail curvatures must be <= 0")
    for j, s in enumerate(sl):
        c = lo + (j + 0.5) * dx
        if c < m and s < 0:
            out.append(f"log slope {j} is negative below the mode")
            break
        if c > m and s > 0:
            out.append(f"log slope {j} is positive above the mode")
            break
    return out


def check(spec, dmax=None, n=2001):
    """Cross-check: the exported block must pass the same validation the C++
    applies on load (validate_spec) and must give a finite positive effective
    variance everywhere, using the same logic as the C++ evaluator. Returns a
    list of offending entries, empty if the density is usable."""
    bad = [("validate", msg) for msg in validate_spec(spec)]
    if bad:
        return bad
    f = Density(spec)
    hi = dmax if dmax is not None else 3.0 * (spec["stable max"] - spec["mode"]) + 1.0
    lo = -3.0 * (spec["mode"] - spec["stable min"]) - 1.0
    for d in np.linspace(lo, hi, n):
        v = f.variance(float(d))
        if not (v > 0) or not np.isfinite(v):
            bad.append((float(d), v))
    return bad
