#!/usr/bin/env python3
"""Generate NUMERIC Format A parity fixtures: exported specs plus
reference score/variance tables for the four menu densities, evaluated
by the Python Density mirror. The C++ ctest loads this file (JSON is
valid YAML, so eckit::YAMLConfiguration reads it directly), constructs
oops::NonGaussianDensity from each case's "spec", and compares
score(d) and variance(d) against "reference" at tolerance "tol".

This extends fixtures.json (validation parity: accept/reject) to
evaluation parity: the two implementations must agree NUMERICALLY on
every branch of the evaluator -- deep tails, tail junctions, interior
bins, the mode window, and the variance fallback. Everything here is
deterministic: caches are built from the ANALYTIC menu densities on a
fixed grid (the oracle_cache construction, self-contained below), so
regeneration is byte-stable and estimation noise never enters the
contract.

Reference d values are placed at bin FRACTION 0.37, never on a bin
edge, so float->int truncation in the bin lookup cannot differ between
implementations.

    python3 make_parity_fixtures.py            # writes parity_fixtures.json
    python3 make_parity_fixtures.py --check    # regenerate and diff
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import doee_to_yaml as DY  # noqa: E402

OUT = os.path.join(HERE, "parity_fixtures.json")
TOL = 1e-9

# the menu densities at scale 1.0, exactly as the l95 injector defines
# them (map_reference.draw_errors); mirrored_gamma is retired
MENU = {
    "gaussian": {"kind": "gaussian", "sigma": 0.4},
    "heavy": {"kind": "mixture", "w": 0.85, "sigma1": 0.25, "sigma2": 1.0},
    "skewed": {"kind": "skewmix", "w": 0.8, "m1": -0.1, "s1": 0.25,
               "m2": 0.4, "s2": 0.45},
    "laplace": {"kind": "laplace", "b": 0.3},
}


def density(spec, x):
    x = np.asarray(x, float)
    k = spec["kind"]
    if k == "gaussian":
        s = spec["sigma"]
        return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))
    if k == "mixture":
        w, s1, s2 = spec["w"], spec["sigma1"], spec["sigma2"]
        return (w * np.exp(-0.5 * (x / s1) ** 2) / (s1 * np.sqrt(2 * np.pi))
                + (1 - w) * np.exp(-0.5 * (x / s2) ** 2)
                / (s2 * np.sqrt(2 * np.pi)))
    if k == "skewmix":
        w, m1, s1 = spec["w"], spec["m1"], spec["s1"]
        m2, s2 = spec["m2"], spec["s2"]
        return (w * np.exp(-0.5 * ((x - m1) / s1) ** 2)
                / (s1 * np.sqrt(2 * np.pi))
                + (1 - w) * np.exp(-0.5 * ((x - m2) / s2) ** 2)
                / (s2 * np.sqrt(2 * np.pi)))
    if k == "laplace":
        b = spec["b"]
        return np.exp(-np.abs(x) / b) / (2 * b)
    raise ValueError(k)


def sample_sigma(spec):
    k = spec["kind"]
    if k == "gaussian":
        return spec["sigma"]
    if k == "mixture":
        return float(np.sqrt(spec["w"] * spec["sigma1"] ** 2
                             + (1 - spec["w"]) * spec["sigma2"] ** 2))
    if k == "skewmix":
        w = spec["w"]
        return float(np.sqrt(w * (spec["s1"] ** 2 + spec["m1"] ** 2)
                             + (1 - w) * (spec["s2"] ** 2
                                          + spec["m2"] ** 2)))
    if k == "laplace":
        return float(np.sqrt(2.0) * spec["b"])
    raise ValueError(k)


def oracle_cache(spec):
    """stable_DOEE-style cache from the analytic density (grid-point
    convention), the self-contained twin of stage_a_end_to_end's."""
    s = sample_sigma(spec)
    dx = s / 10.0
    half = 5.0 * s
    n = 2 * int(round(half / dx)) + 1
    xg = (np.arange(n) - (n - 1) / 2) * dx
    f = density(spec, xg)
    keep = f > 1e-12
    padded = np.r_[0, keep.astype(int), 0]
    d = np.diff(padded)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    k = int(np.argmax([f[s0:e0].sum() for s0, e0 in zip(starts, ends)]))
    xg, f = xg[starts[k]:ends[k]], f[starts[k]:ends[k]]
    logf = np.log(f)
    slopes = np.empty_like(logf)
    slopes[0] = (logf[1] - logf[0]) / dx
    slopes[-1] = (logf[-1] - logf[-2]) / dx
    slopes[1:-1] = (logf[2:] - logf[:-2]) / (2 * dx)
    return {"dx": dx, "stable_min": float(xg[0]),
            "stable_max": float(xg[-1]), "slopes_log": slopes,
            "left_log_slope": float(slopes[0]),
            "left_dd": min(float((slopes[1] - slopes[0]) / dx), 0.0),
            "right_log_slope": float(slopes[-1]),
            "right_dd": min(float((slopes[-1] - slopes[-2]) / dx), 0.0),
            "kept_mass": 1.0}


def reference_points(spec_out):
    """d values covering every evaluator branch, never on a bin edge:
    deep tails, both junctions, a spread of interior bins, and the mode
    window (variance's analytic-sigma branch)."""
    lo, hi = spec_out["stable min"], spec_out["stable max"]
    dx = spec_out["grid spacing"]
    m, win = spec_out["mode"], spec_out["mode window"]
    nb = int(round((hi - lo) / dx))
    ds = [lo - 1.5, lo - 0.4, lo - 0.25 * dx]
    for k in np.linspace(1, nb - 2, 15).astype(int):
        ds.append(lo + (k + 0.37) * dx)
    ds += [m - 0.49 * win, m + 0.49 * win]
    ds += [hi + 0.25 * dx, hi + 0.4, hi + 1.5]
    return sorted(float(d) for d in ds)


def _round(o):
    """12 significant digits everywhere: regeneration is then byte-stable
    across platforms and BLAS builds (lstsq differs in the last ulps),
    while the 1e-9 parity tolerance is untouched."""
    if isinstance(o, float):
        return float(f"{o:.12g}")
    if isinstance(o, list):
        return [_round(v) for v in o]
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    return o


def build_case(name, inj):
    cache = oracle_cache(inj)
    spec, nfixed = DY.to_spec(cache)
    bad = DY.check(spec)
    if bad:
        raise RuntimeError(f"{name}: exported spec fails check(): "
                           f"{bad[:3]}")
    den = DY.Density(spec)
    ds = reference_points(spec)
    return {"name": name, "injected": inj, "nfixed": int(nfixed),
            "spec": spec,
            "reference": {
                "d": ds,
                "score": [den.score(d) for d in ds],
                "variance": [den.variance(d) for d in ds]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="regenerate and diff against the committed file")
    a = ap.parse_args()
    doc = {"tol": TOL,
           "note": "evaluation-parity fixtures for oops::"
                   "NonGaussianDensity; regenerate with "
                   "make_parity_fixtures.py; JSON is valid YAML for "
                   "eckit::YAMLConfiguration",
           "cases": [build_case(n, s) for n, s in MENU.items()]}
    text = json.dumps(_round(doc), indent=1, sort_keys=True)
    if a.check:
        with open(OUT) as f:
            if f.read() != text + "\n":
                print("parity_fixtures.json is STALE: regenerate")
                return 1
        print("parity_fixtures.json is current")
        return 0
    with open(OUT, "w") as f:
        f.write(text + "\n")
    for case in doc["cases"]:
        sp = case["spec"]
        print(f"  {case['name']:9s} mode {sp['mode']:+.4f} "
              f"sigma@mode {sp['sigma at mode']:.4f} "
              f"interior [{sp['stable min']:+.3f}, "
              f"{sp['stable max']:+.3f}] "
              f"slopes {len(sp['log slopes'])} nfixed {case['nfixed']} "
              f"refs {len(case['reference']['d'])}")
    print(f"wrote {OUT} (tol {TOL:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
