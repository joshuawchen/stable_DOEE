#!/usr/bin/env python3
"""Inject a known observation-error density into an l95 observation file.

l95 generates observations by saving H(truth) directly -- `make obs: true` in
the HofX app does `hofx.save("ObsValue")` and adds no noise, while `obs_error`
in the generate block only fills the ObsError column with the assumed error. So
the generated file holds the truth in observation space, and any error density
can be added to it.

That is what makes the closed loop testable: the density that went in is known
exactly, so a recovered density can be compared against it rather than against
another estimate.

    python3 inject_obs_error.py truth3d.obt noisy.obt --density heavy --seed 1

The draws and the density parameters are written alongside as an npz, so a later
comparison does not depend on regenerating the same random numbers.

File format (plain text):
    line 1        number of data columns
    lines 2..1+n  column names
    line 2+n      number of observations
    remaining     index  datetime  location  col1 ... coln
"""

import argparse
import json

import numpy as np


def draw(name, rng, n, scale=1.0):
    """Draws from a chosen density, with a description of what was drawn."""
    if name == "gaussian":
        s = 0.4 * scale
        return rng.normal(0.0, s, n), {"kind": "gaussian", "sigma": s}
    if name == "heavy":                      # 85/15 variance mixture
        s1, s2, w = 0.25 * scale, 1.0 * scale, 0.85
        pick = rng.random(n) < w
        e = np.where(pick, rng.normal(0, s1, n), rng.normal(0, s2, n))
        return e, {"kind": "mixture", "w": w, "sigma1": s1, "sigma2": s2}
    if name == "laplace":
        b = 0.3 * scale
        return rng.laplace(0.0, b, n), {"kind": "laplace", "b": b}
    if name == "mirrored_gamma":
        # the density of the reference's idealised experiment, rescaled:
        #   f(x) = -(x-2)/4 exp((x-2)/2) for x <= 2, mode at 0
        g = rng.gamma(shape=2.0, scale=2.0, size=n)
        e = 2.0 - g
        e = e * (0.4 * scale / e.std())
        return e, {"kind": "mirrored_gamma", "shape": 2.0, "scale": 2.0,
                   "rescaled_to_sigma": 0.4 * scale}
    raise ValueError(f"unknown density '{name}'")


def read_obt(path):
    with open(path) as f:
        lines = f.read().splitlines()
    ncol = int(lines[0].strip())
    names = [lines[1 + i].strip() for i in range(ncol)]
    nobs = int(lines[1 + ncol].strip())
    rows = [ln.split() for ln in lines[2 + ncol: 2 + ncol + nobs]]
    if len(rows) != nobs:
        raise ValueError(f"{path}: header says {nobs} observations, "
                         f"found {len(rows)}")
    return names, rows, nobs


def write_obt(path, names, rows):
    with open(path, "w") as f:
        f.write(f"{len(names)}\n")
        for nm in names:
            f.write(f"{nm}\n")
        f.write(f"{len(rows)}\n")
        for r in rows:
            f.write("  ".join(r) + "\n")


def inject(in_path, out_path, density="heavy", scale=1.0, seed=0,
           assumed_error=None, truth_path=None):
    names, rows, nobs = read_obt(in_path)
    if "ObsValue" not in names:
        raise KeyError(f"{in_path}: no ObsValue column (found {names})")
    col_of = {nm: 3 + i for i, nm in enumerate(names)}   # index, time, location
    iv = col_of["ObsValue"]

    rng = np.random.default_rng(seed)
    eps, spec = draw(density, rng, nobs, scale)

    truth_vals = np.array([float(r[iv]) for r in rows])
    for j, r in enumerate(rows):
        r[iv] = repr(float(truth_vals[j] + eps[j]))
        if assumed_error is not None and "ObsError" in col_of:
            r[col_of["ObsError"]] = repr(float(assumed_error))

    write_obt(out_path, names, rows)

    spec.update({"density": density, "scale": scale, "seed": seed, "n": nobs,
                 "sample_sigma": float(eps.std()),
                 "sample_excess_kurtosis":
                     float(((eps - eps.mean()) ** 4).mean()
                           / eps.var() ** 2 - 3)})
    if truth_path:
        np.savez(truth_path, errors=eps, truth_obs=truth_vals,
                 spec=json.dumps(spec))
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--density", default="heavy",
                    choices=["gaussian", "heavy", "laplace", "mirrored_gamma"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assumed-error", type=float, default=None,
                    help="value written to the ObsError column, i.e. what a "
                         "Gaussian control run will assume")
    ap.add_argument("--truth-out", default=None,
                    help="npz holding the injected draws and the density spec")
    a = ap.parse_args()
    spec = inject(a.infile, a.outfile, a.density, a.scale, a.seed,
                  a.assumed_error, a.truth_out)
    print(f"injected {spec['density']} into {spec['n']} observations")
    print(f"  sample sigma           {spec['sample_sigma']:.4f}")
    print(f"  sample excess kurtosis {spec['sample_excess_kurtosis']:+.3f}")
    if a.assumed_error is not None:
        print(f"  ObsError column set to {a.assumed_error} "
              "(what a Gaussian control assumes)")


if __name__ == "__main__":
    main()
