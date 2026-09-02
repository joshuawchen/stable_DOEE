#!/usr/bin/env python3
"""Plot a saved testbed_recovered.npz against the analytic injected truth.

The driver writes the recovered density to Data/testbed_recovered.npz on
every run (including gate failures, whose estimates are exactly the ones
worth looking at) and overwrites it on the next run; this script turns a
rescued copy into the same noise-density picture the driver now embeds:
linear and log panels, truth against estimate.

    python3 plot_recovered.py ~/d800_recovered.npz --density heavy --scale 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a_end_to_end import analytic_density  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--density", default="heavy",
                    choices=["gaussian", "heavy", "laplace",
                             "mirrored_gamma"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--truth-npz", type=Path, default=None,
                    help="testbed_truth.npz from the same run; when given, "
                         "its resolved spec is used and --density/--scale "
                         "are ignored")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(a.npz)
    xg, pi = d["xg"], d["pi"]
    dx = xg[1] - xg[0]
    p = np.maximum(pi, 0.0)
    tot = p.sum() * dx
    if tot > 0:
        p = p / tot
    if a.truth_npz is not None:
        import json
        inj = json.loads(str(np.load(a.truth_npz)["spec"]))
    else:
        # resolve the CLI name to a spec the same way the injector does
        import inject_obs_error as IJ
        rng = np.random.default_rng(0)
        _, inj = IJ.draw(a.density, rng, 8, a.scale)
    tru = analytic_density(inj, xg)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].plot(xg, tru, "k-", lw=1.8, label="injected truth")
    ax[0].plot(xg, p, "-", color="tab:red", lw=1.4,
               label=f"estimate (lam/mu {float(d['lam']):.1e})")
    ax[0].set_title("noise density")
    ax[0].legend(frameon=False)
    ax[1].semilogy(xg, np.maximum(tru, 1e-7), "k-", lw=1.8)
    ax[1].semilogy(xg, np.maximum(p, 1e-7), "-", color="tab:red", lw=1.4)
    ax[1].set_ylim(1e-6, None)
    ax[1].set_title("log scale (tails and satellites)")
    for a_ in ax:
        a_.set_xlabel("obs error")
        a_.set_xlim(xg[0], xg[-1])
    fig.tight_layout()
    out = a.out or a.npz.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
