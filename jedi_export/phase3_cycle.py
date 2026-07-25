#!/usr/bin/env python3
"""Phase 3, rung 1: the first closed DOEE loop through real JEDI.

Fixed l95 window, iterated density estimation -- the JEDI twin of the
sandbox's single-window pipeline (stage_c --pipeline). Per iteration:

  1. run the shared-likelihood ensemble analysis (N members, obs
     perturbation amplitude 0 -- the pff-calibration template) under
     the CURRENT Format A density via l95_eda.x
  2. read each member's analysis departures (oman) from its obt output
  3. LOO-reweight the members per observation under the same density
     the analysis used (defended weights, delta 0.01)
  4. adaptive DOEE on the (y, hofx) rows; export through doee_to_yaml
     with the self-gap refusal gate
  5. write the next iteration's `non gaussian cost` block

The observation errors are INJECTED from a known menu density
(inject_obs_error.py), so every iteration's L1 against the truth is
measurable -- the loop's convergence is a number, not an impression.
Iteration 0 analyzes under the DEGENERATE Gaussian spec (sigma 0.4,
exact tails: the proven-equivalent configuration), so the whole run
exercises only the non-Gaussian code path.

    python3 phase3_cycle.py --build ~/jedi/src/build/oops/l95/test \\
        --oops ~/jedi/src/oops --members 40 --density heavy \\
        --iters 6 --seed 7 [--dry-run]

FIRST-CONTACT NOTES (expect one-line fixes, as with every branch):
the oman column name in the member obt outputs; mpiexec oversubscribe
syntax for members > cores; genenspert runtime at large N. --dry-run
prints every command without running anything.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "l95-testbed"))
sys.path.insert(0, os.path.join(HERE, ".."))

import doee_to_yaml as DY                                  # noqa: E402
from make_parity_fixtures import density as menu_density   # noqa: E402
from inject_obs_error import inject, read_obt              # noqa: E402
from map_reference import spec_nll                         # noqa: E402
from stage_c_smoothing import density_to_spec, loo_hofx    # noqa: E402

GAUSS0 = {
    "mode": 0.0, "grid spacing": 2.0e-6, "stable min": -1.0e-6,
    "stable max": 1.0e-6, "log slopes": [0.0],
    "left log slope": 6.25e-6, "left curvature": -6.25,
    "right log slope": -6.25e-6, "right curvature": -6.25,
    "sigma at mode": 0.4, "mode window": 0.0, "sigma floor": 1.0e-3,
}


def sh(cmd, a, cwd):
    print(f"  $ {cmd}")
    if a.dry_run:
        return
    r = subprocess.run(cmd, shell=True, cwd=cwd,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"command failed: {cmd}")


def block_from_spec(spec):
    """The observer-level `non gaussian cost` block, indented for the
    member yaml (observer entries sit at 6 spaces in the l95 configs)."""
    return DY.to_yaml(spec, indent=6)


def set_outer_iterations(y, T):
    """Replicate the first outer-iteration block T times: the flow's
    cumulative transport goes like eps*T/N (the update carries 1/N and
    both eps and the iteration budget are fixed in the template), so
    the budget must scale with member count."""
    head, sep, rest = y.partition("  iterations:\n")
    if not sep:
        return y
    m = re.search(r"^\S", rest, re.M)
    body, tail = rest[:m.start()], rest[m.start():]
    items = [it for it in re.split(r"(?=^  - )", body, flags=re.M)
             if it.strip()]
    return head + sep + items[0] * T + tail


def member_yaml(base, n, nmem, blk, a):
    """One member's yaml from the calibration template: member number,
    noisy obs in, per-member obs out, output exp, and the density block
    inserted under the observer. An empty blk means plain Gaussian Jo
    (the --jo gaussian discriminator): no block, no jo type."""
    y = base
    y = y.replace("forecast.ens.1.", f"forecast.ens.{n}.")
    y = y.replace("mem001.pff_calibration", f"mem{n:03d}.phase3")
    y = y.replace("exp: pff_calibration.mem001", f"exp: phase3.mem{n:03d}")
    y = re.sub(r"obsdatain:\n(\s+)obsfile: [^\n]+",
               lambda m: f"obsdatain:\n{m.group(1)}obsfile: "
                         f"Data/phase3_noisy.obt", y)
    y = re.sub(r"eps: [0-9.eE+-]+", f"eps: {a.pff_eps:g}", y)
    y = re.sub(r"ct check: \d+", f"ct check: {a.pff_ctcheck}", y)
    if a.pff_bandwidth_sd != 0.6:
        # ONLY the minimizer's kernel-bandwidth key (the LAST
        # standard_deviation in the file); the earlier one is the B
        # covariance and must stay 0.6 -- the prior weight is not a
        # tuning knob, the kernel bandwidth is
        head, _, tail = y.rpartition("standard_deviation: 0.6")
        y = (head + f"standard_deviation: {a.pff_bandwidth_sd:g}"
             + tail)
    if a.pff_outer > 0:
        y = set_outer_iterations(y, a.pff_outer)
    if blk:
        y = y.replace("      obs operator: {}",
                      "      obs operator: {}\n" + blk, 1)
        if "jo type" not in y:
            y = y.replace("  observations:\n    observers:",
                          "  observations:\n    jo type: evolving "
                          "gaussian\n    observers:")
    y = re.sub(r"\ntest:\n(  [^\n]+\n?)+", "\n", y)
    return y


def collect_oman(a, it):
    """Departures y - H(x_analysis) per member from the obt outputs."""
    deps = []
    names_seen = None
    for n in range(1, a.members + 1):
        path = os.path.join(a.build, "Data",
                            f"mem{n:03d}.phase3.2010-01-02T00:00:00Z.obt")
        names, rows, _ = read_obt(path)
        names_seen = names
        cand = [j for j, nm in enumerate(names) if "oman" in nm.lower()]
        if not cand:
            raise SystemExit(
                f"no oman column in {path}; columns: {names}")
        deps.append(np.array([float(r[3 + cand[-1]]) for r in rows]))
    print(f"    [it {it}] columns available: {names_seen}")
    return np.column_stack(deps)          # n_obs x members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True,
                    help="the build's l95/test dir (has Data/, "
                         "testinput/, ../../bin/)")
    ap.add_argument("--oops", required=True, help="oops checkout")
    ap.add_argument("--members", type=int, default=40)
    ap.add_argument("--density", default="heavy",
                    choices=["gaussian", "heavy", "skewed", "laplace"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--loo-defense", type=float, default=0.01)
    ap.add_argument("--export-gap-max", type=float, default=0.4)
    ap.add_argument("--assumed-error", type=float, default=0.4)
    ap.add_argument("--pff-eps", type=float, default=0.05,
                    help="initial flow learning rate. The componentwise "
                         "kernel's 1-D neighbor spacing shrinks ~1/N "
                         "while the repulsion prefactor grows ~N, so "
                         "the stable eps ceiling FALLS with member "
                         "count (N=4 stable at 0.05; N=40 diverges); "
                         "start small and let the x1.5 schedule climb")
    ap.add_argument("--pff-bandwidth-sd", type=float, default=0.6,
                    help="the minimizer's kernel-bandwidth SD (h^2 = "
                         "SD^2/N). The paper's construction assumes "
                         "the ensemble spread EQUALS this; ours grew "
                         "to ~1.6 in 24h, so the default 0.6 breaks "
                         "the repulsion/attraction balance once the "
                         "componentwise kernels revive at large N")
    ap.add_argument("--pff-ctcheck", type=int, default=7,
                    help="iterations of stable norm before the x1.5 "
                         "eps growth. The ratchet has NO ceiling and "
                         "the norm check is partially blind, so on "
                         "long budgets it walks eps into instability "
                         "and the redo branch cannot roll positions "
                         "back; a huge value disables growth for "
                         "fixed-eps descent with a budgetable "
                         "transport eps*T/N")
    ap.add_argument("--pff-outer", type=int, default=0,
                    help="outer-iteration budget (0 = template count). "
                         "Transport goes like eps*T/N, so T must scale "
                         "with member count at fixed eps")
    ap.add_argument("--jo", default="nongaussian",
                    choices=["nongaussian", "gaussian"],
                    help="gaussian = plain Gaussian Jo (no Format A "
                         "block): the discriminator for the PFF x "
                         "CostJoNonGaussian pairing")
    ap.add_argument("--mpiexec", default="mpiexec --oversubscribe")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.build = os.path.expanduser(a.build)
    a.oops = os.path.expanduser(a.oops)
    binp = os.path.join(a.build, "..", "..", "bin")

    print(f"phase 3 rung 1: density {a.density}, members {a.members}, "
          f"iters {a.iters}, seed {a.seed}")

    # --- ensemble backgrounds at the requested member count ------------
    gy = open(os.path.join(a.oops,
              "l95/test/testinput/genenspert.yaml")).read()
    gy = re.sub(r"members: \d+", f"members: {a.members}", gy)
    gy = re.sub(r"\ntest:\n(  [^\n]+\n?)+", "\n", gy)
    gpath = os.path.join(a.build, "testinput", "phase3_genens.yaml")
    if not a.dry_run:
        open(gpath, "w").write(gy)
    sh(f"{binp}/l95_genpert.x testinput/phase3_genens.yaml "
       f"> phase3_genens.log 2>&1", a, a.build)

    # --- inject the known error density into the truth obs -------------
    inj = None
    if not a.dry_run:
        inj = inject(os.path.join(a.build, "Data",
                                  "truth3d.2010-01-02T00:00:00Z.obt"),
                     os.path.join(a.build, "Data", "phase3_noisy.obt"),
                     density=a.density, scale=a.scale, seed=a.seed)
        print(f"  injected {inj['density']}: sample sigma "
              f"{inj['sample_sigma']:.3f}")

    base = open(os.path.join(
        a.oops, "l95/test/testinput/pff_calibration_1.yaml")).read()

    est_args = SimpleNamespace(lam=None, adaptive=True,
                               assumed_error=a.assumed_error,
                               export_gap_max=a.export_gap_max)
    spec = dict(GAUSS0)
    rng = np.random.default_rng(a.seed + 100)
    fine = np.arange(-6.0, 6.0001, 0.01)

    for it in range(a.iters):
        blk = block_from_spec(spec) if a.jo == "nongaussian" else ""
        files = []
        for n in range(1, a.members + 1):
            my = member_yaml(base, n, a.members, blk, a)
            p = os.path.join(a.build, "testinput",
                             f"phase3_mem{n:03d}.yaml")
            if not a.dry_run:
                open(p, "w").write(my)
            files.append(f"testinput/phase3_mem{n:03d}.yaml")
        up = os.path.join(a.build, "testinput", "phase3_eda.yaml")
        if not a.dry_run:
            open(up, "w").write(
                "files:\n" + "".join(f"- {f}\n" for f in files))
        sh(f"{a.mpiexec} -n {a.members} {binp}/l95_eda.x "
           f"testinput/phase3_eda.yaml > phase3_it{it}.log 2>&1",
           a, a.build)
        if a.dry_run:
            print(f"  [it {it}] (dry run: collection and estimation "
                  f"skipped)")
            continue

        warn = 0
        logp = os.path.join(a.build, f"phase3_it{it}.log")
        if os.path.exists(logp):
            warn = open(logp).read().count("JoJc is negative")

        dep = collect_oman(a, it)                     # y - H(x_a)
        names, rows, _ = read_obt(os.path.join(a.build, "Data",
                                               "phase3_noisy.obt"))
        jv = [j for j, nm in enumerate(names) if nm == "ObsValue"][0]
        y = np.array([float(r[3 + jv]) for r in rows])
        hofx = y[:, None] - dep

        nll_e, _ = spec_nll(spec, 12.0 * a.assumed_error)
        h_loo, ess = loo_hofx(y, hofx, nll_e, a.members,
                              np.random.default_rng(a.seed + 7 + it),
                              a.loo_defense)
        new_spec, sd, (xg, pi) = density_to_spec(y, h_loo, est_args,
                                                 a.seed + it)
        pt = menu_density(inj, fine) if inj else None
        pe = np.interp(fine, xg, pi, left=0.0, right=0.0)
        l1 = (float(np.trapezoid(np.abs(pe - pt), fine))
              if pt is not None else float("nan"))
        gate = "accepted" if new_spec is not None else "REFUSED"
        print(f"  it {it}: sd {sd:.3f}  L1(truth) {l1:.3f}  "
              f"ESS {ess:.0f}  export {gate}  jojc-warnings {warn}")
        if new_spec is not None:
            spec = new_spec

    if not a.dry_run:
        out = os.path.join(a.build, "phase3_final_spec.yaml")
        open(out, "w").write(DY.to_yaml(spec, indent=8) + "\n")
        print(f"final Format A block: {out}")


if __name__ == "__main__":
    main()
