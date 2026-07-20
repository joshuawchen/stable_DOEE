#!/usr/bin/env python3
"""JEDI diagnostic files -> DOEE density -> JEDI YAML.

    # survey first: what strata exist, and is any of them big enough?
    python3 run_doee_export.py --diags 'diags/jdiag_adpsfc_*.nc' \
        --variable airTemperature \
        --members hofx_mem001 hofx_mem002 hofx_mem003 --survey

    # then estimate and write, for ONE stratum
    python3 run_doee_export.py --diags 'diags/jdiag_adpsfc_*.nc' \
        --variable airTemperature \
        --members hofx_mem001 hofx_mem002 hofx_mem003 \
        --obstype 181 \
        --obtype-yaml .../obtype_config/adpsfc_airTemperature_181.yaml \
        --basic-yaml  .../basic_config/mpasjedi_hybrid3denvar.yaml

The two YAML edits are in different files and do different things: the density
goes in the obtype template, and `jo type: evolving gaussian` goes under
`observations:` in the basic config, which is what switches the cost function
on. Without the second the density is inert.

The estimator is imported from the stable_doee module at the repository root.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import samples as S
import patch_yaml as P
import doee_to_yaml as X

MIN_N = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diags", required=True, help="glob for IODA diag files")
    ap.add_argument("--variable", required=True)
    ap.add_argument("--members", nargs="+", required=True,
                    help="H(x) group per ensemble member; at least two")
    ap.add_argument("--qc-group", default="EffectiveQC0")
    ap.add_argument("--group-by", nargs="*", default=["obstype"],
                    help="stratification axes, e.g. obstype level")
    ap.add_argument("--level-edges", nargs="*", type=float, default=None)
    ap.add_argument("--obstype", type=float, default=None,
                    help="select a single stratum by ObsType")
    ap.add_argument("--survey", action="store_true",
                    help="report strata and counts, estimate nothing")
    ap.add_argument("--obtype-yaml", help="template to receive the density")
    ap.add_argument("--basic-yaml", help="basic config to receive `jo type`")
    ap.add_argument("--indent", type=int, default=6)
    ap.add_argument("--enforce", choices=["monotone", "strict"],
                    default="monotone")
    ap.add_argument("--mode", type=float, default=None)
    ap.add_argument("--save-sigma", action="store_true")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rec, meta = S.load(a.diags, a.variable, tuple(a.members), a.qc_group)
    print(f"{meta['variable']}: {meta['n_after_qc']} of {meta['n_before_qc']} "
          f"observations passed QC, from {meta['files']} files x "
          f"{meta['members']} members")
    print(f"  span {meta['span'][0]} .. {meta['span'][1]}")

    strata = S.group(rec, tuple(a.group_by), a.level_edges)
    print(f"\nstrata by {a.group_by}:")
    for key in sorted(strata, key=lambda k: -len(strata[k]["obs"])):
        S.report(strata[key], key, min_n=a.min_n)
        h = S.homogeneity(strata[key])
        if h and "INHOM" in h["note"]:
            print(f"      spread differs across the window "
                  f"({h['std_a']:.3g} vs {h['std_b']:.3g}); {h['note']}")

    if a.survey:
        print("\nSurvey only. Choose a stratum with --obstype, or stratify "
              "further with --group-by, before estimating.")
        return 0

    if a.obstype is None:
        if len(strata) != 1:
            print("\nSeveral strata present; select one with --obstype, or "
                  "re-run per stratum. Pooling them would mix populations with "
                  "different error characteristics and manufacture apparent "
                  "non-Gaussianity.")
            return 1
        sub = next(iter(strata.values()))
    else:
        hit = [v for k, v in strata.items() if float(k[0]) == a.obstype]
        if not hit:
            print(f"\nno stratum with ObsType {a.obstype}")
            return 1
        sub = hit[0]

    Y, Xs = S.samples(sub)
    n_obs = int(sub["obs"].size)
    if n_obs < a.min_n:
        print(f"\n{n_obs} observations is below --min-n={a.min_n}. The tails "
              "would be noise, and the tails are the point of the method. "
              "Pool more cycles.")
        return 1

    import stable_doee as C
    print(f"\nestimating from {Y.size} innovations and {Xs.size} perturbations "
          f"({n_obs} observations x {meta['members']} members)")
    grid, pi, cache = C.estimate_noise_pmf(Xs, Y)
    cache = C.finalize_pdf_cache(grid, cache)

    spec, nfixed = X.to_spec(cache, mode=a.mode, enforce=a.enforce,
                             save_sigma=a.save_sigma)
    bad = X.check(spec)
    print(f"mode = {spec['mode']:+.4f}   sigma at mode = {spec['sigma at mode']:.4f}"
          f"   slopes projected = {nfixed}")
    if nfixed > max(3, 0.05 * len(spec["log slopes"])):
        print(f"  WARNING: {nfixed} slopes projected. The estimate is noisy; "
              "widen the bins or pool more cycles rather than assimilating "
              "the projection.")
    if bad:
        print(f"  ERROR: effective variance invalid at {len(bad)} points, "
              f"e.g. {bad[:3]}")
        return 1

    block = X.to_yaml(spec, indent=a.indent)
    if a.dry_run or not a.obtype_yaml:
        print("\n" + block)
        return 0

    action, _ = P.patch_file(a.obtype_yaml, block)
    print(f"\n{action} `non gaussian cost` in {a.obtype_yaml}")
    if a.basic_yaml:
        act2, _ = P.ensure_jo_type_file(a.basic_yaml)
        print(f"{act2} `jo type: evolving gaussian` in {a.basic_yaml}")
    else:
        print("NOTE: no --basic-yaml given, so the method is NOT switched on. "
              "The density is inert until `jo type: evolving gaussian` is set "
              "under `observations:` in the basic config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
