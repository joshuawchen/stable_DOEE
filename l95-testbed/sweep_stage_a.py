#!/usr/bin/env python3
"""Replication sweep for Stage A: many seeds, per-seed stamps, and the
across-seed interval that turns "wins on this realization" into a claim.

Every Stage A number so far is conditional on seed 7; its bootstrap CI
covers within-realization noise only. This harness runs the driver over
a seed list (and optionally several obs densities), preserves each
run's scoring artifacts (the Data dir is single-slot and overwrites),
stamps each realization with the paired block bootstrap, and reports
per density: each seed's margin and CI, the win/tie/refusal tally, and
the across-seed mean with a t interval over S independent replications
-- the primary statistic, since seeds are genuine replications while
observations within one seed are not.

    python3 sweep_stage_a.py --build ~/jedi/src/build/oops \
        --densities 200,400 --seeds 1,2,3,4,5,6,7,8 --adaptive \
        --pert-sd 0.20 --members 50 --scale 2 --assumed-error 0.9 \
        --out ~/runs/sweep_v1

    python3 sweep_stage_a.py --stamp-only ~/runs/sweep_v1/d200_s*

Gate refusals are recorded and excluded from the interval (reported as
refusals, which are themselves a finding), and a run dir without DA
outputs is recorded as missing rather than crashing the sweep.
--synthetic dry-runs the launch loop against the fabricated testbed
(no DA binaries, so runs stamp as missing by design).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_bootstrap import per_ob_regret, moving_block_ci  # noqa: E402

ARTIFACTS = ("testbed_control.obt", "testbed_treatment.obt",
             "testbed_truth.npz")
# two-sided 97.5% t quantiles by degrees of freedom; 1.96 beyond
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
        8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16,
        14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 18: 2.10, 19: 2.09}


def stamp(run_dir):
    """Bootstrap stamp of one preserved run dir. Returns a dict with
    status 'ok'|'missing', and for 'ok' the margin, CI, and regrets."""
    run_dir = Path(run_dir)
    if not all((run_dir / a).exists() for a in ARTIFACTS):
        return {"dir": str(run_dir), "status": "missing"}
    ra = per_ob_regret(run_dir, "testbed_control.obt")
    rb = per_ob_regret(run_dir, "testbed_treatment.obt")
    m, lo, hi, L, p = moving_block_ci(ra - rb)
    return {"dir": str(run_dir), "status": "ok", "margin": m, "lo": lo,
            "hi": hi, "block": L, "p_pos": p,
            "regret_ctl": float(ra.mean()), "regret_trt": float(rb.mean())}


def launch(a, density, seed, out_dir):
    """One driver run; preserve artifacts; return the stamp record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "stage_a_end_to_end.py",
           "--obs-density", str(density), "--seed", str(seed),
           "--members", str(a.members), "--pert-sd", str(a.pert_sd),
           "--scale", str(a.scale), "--assumed-error", str(a.assumed_error),
           "--floor-trials", str(a.floor_trials),
           "--junction-frac", str(a.junction_frac)]
    if a.build:
        cmd += ["--build", a.build]
    if a.adaptive:
        cmd += ["--adaptive"]
    if a.synthetic:
        cmd += ["--synthetic"]
    log = out_dir / "driver.log"
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent),
                           stdout=f, stderr=subprocess.STDOUT)
    if a.build:
        data = Path(a.build).expanduser() / "l95" / "test" / "Data"
    else:
        data = Path(__file__).resolve().parent / "testbed_work"
    for art in ARTIFACTS:
        src = data / art
        if src.exists():
            shutil.copy2(src, out_dir / art)
    rec = stamp(out_dir)
    if r.returncode != 0:
        reason = ""
        for line in log.read_text().splitlines():
            if "hard failure" in line or "FAIL " in line:
                reason = line.strip()
                break
        rec = {"dir": str(out_dir), "status": "refused", "reason": reason}
    rec.update({"density": density, "seed": seed})
    return rec


def summarize(records, label=""):
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\n=== {label}  ({len(ok)} stamped, "
          f"{sum(r['status'] == 'refused' for r in records)} refused, "
          f"{sum(r['status'] == 'missing' for r in records)} missing)")
    for r in records:
        if r["status"] == "ok":
            verdict = ("win" if r["lo"] > 0 else
                       "loss" if r["hi"] < 0 else "tie")
            seed = f"seed {r['seed']}" if "seed" in r else Path(r["dir"]).name
            print(f"  {seed:>12}: margin {r['margin']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {verdict}"
                  f"  (ctl {r['regret_ctl']:.4f} trt {r['regret_trt']:.4f})")
        else:
            tag = f"seed {r.get('seed', '?')}" if "seed" in r \
                else Path(r["dir"]).name
            print(f"  {tag:>12}: {r['status']}"
                  + (f"  {r.get('reason', '')}" if r.get("reason") else ""))
    if len(ok) >= 2:
        ms = np.array([r["margin"] for r in ok])
        se = ms.std(ddof=1) / np.sqrt(ms.size)
        t = T975.get(ms.size - 1, 1.96)
        print(f"  across-seed mean margin {ms.mean():+.4f} "
              f"+- {t * se:.4f} (t 95%, S={ms.size})  "
              f"wins {sum(r['lo'] > 0 for r in ok)}"
              f"/ties {sum(r['lo'] <= 0 <= r['hi'] for r in ok)}"
              f"/losses {sum(r['hi'] < 0 for r in ok)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=None)
    ap.add_argument("--densities", default="200")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--pert-sd", type=float, default=0.20)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--assumed-error", type=float, default=0.9)
    ap.add_argument("--floor-trials", type=int, default=4)
    ap.add_argument("--junction-frac", type=float, default=0.98)
    ap.add_argument("--out", type=Path, default=Path("sweep_out"))
    ap.add_argument("--stamp-only", nargs="*", default=None,
                    help="skip launching; stamp these existing run dirs")
    a = ap.parse_args()

    if a.stamp_only is not None:
        recs = [stamp(d) for d in a.stamp_only]
        summarize(recs, "stamp-only")
        return

    densities = [int(x) for x in a.densities.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    for density in densities:
        recs = []
        for seed in seeds:
            out_dir = Path(a.out).expanduser() / f"d{density}_s{seed}"
            print(f"--- d{density} seed {seed} -> {out_dir}", flush=True)
            recs.append(launch(a, density, seed, out_dir))
        summarize(recs, f"obs-density {density}")


if __name__ == "__main__":
    main()
