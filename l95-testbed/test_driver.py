#!/usr/bin/env python3
"""Regression battery for stage_a_end_to_end's synthetic mode.

The synthetic mode exercises every link of the driver except the model
binaries -- collection, estimation (both estimators), the null floor,
the tail/junction export policy, the gates, the density plot, and the
config writing -- so it pins the driver surface that grew on
2026-07-21: the --adaptive branch, the per-run density plot, and the
gaussian_tails export (windowed tail curvature capped at
Gaussian-of-3-recovered-sigmas, interior retracted to the central 98%
of mass per side, which is where the DA actually evaluates).

Runs the driver as a subprocess the way a user does; each case asserts
on the report text, which is the driver's contract with its reader.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail
                                                     else ""))
    if not ok:
        FAILS.append(name)


def run(extra, timeout=600):
    r = subprocess.run(
        [sys.executable, str(HERE / "stage_a_end_to_end.py"), "--synthetic",
         "--floor-trials", "2"] + extra,
        cwd=str(HERE), capture_output=True, text=True, timeout=timeout)
    return r


# A. legacy synthetic passes end to end with the export policy applied
r = run([])
check("legacy synthetic passes", r.returncode == 0
      and "exit 0 (PASS)" in r.stdout)
check("legacy export retracted to the mass junction",
      "export interior retracted" in r.stdout)

# B. adaptive synthetic passes; the selection trace and plot are reported
r = run(["--adaptive"])
check("adaptive synthetic passes", r.returncode == 0
      and "exit 0 (PASS)" in r.stdout)
check("adaptive selection trace present", "cross-split argmin" in r.stdout)
check("tail sigma reported per side", r.stdout.count("tail sigma") == 2)
check("density plot written or matplotlib noted",
      "density plot written" in r.stdout
      or "matplotlib is not installed" in r.stdout)
work = HERE / "testbed_work"
check("recovered npz on disk", (work / "testbed_recovered.npz").exists())

# C. the exported tails are at least Gaussian: every reported curvature
# is at or below the cap of -1/(3 sd)^2 (parse the report lines)
caps_ok = True
for line in r.stdout.splitlines():
    if "_dd: exported" in line:
        dd = float(line.split("exported")[1].split("(")[0])
        sd = float(line.rstrip(")").split("recovered sd")[1])
        caps_ok &= dd <= -1.0 / (3.0 * sd) ** 2 + 1e-9
check("exported curvatures respect the Gaussian cap", caps_ok)

# D. --adaptive refuses --lam
r = run(["--adaptive", "--lam", "30"])
check("--adaptive refuses --lam", r.returncode != 0
      and "no meaning" in r.stderr)

# E. oracle mode passes through the same export policy
r = run(["--oracle-density"])
check("oracle synthetic passes", r.returncode == 0
      and "exit 0 (PASS)" in r.stdout)
check("oracle export also retracted",
      "export interior retracted" in r.stdout)

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
