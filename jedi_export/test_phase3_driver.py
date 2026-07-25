#!/usr/bin/env python3
"""Offline end-to-end test of the Phase 3 driver: builds a stub build
tree (fabricated truth obt, template yamls from an oops checkout, stub
executables producing member obt outputs and flow logs in the real
formats), runs phase3_cycle.py through it, and asserts the recorded
table, the per-iteration density npz files, and the flow monitor.
Catches interface bugs before any edit reaches a real machine.

    python3 test_phase3_driver.py --oops /path/to/oops-checkout
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_MPIEXEC = r"""
#!/usr/bin/env python3
# Stub mpiexec+l95_eda.x: reads the umbrella + member yamls, fabricates
# each member's output obt (hofx0..K, oman, ombg columns) and a run log
# with norm/Jo lines, exercising every parsing path in the driver.
import re, sys, numpy as np
args = sys.argv[1:]
n = int(args[args.index("-n") + 1])
yaml = args[-1]
files = re.findall(r"- (testinput/\S+)", open(yaml).read())
rng = np.random.default_rng(11)
hdr = open("Data/phase3_noisy.obt").read().splitlines()
ncol = int(hdr[0]); names = hdr[1:1+ncol]; nobs = int(hdr[1+ncol])
body = [ln.split() for ln in hdr[2+ncol:2+ncol+nobs]]
jv = 3 + names.index("ObsValue")
y = np.array([float(r[jv]) for r in body])
K = 11
for m, f in enumerate(files, 1):
    dep0 = 1.6*rng.standard_normal(nobs)
    cols, vals = [], []
    for k in range(K):
        d = dep0*(0.35/1.6 + (1.6-0.35)/1.6*(1-k/(K-1)))
        cols.append(f"hofx{k}"); vals.append(y - d)
    cols += ["hofx", "oman", "ombg", "ObsValue", "ObsError"]
    vals += [y, y-(vals[K-1]*0-1)*0 - (y-vals[K-1]), dep0*0+y-(y-dep0),
             y, np.full(nobs, 0.4)]
    # oman = final departures; ombg = initial
    vals[K+1] = y - vals[K-1]
    vals[K+2] = dep0
    out = f.replace("testinput/phase3_mem",
                    "Data/mem").replace(".yaml",
                    ".phase3.2010-01-02T00:00:00Z.obt")
    with open(out, "w") as g:
        g.write(f"{len(cols)}\n")
        for c in cols: g.write(c+"\n")
        g.write(f"{nobs}\n")
        for i in range(nobs):
            g.write(f"{i}  2010-01-02T00:00:00Z  {(i%40)/40:.6f}  "
                    + "  ".join(repr(float(v[i])) for v in vals) + "\n")
log = ["iteration: %d, eps: 0.05 norm: %.1f" % (k, 100*np.exp(-k/60))
       for k in range(210)]
log += ["CostJo   : Nonlinear Jo(Lorenz 95) = %.2f, nobs = 120" % j
        for j in (1044.0, 400.0, 120.0)]
sys.stdout.write("\n".join(log) + "\n")

"""

STUB_GENPERT = "#!/usr/bin/env python3\nimport sys\nprint('stub ok')\n"


def build_tree(root):
    bt = os.path.join(root, "l95", "test")
    os.makedirs(os.path.join(bt, "testinput"))
    os.makedirs(os.path.join(bt, "Data"))
    binp = os.path.join(root, "bin")
    os.makedirs(binp)
    rng = np.random.default_rng(3)
    xt = 2.0 * rng.standard_normal(40)
    with open(os.path.join(
            bt, "Data", "truth3d.2010-01-02T00:00:00Z.obt"), "w") as f:
        f.write("3\nObsError\nObsValue\nhofx\n120\n")
        i = 0
        for t in ("2010-01-01T22:30:00Z", "2010-01-02T00:00:00Z",
                  "2010-01-02T01:30:00Z"):
            for L in range(40):
                v = float(xt[L])
                f.write(f"{i}  {t}  {L/40:.6f}  0.4  {v!r}  {v!r}\n")
                i += 1
    with open(os.path.join(binp, "l95_genpert.x"), "w") as f:
        f.write(STUB_GENPERT)
    fm = os.path.join(binp, "fake_mpiexec")
    with open(fm, "w") as f:
        f.write(FAKE_MPIEXEC.lstrip())
    for p in (os.path.join(binp, "l95_genpert.x"), fm):
        os.chmod(p, 0o755)
    return bt, fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oops", required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as root:
        bt, fm = build_tree(root)
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "phase3_cycle.py"),
             "--build", bt, "--oops", os.path.expanduser(a.oops),
             "--members", "6", "--density", "heavy", "--iters", "2",
             "--seed", "7", "--pff-outer", "30", "--pff-ctcheck", "999",
             "--mpiexec", fm],
            capture_output=True, text=True)
        out = r.stdout + r.stderr
        ok = True

        def check(name, cond):
            nonlocal ok
            print(f"  {'PASS' if cond else 'FAIL'}  {name}")
            ok = ok and cond

        check("driver exits 0", r.returncode == 0)
        check("two table rows",
              out.count("it 0:") == 1 and out.count("it 1:") == 1)
        check("flow monitor parsed", "flow[norm" in out)
        check("table log written",
              os.path.exists(os.path.join(bt, "phase3_table.log")))
        check("density npz saved", all(
            os.path.exists(os.path.join(bt, f"phase3_it{i}_density.npz"))
            for i in (0, 1)))
        check("final spec written",
              os.path.exists(os.path.join(bt, "phase3_final_spec.yaml")))
        if not ok:
            print(out[-2500:])
            raise SystemExit(1)
        print("ALL PASS")


if __name__ == "__main__":
    main()
