#!/usr/bin/env python3
"""Extract the mirror's calibration data from a phase-3 build tree into
one compact npz: y, injected r, H(truth) at all obs times, and every
member's background H(x_b) -- the real JEDI fields the offline twin
runs on instead of synthesized statistics.

    python3 extract_mirror_data.py --build ~/jedi/src/build/oops/l95/test \
        --members 40 --out jedi_export/vm_data/phase3_fields.npz
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "l95-testbed"))
from inject_obs_error import read_obt  # noqa: E402


def col(names, rows, name):
    j = 3 + names.index(name)
    return np.array([float(r[j]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--members", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(
        HERE, "vm_data", "phase3_fields.npz"))
    a = ap.parse_args()
    b = os.path.expanduser(a.build)
    names, rows, n = read_obt(
        os.path.join(b, "Data", "phase3_noisy.obt"))
    y = col(names, rows, "ObsValue")
    htruth = col(names, rows, "hofx")          # stale = H(truth(t_obs))
    times = [r[1] for r in rows]
    locs = np.array([float(r[2]) for r in rows])
    Xb = np.empty((a.members, n))
    oman = np.empty((a.members, n))
    for k in range(1, a.members + 1):
        nm, rw, _ = read_obt(os.path.join(
            b, "Data", f"mem{k:03d}.phase3.2010-01-02T00:00:00Z.obt"))
        Xb[k - 1] = col(nm, rw, "hofx0")       # H(background)
        oman[k - 1] = col(nm, rw, "oman")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, y=y, htruth=htruth, Xb=Xb, oman=oman,
                        times=np.array(times), locs=locs)
    kb = os.path.getsize(a.out) / 1024
    print(f"wrote {a.out} ({kb:.0f} KB): y/htruth ({n}), "
          f"Xb/oman ({a.members}x{n})")


if __name__ == "__main__":
    main()
