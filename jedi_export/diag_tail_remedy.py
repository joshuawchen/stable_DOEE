#!/usr/bin/env python3
"""Remedy test for the tail-curvature feedback drift.

E0  baseline depth-6 (drifts; pinned above at 0.158 -> 0.459)
E1  steep-symmetric continuation: after gaussian_tails, both sides
    take the STEEPER of the two fitted curvatures -- the asymmetric
    noise kick in the outer-mass polyfit cannot open one tail alone
E2  refresh control (rung-2 semantics: new error draw each iteration,
    same truth) at the unmodified tail policy
E3  refresh + steep-symmetric together

Success criterion for E1: the loop holds the closed-export level
(L1 ~0.12-0.16) through depth 6 with no release iteration.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "l95-testbed"))
sys.path.insert(0, os.path.join(HERE, ".."))

import l95_mirror as M                          # noqa: E402
import stage_c_smoothing as SC                  # noqa: E402

NPZ = os.path.join(HERE, "vm_data", "phase3_fields.npz")
_ORIG = SC.gaussian_tails


def steep_symmetric(cache, xg, pi, sd, rep, **kw):
    _ORIG(cache, xg, pi, sd, rep, **kw)
    dd = min(cache["left_dd"], cache["right_dd"])   # steeper wins
    cache["left_dd"] = cache["right_dd"] = dd


def report(tag, rows):
    print(f"  {tag}:")
    for i, m in enumerate(rows):
        if m.spec is not None:
            st = (f"{np.sqrt(-1 / m.spec['left curvature']):.2f}/"
                  f"{np.sqrt(-1 / m.spec['right curvature']):.2f}")
        else:
            st = "REFUSED"
        print(f"    it {i}: sd {m.sd:.3f}  L1 {m.l1:.3f}  "
              f"ESS {m.ess:.0f}  core {m.core:.2f}  tail {m.tail:.2f}  "
              f"tail-sig {st}")


def main():
    for tag, patch, refresh in (
            ("E1 steep-symmetric, fixed data", True, False),
            ("E2 refresh, stock tails", False, True),
            ("E3 refresh + steep-symmetric", True, True)):
        SC.gaussian_tails = steep_symmetric if patch else _ORIG
        rows = M.run_loop(members=40, iters=6, seed=7, fields=NPZ,
                          refresh=refresh, quiet=True)
        report(tag, rows)
        SC.gaussian_tails = _ORIG


if __name__ == "__main__":
    main()
