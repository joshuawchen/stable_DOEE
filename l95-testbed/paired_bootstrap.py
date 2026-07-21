#!/usr/bin/env python3
"""Paired block bootstrap for the regret margin between two DA arms.

The driver reports mean regrets; differences between arms at one
configuration are a few thousandths of a nat, and observations are
spatially correlated, so whether a margin is real needs a confidence
interval that respects both the pairing and the correlation. This
script reads two analysis .obt files that assimilated the SAME
observations (control and treatment of one run, or the same arm from
two runs at identical seed and configuration, e.g. a --junction-frac
sweep), scores each observation by the true-density log score exactly
as the driver does (oman - eps against the injected spec), and
bootstraps the PAIRED per-observation differences in circular moving
blocks, with the block length set by the differences' own integrated
autocorrelation time. Reported: each arm's regret, the margin
(arm A minus arm B; positive means B is better), its 95% CI, the block
length used, and P(margin > 0) over the resamples.

    python3 paired_bootstrap.py DATA_DIR
    python3 paired_bootstrap.py DATA_DIR --arm-b testbed_treatment.obt \
        --data-b OTHER_RUN_DIR          # cross-run, same seed/config

--selftest validates the machinery on AR(1) differences with known
mean before it is trusted on real logs: 95% CI coverage within
Monte-Carlo error, and the naive iid interval shown undercovering on
the same data, which is why the blocks exist.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_ensemble as CE                      # noqa: E402
from stage_a_end_to_end import true_log_density    # noqa: E402


def per_ob_regret(data_dir, obt_name):
    """Per-observation regret under the injected truth, driver-identical:
    oman = y - H(x_a), y = H(truth) + eps, so the noise-free departure is
    oman - eps, scored by -log f_true relative to a perfect analysis."""
    data_dir = Path(data_dir)
    saved = np.load(data_dir / "testbed_truth.npz")
    eps = np.asarray(saved["errors"], float)
    inj = json.loads(str(saved["spec"]))
    rec = CE.read_obt(str(data_dir / obt_name))
    if len(rec["oman"]) != len(eps):
        raise SystemExit(f"{obt_name} has {len(rec['oman'])} rows but "
                         f"{len(eps)} errors were injected: not the same "
                         "run's observations")
    d_an = np.asarray(rec["oman"], float) - eps
    ll0 = float(true_log_density(inj, np.zeros(1))[0])
    return ll0 - true_log_density(inj, d_an)


def block_length(delta, cap=None):
    """2x the integrated autocorrelation time of the paired differences,
    summing positive lags until the autocorrelation drops below 0.05."""
    x = np.asarray(delta, float)
    x = x - x.mean()
    v = float(np.mean(x * x))
    if v <= 0:
        return 1
    tau = 1.0
    for k in range(1, min(x.size // 4, 200)):
        r = float(np.mean(x[:-k] * x[k:])) / v
        if r < 0.05:
            break
        tau += 2.0 * r
    return int(max(1, min(np.ceil(2.0 * tau), cap or x.size // 10)))


def moving_block_ci(delta, n_boot=4000, alpha=0.05, block=None, seed=0):
    """Circular moving-block bootstrap of the mean. Returns
    (mean, lo, hi, block, p_positive)."""
    d = np.asarray(delta, float)
    n = d.size
    L = block or block_length(d)
    nb = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    means = d[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(d.mean()), float(lo), float(hi), L,
            float(np.mean(means > 0.0)))


def selftest():
    """Coverage on AR(1) with known mean; the iid interval undercovers."""
    rng = np.random.default_rng(7)
    n, phi, mu, trials = 1200, 0.6, 0.004, 150
    hits_blk = hits_iid = 0
    for t in range(trials):
        e = rng.normal(0, 0.05, n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        x += mu
        m, lo, hi, L, _ = moving_block_ci(x, n_boot=800, seed=t)
        hits_blk += int(lo <= mu <= hi)
        se = x.std(ddof=1) / np.sqrt(n)
        hits_iid += int(m - 1.96 * se <= mu <= m + 1.96 * se)
    cov_blk, cov_iid = hits_blk / trials, hits_iid / trials
    print(f"block CI coverage {cov_blk:.2f} (target 0.95)  "
          f"iid CI coverage {cov_iid:.2f} (undercovers under phi={phi})")
    ok = 0.90 <= cov_blk <= 0.99 and cov_iid < cov_blk
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", type=Path,
                    help="Data dir holding testbed_truth.npz and the .obt "
                         "analyses")
    ap.add_argument("--arm-a", default="testbed_control.obt")
    ap.add_argument("--arm-b", default="testbed_treatment.obt")
    ap.add_argument("--data-b", type=Path, default=None,
                    help="Data dir for arm B when comparing across runs at "
                         "identical seed and configuration")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--block", type=int, default=None,
                    help="override the autocorrelation-chosen block length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.data is None:
        ap.error("a Data dir is required (or --selftest)")

    ra = per_ob_regret(a.data, a.arm_a)
    rb = per_ob_regret(a.data_b or a.data, a.arm_b)
    if ra.size != rb.size:
        raise SystemExit("the two arms score different observation counts; "
                         "they are not the same run's observations")
    delta = ra - rb                     # positive: arm B is better
    m, lo, hi, L, p_pos = moving_block_ci(delta, n_boot=a.n_boot,
                                          block=a.block, seed=a.seed)
    print(f"arm A ({a.arm_a}): regret {ra.mean():.4f} nats/ob")
    print(f"arm B ({a.arm_b}): regret {rb.mean():.4f} nats/ob")
    print(f"margin (A - B): {m:+.4f} nats/ob  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"(block {L}, {a.n_boot} resamples, n {delta.size})")
    print(f"P(margin > 0) = {p_pos:.3f}")
    if lo > 0:
        print("verdict: arm B beats arm A at 95% on this realization")
    elif hi < 0:
        print("verdict: arm A beats arm B at 95% on this realization")
    else:
        print("verdict: statistical tie at 95% on this realization")


if __name__ == "__main__":
    main()
