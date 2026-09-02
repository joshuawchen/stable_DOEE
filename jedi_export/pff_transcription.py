#!/usr/bin/env python3
"""Verbatim numpy transcription of JEDI's PFF (oops PFF.h, conformant
flow as merged at pff-norm-fix): the offline instrument that found the
particle-0 catapult and validated the eps*T/N transport law.

Update law per outer iteration, per particle i (exact structure):
  rr_j   = H^T R^-1 (y - H x_j)            (Jo-only, current positions)
  F_i    = (1/N) sum_j [ K_ij o (B rr_j - (x_j - xbar_b))
                         + [j != i] (x_i - x_j) o K_ij / alpha ]
  K_ij   = exp(-(x_i - x_j)^2 / (2 h^2)) componentwise, h^2 = SD^2/N
  alpha  = 1/N;  xbar_b = the FIXED background mean
with the eps controller: backtrack /1.5 on >2% norm growth (dx zeroed,
positions NOT rolled back), x1.5 growth after ct-check stable
iterations. B is the ErrorCovarianceL95 spectral-Gaussian circulant
(sigma^2 exp(-0.5 d^2), d in gridpoints).

--catapult replays the pre-fix computeNorm defect: rank 0's update is
replaced by the N-particle SUM (the by-reference clobber), and each
rank's controller watches a different norm.

--selftest asserts the session's pinned findings.
"""

import argparse

import numpy as np


def make_B(d=40, sigb2=0.36):
    idx = np.arange(d)
    dist = np.minimum(np.abs(idx[:, None] - idx[None, :]),
                      d - np.abs(idx[:, None] - idx[None, :]))
    C = np.exp(-0.5 * dist ** 2)
    return sigb2 * C, C


def run(N, SD, eps0, T, ctCheck, catapult=False, seed=7, d=40,
        common=1.3, indiv=0.85, oerr=0.345, R=0.16, trace=None):
    rng = np.random.default_rng(seed)
    B, C = make_B(d)
    lam, U = np.linalg.eigh(C)
    Ch = U @ np.diag(np.sqrt(np.maximum(lam, 0))) @ U.T
    xt = 2.0 * rng.standard_normal(d)
    y = np.repeat(xt, 3) + oerr * rng.standard_normal(3 * d)
    xan = xt + common * (Ch @ rng.standard_normal(d))
    X = xan[None] + indiv * (rng.standard_normal((N, d)) @ Ch.T)
    xbar = X.mean(0)
    alpha = 1.0 / N
    h2 = alpha * SD * SD
    eps = np.full(N, eps0)
    ct = np.zeros(N, int)
    n1 = np.full(N, -1.0)
    for it in range(T):
        dob = y[None] - np.repeat(X, 3, axis=1)
        rr = dob.reshape(N, d, 3).sum(2) / R
        F = np.zeros_like(X)
        for i in range(N):
            acc = np.zeros(d)
            for j in range(N):
                Kc = np.exp(-(X[i] - X[j]) ** 2 / (2 * h2))
                t = Kc * (B @ rr[j] - (X[j] - xbar))
                if j != i:
                    t += (X[i] - X[j]) * Kc / alpha
                acc += t
            F[i] = acc / N
        Fsum = F.sum(0)
        for i in range(N):
            Fi = Fsum if (catapult and i == 0) else F[i]
            nr = (np.sqrt((Fsum ** 2).sum() / (N * d))
                  if (catapult and i == 0)
                  else np.sqrt((F[i] ** 2).sum() / (N * d)))
            if not catapult:
                # post-fix: one collective norm for every controller
                nr = np.sqrt((Fsum ** 2).sum() / (N * d))
            if n1[i] < 0:
                n1[i] = nr; X[i] += eps[i] * Fi; ct[i] += 1
            elif nr > 1.02 * n1[i]:
                eps[i] /= 1.5; ct[i] = 0
            elif ct[i] >= ctCheck:
                ct[i] = 0; X[i] += eps[i] * 1.5 * Fi
            else:
                X[i] += eps[i] * Fi; ct[i] += 1
            n1[i] = nr
        if trace and (it + 1) % trace == 0:
            dep = y[None] - np.repeat(X, 3, axis=1)
            d0 = y - np.repeat(X[0], 3)
            print(f"  k {it+1:>4}: all sd {dep.std():.3f}  "
                  f"mem0 sd {d0.std():.3f}  mean {dep.mean():+.3f}")
    dep = y[None] - np.repeat(X, 3, axis=1)
    d0 = y - np.repeat(X[0], 3)
    return float(dep.std()), float(d0.std())


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    a_off, m_off = run(40, 0.2, 0.05, 210, 10 ** 9, catapult=False)
    a_on, m_on = run(40, 0.2, 0.05, 210, 10 ** 9, catapult=True)
    check(f"catapult explodes member 0 ({m_on:.1f})", m_on > 10.0)
    check(f"catapult contaminates the ensemble ({a_on:.2f})",
          a_on > 3.0)
    check(f"fixed flow healthy at N=40 ({a_off:.2f})", a_off < 1.0)
    s21, _ = run(40, 0.2, 0.05, 21, 10 ** 9)
    check(f"transport law: T=21 starved ({s21:.2f}) vs T=210 "
          f"({a_off:.2f})", s21 > 1.5 * a_off)
    a4, m4 = run(4, 0.6, 0.05, 21, 7, catapult=True)
    check(f"N=4 survives the catapult ({a4:.2f})", a4 < 1.2)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=40)
    ap.add_argument("--bandwidth-sd", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--outer", type=int, default=210)
    ap.add_argument("--ctcheck", type=int, default=10 ** 9)
    ap.add_argument("--catapult", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trace", type=int, default=15)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    s, m = run(a.members, a.bandwidth_sd, a.eps, a.outer, a.ctcheck,
               catapult=a.catapult, seed=a.seed, trace=a.trace)
    print(f"final: ensemble sd {s:.3f}  mem0 sd {m:.3f}")


if __name__ == "__main__":
    main()
