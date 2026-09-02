# Stage A handoff: state, results, and open problems

Written 2026-07-21, at the close of the session that produced the first
realizable win. Everything below is tested and pushed; a fresh session can
start from this file plus PROCESS_NONGAUSSIAN_JO.md (kept outside git at
the rrfs-ufo-update root on the Mac).

VERIFIED 2026-07-21 on the VM by a cold top-to-bottom pass: both tips,
the full python battery, ctest 12/12, and the record reproduction hit
every stated number exactly, including regret 0.0462 -- the determinism
claim is itself tested.

## Where things run

- Mac work root: `~/Downloads/scas-paper/rrfs-ufo-update/` with `stable_DOEE/`
  (branch `jedi-density-export`; last code-bearing commit 34efa8d, this
  handoff and later notes sit on top) and `oops-fork/`
  (branch `nongaussian-costjo`, tip 79388e44). Both pushed.
- VM (OrbStack): source `~/jedi/src/oops` and `~/jedi/src/stable_DOEE`,
  build `~/jedi/src/build/oops`, venv `~/jedi/venv`.
- l95 static test data is COPIED from `l95/test/testdata/` into the build's
  `Data/` at cmake configure time; if `Data/` is cleaned, restore with
  `cp ~/jedi/src/oops/l95/test/testdata/* ~/jedi/src/build/oops/l95/test/Data/`.
- The testbed ensemble is genenspert-seeded and deterministic: identical
  configurations reproduce spread 0.532, reliability 0.99, bias -0.059
  exactly, so all comparisons on one configuration are paired.

## The record result (heavy mixture, scale 2, pert-sd 0.20, 100 members,
## 1200 obs, reliability 0.99)

Regret in nats/ob under the true-density log score at the noise-free
observable (rmse in parentheses), all runs paired:

    control, mistuned R (sigma 0.4)            0.0526  (0.166)
    control, tuned R (sigma 0.9)               0.0494  (0.161)
    treatment, CV-default estimate             1.5484  (1.241)
    treatment, lam 30, neighbor-copy project   0.0556  (0.171)
    treatment, lam 30, interpolation project   0.0462  (0.156)
    treatment, oracle (true density)           0.0290  (0.123)

Fair-fight bootstrap (treatment lam 30 vs tuned control): -0.0032 nats/ob,
naive CI [-0.0072, +0.0009], 40-block CI [-0.0149, +0.0079]. Calibrated
claim: statistical tie with a perfectly tuned control, point estimate
favoring the treatment; decisive win over a mistuned control; the
treatment reads no R at all (the density supplies everything), which is a
real operational argument. One cycle resolves about +-0.011 nats (block),
so margin growth beats replication (~25 seeds would be needed at the
current margin).

Reproduce the record treatment run (VM):

    cd ~/jedi/src/stable_DOEE/l95-testbed
    python3 stage_a_end_to_end.py --build ~/jedi/src/build/oops \
        --reuse-ensemble --lam 30 \
        --pert-sd 0.20 --members 100 --obs-density 400 \
        --scale 2 --assumed-error 0.9 --floor-trials 12

Expected: DOEE L1 0.154, sigma 0.883, kurt +7.17, sigma at mode 0.449,
kept 99.9%, projected-bin mass 2.15%, treatment regret 0.0462.

## What the session established (mechanisms, all measured)

1. The method is sound in this regime: the oracle treatment beats the
   tuned control by 41% with EvolvingSigma spanning the mixture components
   (0.44-2.01) and converging across outer loops. The whole adverse result
   was estimation error.
2. More members HARMED the estimate: bins grow as sqrt(n*K), the
   third-difference penalty's continuum strength scales like lam*dx^5, so
   K=25 -> 100 silently collapsed the smoothing 32x. Quality tracks bin
   count, not K. Independent information is the OBSERVATION count n (each
   obs contributes one error draw; members refine duplicates).
3. Two export fixes, both in doee_to_yaml/driver: a two-bin mode band is
   exempt from the projection-mass gate (flat-top slope chatter on the
   densest bins), and slope-run lobes are projected by interpolation
   between flanking valid slopes (neighbor copying left near-zero shelves
   where (d-m)/g exploded to sigma 10.8, silencing exactly the
   moderate-tail obs the method exists for).
4. Remaining estimation budget (0.017 nats to the oracle): lobe shoulders
   still drive EvolvingSigma max ~10 (legitimately tiny correct-sign
   slopes flanking the lobes -- estimate error, not projection error),
   center sigma 0.449 vs 0.500, L1 0.154.

## Open problem: principled lambda selection (three designs failed,
## instructively)

`--lam` remains an explicit knob; selection still uses the calibrated
default. Three automatic criteria were built and measured, each with a
precise failure mode -- do not retry them as-is:

- Cross-resolution stability (rebin the histogram, compare solutions):
  the coarse histogram is a deterministic function of the SAME noise
  realization, so both resolutions fit the same noise coherently; it
  certified a collapsed estimate as stable.
- Split-half solution agreement: gaps fall monotonically in lambda
  (over-smoothed halves trivially agree), so it anchors at maximal
  smoothing; it Gaussianized a +7.2-kurtosis truth to +0.23.
- Discrepancy principle with measured noise (residual vs split-measured
  delta): innovation-space residuals are nearly flat in lambda -- the
  same ill-posedness that flattens CV -- so the crossing is a loose upper
  bound, and delta wobbles +-25% across split seeds, flipping the choice
  by a decade. NOTE the residual is L1(dx*(Eta@pi) - f_d); forgetting the
  dx gives residuals ~19 and total nonsense.

Distilled: innovation-space criteria cannot SEE the density being
destroyed; density-space criteria cannot distinguish agreeing-on-signal
from agreeing-on-smoothness. Proxy cases misled twice; the next design
must be measured against the record member files on the VM first.

## The obs-density regime finding (2026-07-21, later)

The big-observations run (obs-density 1600, members 50, n=4800,
reliability 0.97) reframed the margin question. All runs paired on the
same fresh ensemble; regret in nats/ob:

    regime               control   oracle    est lam30   est lam100
    sparse (400/var30)   0.0494    0.0290    0.0462      --
    dense (1600/var120)  0.0187    0.0124    0.0221      0.0225

Dense observations Gaussianize the effective problem: the control's own
regret fell 2.6x and the ABSOLUTE oracle margin compressed 3x (0.0204 ->
0.0063) even though the RELATIVE margin barely moved (41% -> 34%).
Meanwhile the estimate genuinely improved with 4x the draws -- at lam
100 (compensating the ~490-bin dilution of the penalty) the recovery is
the cleanest ever: L1 0.135, sigma +0.9%, sigma at mode 0.493 vs 0.500,
zero lobes, zero projections, EvolvingSigma 0.37-2.49 with no shelves --
and it STILL trails the control by ~0.004. So realizable estimation
error has a floor (~0.010 nats here, dominated by fine sigma(d) profile
errors: the near-mode dip to 0.37 vs the true 0.5, and mild kurtosis
undershoot), and when the available margin compresses below that floor,
the sign flips regardless of density quality. Conclusion: the method's
realizable value concentrates in the obs-sparse regime, which is where
the win stands and where much of real DA lives. The center-sharpening
problem from the record ensemble is SOLVED by more observations (sigma
at mode 0.514 at lam 30, 0.493 at lam 100); the lam-dilution rule of
thumb held (30 at 347 bins ~ 100+ at 490 bins).

## The adaptive estimator (2026-07-21, latest; EXPERIMENTAL)

Answers the lambda-selection problem from theory instead of tuning, and
supersedes the lambda-selection research line. Code: an additive section
in `stable_doee_reg.py` (`measure_bin_noise`, `_solve_whitened`,
`adaptive_bin_count`, `estimate_adaptive`,
`estimate_adaptive_from_ensemble`); battery:
`l95-testbed/test_adaptive.py`, 7/7 at this tip. Not wired to the
driver yet, deliberately: prototype grade.

Design, derived (full derivation in the module's section comment):
per-bin histogram noise is MEASURED from grouped half-splits (unbiased
including the shared-epsilon correlations no clean model captures), with
an all-samples-independent Poisson floor for sparse bins; the data term
is whitened by it; the penalty carries dx^-5 so the smoothing strength
mu means the same thing at every bin count (the unnormalized form is
why more members silently collapsed the smoothing 32x); mu is chosen by
the discrepancy principle against the parameter-free target
chi2 = n_bins, exact by construction of the empirical variances. That
is the adaptivity the problem demands: no smoothness assumed, more data
shrinks the noise, shrinks the feasible mu, and lets the estimate be as
rough as the data supports. Infeasibility (min chi2 > n_bins) is the
over-dispersion signature, reported as cache['chi2_ratio'].

Measured scoreboard, identical data, legacy at its best lambda:
gaussian 2000x20 adaptive WINS (L1 0.125 vs 0.182; the battery's null
is L1 0.067, the best in the project); record-class heavy 1200x100
adaptive MATCHES hand-tuned lam 30 (0.070 vs 0.077) with zero tuning;
over-dispersed x1.24 adaptive RECOVERS (L1 0.18, sd 0.855) where every
legacy configuration fragments; irregular bimodal truth: both modes at
every n, sd 1.37-1.40 vs 1.386, mu falls with n. KNOWN GAP: heavy
2000x20 is ~2x worse than legacy (0.27-0.30 vs 0.08-0.16 over seeds
11/12/13, mu pinning at 2e-2; suspects: the coarse 10^0.75 mu-grid
steps and the Poisson-floor interplay at moderate tails). KNOWN
WRINKLE: resolution invariance nb vs 2nb gave L1 0.186 between the two
solutions against a <0.06 target. Both are pinned in the battery so
improvement is visible and regression fails.

Debugging lessons paid for: (1) the group-limited noise rate
f/(n_groups dx) as a conservative envelope over-states fine-bin noise
20-50x and smoothed every recovery flat -- the correct sparse-bin
repair is the pure Poisson floor f/(N dx) (documented in the code);
(2) `_make_cache` trims to the contiguous main run by design (unimodal
export), which silently cost a bimodal recovery its second mode and 70%
of its variance -- `estimate_adaptive` therefore returns the FULL
grid/pi and keeps the trimmed cache for export only.

## Queue, in order

1. Obs-density sweep, the regime map and likely the paper figure:
   densities 200/400/800/1600 x {oracle, lam 30} at members 50 (the 1600
   pair is done; 400 needs a members-50 rerun for homogeneity). The
   margin-vs-density curve from the oracle column is estimate-free; the
   realizable column shows where the estimation floor crosses it.
2. Adaptive estimator, close the two open items above (heavy-2000x20
   gap, resolution invariance), then a driver `--adaptive` flag and VM
   acceptance on the record and dense member files -- expect lam-30- and
   lam-100-class quality automatically. Then re-pin floors and records
   under mu. (Supersedes the lambda-selection research line.)
3. oops fork, next build session: the C++ consumer of
   `jedi_export/fixtures.json` (8 cases, C++ key names; the assimilation
   unit tests take `test/testinput/empty.yaml` via TestEnvironment, so a
   config-path route exists). Python side (`check_fixtures.py`) passes 8/8.
4. Stage B: cycling driver, the PFF switch (the joValue fix makes PFF
   free), Orion/Intel, RRFS.

## Test battery (all green at this tip)

    cd stable_DOEE/jedi_export && python3 selftest.py && python3 check_fixtures.py
    cd ../l95-testbed && python3 test_regressions.py
    python3 test_adaptive.py                             # adaptive estimator, 7/7
    python3 stage_a_end_to_end.py --synthetic            # + --density mirrored_gamma, laplace
    python3 stage_a_end_to_end.py --synthetic --n-obs 1200 --members 100 \
        --scale 2 --assumed-error 0.9 --sigma-b 0.55 --lam 30
    python3 stage_a_end_to_end.py --synthetic --oracle-density --scale 2 --assumed-error 0.9

oops fork: `ctest -R "nongaussiandensity|evolvinggaussian|nongaussian|evolvingsigma|gaussbias"`
must give 12/12.
