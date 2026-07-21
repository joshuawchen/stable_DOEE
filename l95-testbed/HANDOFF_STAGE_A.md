# Stage A handoff: state, results, and open problems

Written 2026-07-21, at the close of the session that produced the first
realizable win. Everything below is tested and pushed; a fresh session can
start from this file plus PROCESS_NONGAUSSIAN_JO.md (kept outside git at
the rrfs-ufo-update root on the Mac).

## Where things run

- Mac work root: `~/Downloads/scas-paper/rrfs-ufo-update/` with `stable_DOEE/`
  (branch `jedi-density-export`, tip 34efa8d) and `oops-fork/`
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

## Queue, in order

1. Big-observations run (margin growth; also shrinks the noise that makes
   selection hard): `--obs-density 1600 --members 50 --lam 30`, fresh
   ensemble. 4x the independent draws, attacks all residual error terms,
   halves the per-cycle CI.
2. Lambda-selection research, starting with a measurement script on the
   VM producing criterion curves on the real member files.
3. oops fork, next build session: the C++ consumer of
   `jedi_export/fixtures.json` (8 cases, C++ key names; the assimilation
   unit tests take `test/testinput/empty.yaml` via TestEnvironment, so a
   config-path route exists). Python side (`check_fixtures.py`) passes 8/8.
4. Stage B: cycling driver, the PFF switch (the joValue fix makes PFF
   free), Orion/Intel, RRFS.

## Test battery (all green at this tip)

    cd stable_DOEE/jedi_export && python3 selftest.py && python3 check_fixtures.py
    cd ../l95-testbed && python3 test_regressions.py
    python3 stage_a_end_to_end.py --synthetic            # + --density mirrored_gamma, laplace
    python3 stage_a_end_to_end.py --synthetic --n-obs 1200 --members 100 \
        --scale 2 --assumed-error 0.9 --sigma-b 0.55 --lam 30
    python3 stage_a_end_to_end.py --synthetic --oracle-density --scale 2 --assumed-error 0.9

oops fork: `ctest -R "nongaussiandensity|evolvinggaussian|nongaussian|evolvingsigma|gaussbias"`
must give 12/12.
