# HANDOFF: the MAP reference harness (map_reference.py)

## What this is

A single-window 3D analysis testbed that scores observation-error density
error by its effect on the analysis mode, with no sampling and no minimizer
noise: J(x) = 0.5 (x-x_b)' C^-1 (x-x_b) - sum log pi(y_i - (Hx)_i) on the
periodic 40-point grid, solved to machine precision per arm. The prior is
exactly specified by construction (background error drawn from C), so the
observation-error density is the only mis-specification anywhere. Truth is
injected analytically (mirrored_gamma with the fixed population scale, so
the true arm is exact for all four kinds).

Arms: gauss-assumed (status quo 0.4), gauss-matched (population sigma,
perfect variance-only estimation), gauss-var (the DOEE estimate's own sd),
gauss-best (per-window oracle-tuned sigma, an upper bound on every
Gaussian), estimated (DOEE through the Format A export, gaussian_tails
included), true (analytic, defines the reference MAP), true-fmt (the true
density through Format A). Gaussian arms are closed form; the rest are
curvature-clipped Newton with Armijo, multistart, convergence by step
collapse (Format A scores are piecewise constant, so their MAP sits where
the summed score jumps through zero).

Metrics per replicate, against the true-density MAP: state rmse and
true-cost regret (J_true(x*_arm) - J_true(x*_true))/n_obs in nats/ob,
nonnegative by construction (a negative value means the true-arm multistart
missed a basin and warns).

## Certified error budget (heavy, n_obs 400, members 50, 8 replicates)

1. Format A representation: true-fmt sits 0.0020 rmse / 0.00002 nats/ob
   from the analytic true MAP. The export is effectively lossless.
2. Evolving-Gaussian machinery: --jedi-semantics (Density.variance
   verbatim, mode window / floor / fallback branches) changes regrets by
   <= 7e-4 nats/ob. The Eq. 9 fixed point is the MAP to sub-milli-nat
   precision. The only unmodeled distance to real JEDI is finite outer
   loops.
3. Everything else is density estimation.

## The headline (heavy, lam 3, n_obs 400, members 50, 8 replicates)

    true-cost regret, nats/ob        mean      CI
    gauss-assumed                    0.05784   [0.04716, 0.06925]
    gauss-matched                    0.05693   [0.04639, 0.06812]
    gauss-var                        0.05688   [0.04615, 0.06825]
    gauss-best (oracle per window)   0.05597   [0.04533, 0.06736]
    estimated (DOEE, lam 3)          0.02658   [0.02101, 0.03361]
    true-fmt                         0.00002   [0.00001, 0.00002]

The whole Gaussian family sits in a ~4% band: clairvoyant variance tuning
recovers ~3% of the shape gap; the estimate recovers 53% and beats the
oracle-tuned Gaussian in 8/8 replicates. Paired margin (oracle Gaussian -
estimated): +0.02939 [+0.01589, +0.04330]. This is the JCSDA/NOAA pitch in
one table: for a heavy-tailed error the budget is shape, and no amount of
variance tuning reaches it.

Reproduce:

    python3 map_reference.py --selftest
    python3 map_reference.py --density heavy --replicates 8 --lam 3
    python3 map_reference.py --density heavy --replicates 8 --lam-sweep 1,3,10,30

## Mechanism findings

- Analysis regret tracks the SHAPE of the weight profile log sigma_o(d),
  not its level: at fixed rms weight error, the bias component (uniform
  rescaling) is nearly harmless and the scatter component (relative
  mis-ranking of observations) is expensive. Measured at lam 30: scatter
  correlates +0.98 with regret within the batch; a replicate with L1 0.81
  but nearly pure bias beat a replicate with L1 0.32 and high scatter by
  10x.
- L1 to the density repeatedly dissociates from analysis impact (lam 3 has
  worse L1 than lam 10 and better regret; the lam 30 batch had L1
  anticorrelated with regret at -0.49).
- The correlations require heterogeneity: within the homogeneous lam-3
  batch they vanish (restriction of range), and across the 1..30 sweep the
  scatter correlation is +0.43. Scatter is a useful predictor where
  estimates differ, not a law; the operative metric is regret itself.

## Failed predictions, recorded so they are not retried

- "The regret minimum sits at heavy smoothing (lam 30-300) because bias is
  the cheap side": WRONG at this configuration. Measured regret(lam) over
  1/3/10/30/100/300: 0.035, 0.027, 0.033, 0.097, 0.179, 0.153. The minimum
  is at lam ~3, and lam >= 100 is worse than the plain Gaussian. Heavy
  over-smoothing flattens the core, which is itself relative mis-weighting.
- The lam-30 record from the real-mode d800 ensemble is not contradicted
  (different n, ensemble, innovation noise), but the lam optimum is now
  demonstrably configuration-dependent, which strengthens the case that
  selection must target an analysis-impact criterion.

## Queue

1. Null control: `--density gaussian --replicates 8 --lam 3` ladder run.
   The insurance premium of shape estimation when the truth IS Gaussian is
   the first number a reviewer asks for after the win. (Selftest null shows
   est rmse ~0.049 at the estimation floor; the regret with a CI belongs in
   the pitch table.)
2. Generality: the same ladder + sweep for laplace and mirrored_gamma.
3. Scaling: the 53% recovery fraction and the estimation floor vs n_obs and
   members (the Stage B cycling argument: windows manufacture n).
4. `--adaptive` through the harness: does the one-SE mu land near the
   regret minimum (lam ~3 here) or the L1 minimum (lam ~10)?
5. Replicates 16-32 on the headline configuration to tighten the CIs for
   the writeup.
6. Selection research (Stage A queue item 2), reframed: a truth-hidden
   criterion that targets the analysis-impact optimum; relative weights may
   be estimable where the absolute scale is not.

## Relationship to the rest of Stage A / Stage B

This harness replaces the earlier MCMC-reference plan (dropped: the MAP is
exact, deterministic, and apples-to-apples with VarDA). PFF remains the
Stage B in-the-loop filter; the three PFF.h caveats (obs perturbations from
the assumed R in the eda_3dvar_pff configs, kernel positions pinned at the
background, a possible double-counted prior after outer loop 1) are parked
until Stage B and none block MAP work.
