# HANDOFF: Stage B in the sandbox (stage_b_cycle.py)

## What this is

Cycled DA with the estimation loop closed, in the Python testbed: three
ensemble providers on identical truth, observations, forecast rule, and
per-window Gaussian prior fit, differing only in the analysis step:

    exact  preconditioned MALA on the (Gaussian prior x pi-hat likelihood)
           posterior, ONE INDEPENDENT CHAIN PER MEMBER started at that
           member (a single thinned chain leaves the ensemble
           autocorrelated and starves both the analysis and the kernel at
           sparse observations -- found and fixed)
    pff    the interacting particle flow (Hu & van Leeuwen 2021), SVGD
           form, particle positions live in the kernel, score of pi-hat
    eda    per-member Gaussian 3D-Var with observations perturbed from the
           assumed R -- the operational competitor

Each provider accumulates its own prior-innovation archive across windows,
re-estimates the density with DOEE (lam 3 default), and feeds the estimate
back into its own analyses (eda stays Gaussian; its archive is estimated
from all the same). The evolution x' = a R(x) + sqrt(1-a^2) C^{1/2} eta is
stationary for any a < 1; a (--persistence) sets how much fresh Gaussian
noise launders posterior miscalibration each cycle.

FILTER, NOT SMOOTHER, on purpose: DOEE needs prior innovations, where the
predictor is independent of eps_t. A smoothed state has seen y_t, so
smoothed "innovations" are fitted residuals, correlated with the noise
they contain -- biased for deconvolution by construction.

## The argument (proved, then measured)

DOEE's kernel is unbiased iff truth is exchangeable with the members, iff
the members sample the true filtering prior. Then every window is a Stage
A instance and n accumulates across windows. EDA members sample a Gaussian
posterior calibrated to the ASSUMED R: kernel error first order in the R
error and the non-Gaussianity, irreducible with n. An exact-posterior
filter's members sample the posterior under the current pi-hat: kernel
error second order, vanishing at the fixed point pi-hat = pi. PFF inherits
the exact guarantee in the idealized limit and finite-K approximations
otherwise. The exchangeability ratio ER = var(H truth - H members) /
(2 mean member variance) tests the premise directly (1 iff calibrated).

## CORRECTION (post K-sweep)

Every PFF number above and below was measured on a SCALAR-RBF-kernel
SVGD -- my sandbox implementation choice, not the method. PFF.h and Hu &
van Leeuwen (2021) use a DIMENSION-WISE kernel (the CtrlInc_-valued
kernel applied by schur_product_with), which is the paper's remedy for
exactly the variance collapse measured here (sd ratio ~0.3 at K <= 100
in d = 40 is the known scalar-kernel regime K comparable to dimension).
The sandbox pff now defaults to the componentwise kernel
(--pff-kernel component) with AdaGrad stepping; the K sweep and the
cycling comparisons must be re-measured under it before any claim about
the operational PFF's calibration is made. The first sweep's large-K
rows (sd overshooting 1, mean degrading with K) were fixed-step solver
divergence, not method behavior, and are void.

## Measured results (heavy density throughout)

Benign configuration (assumed 0.4 ~ true sd 0.45, persistence 0.8,
10 x 200 obs, K 50): ER ~0.95-1.0 for ALL providers, final L1 0.08-0.15
for all -- the first-order EDA bias is below the estimation noise at
n = 2000, and fresh forecast noise launders what there is. The provider
matters for the ANALYSIS, not the archive, here: feedback improves mean
analysis rmse ~10% (exact 0.156 vs 0.176 without feedback; eda pinned at
0.182). Where the assumed R is roughly right, estimate from whatever
cycling you have.

Stressed configuration (assumed 0.2 = 2x overconfident, persistence 0.95,
40 x 200 obs, n = 8000): the theorem at finite n.

    exact  ER 0.99  analysis rmse 0.136  final L1 0.057  (0.106 at
           n=2000 -> 0.057 at n=8000; 1/sqrt(n) predicts 0.053)
    pff    ER 1.13  analysis rmse 0.136  final L1 0.136
    eda    ER 1.19  analysis rmse 0.186  final L1 0.133  (no better than
           its n=2000 value: the bias floor, measured at ~0.13)

EDA's archive stops converting n into estimate quality at its
miscalibration floor; exact rides the statistical rate through it (2.3x
better and still improving), and its analyses are 27% better. PFF's
analysis MEAN matches exact but its spread is under-dispersed (the known
finite-K SVGD variance underestimation): estimation floor at EDA's level
despite the unbiased score. At sparse observations (40/cycle) the PFF
under-dispersion is severe (ER excursions to 2.5).

## Consequences for Stage B on oops

1. PFF.h must not run at `inflation factor: 1.00` (the test configs'
   value): the sandbox shows finite-K flow under-disperses, and inflation
   is exactly the knob for it. Tune it with the ER diagnostic as target.
2. Zero the observation perturbations in the eda_3dvar_pff member configs
   (`obs perturbations amplitude: 0.2` reintroduces the assumed-R draw
   PFF was chosen to avoid; all particles should see the same y).
3. The other PFF.h caveats stand from the earlier read: kernel positions
   pinned at the background across outer loops, and a possibly
   double-counted prior after outer loop 1 (addGradientFG folds the
   per-member Jb gradient into rr while the flow also subtracts
   x_j - xbar) -- check against Hu & van Leeuwen (2021) before trusting
   multi-outer-loop flows.
4. A fourth sandbox provider worth ten lines when needed: per-member
   3D-Var with the non-Gaussian cost (the JEDI branch's own analysis) --
   the sharpest operational competitor, likely closing most of the
   analysis gap while keeping EDA-style calibration properties.

## Failed predictions, recorded so they are not retried

- "The provider ordering shows up at 10 cycles in the benign
  configuration": it does not; the bias is below the noise there and
  fresh forecast noise launders EDA's miscalibration. The ordering is a
  large-n, miscalibrated-R statement, demonstrated only in the stressed
  40-cycle run.
- "Overconfident R alone breaks EDA at persistence 0.8": it does not;
  36% fresh noise per cycle is enough laundering. Persistence is the
  second required ingredient.

## Reproduce

    python3 stage_b_cycle.py --selftest
    python3 stage_b_cycle.py --density heavy --cycles 10 --n-obs 200
    python3 stage_b_cycle.py --density heavy --cycles 40 --n-obs 200 \
        --assumed-error 0.2 --persistence 0.95

The selftest is the linear-Gaussian null where EDA is exact: all three
providers must land on the same Kalman posterior (flow to 1e-4, MALA
within MC error, sd ratio ~1).

## Queue

1. Confirm the sparse-obs rerun (10 x 40, persistence 0.95) with the
   per-member-chain exact provider: exact should now beat eda on analysis
   rmse and its density L1 should unstick from ~0.31.
2. The scaling figure: L1 vs accumulated n per provider (the plateau vs
   the 1/sqrt(n) line is the paper's Stage B panel).
3. Generality: laplace and mirrored_gamma through the stressed
   configuration.
4. The oops build session: cycling driver with the config fixes above
   (queue item 4 of HANDOFF_STAGE_A).
