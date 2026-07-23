# NOTES: asymptotics of the DA / noise-learning loop

The question: after many cycles of DA plus density estimation, does the
estimate converge to the true observation-error density pi -- and does it
matter which ensemble provider sits in the loop? Observation errors eps_t
are assumed independent across time and of the state.

## 1. The convolution identity (every provider)

Prior innovation for member k at cycle t:

    d_t^k = y_t - H x_t^k = eps_t + H(x_t - x_t^k) =: eps_t + b_t^k.

Prior members are functions of PAST data only and eps_t is independent of
the past, so eps_t is independent of b_t^k for every provider, EDA
included. Hence law(d) = pi * rho_b exactly (pooled over cycles and
locations). This is what the independence assumption buys, and it is why
the FILTER is mandatory in the innovation view: a smoothed state has seen
y_t, independence breaks, and smoothed residuals are not a convolution of
pi with anything (see section 8 for the corrected smoothing view).

## 2. What DOEE identifies

The estimator inverts the convolution with a kernel built from
same-location member differences, law rho_D = law(H(x^j - x^k)).
Asymptotically (n -> infinity; kernel characteristic function without
zeros; location identified up to the ensemble-mean bias) pi-hat solves
pi-hat * rho_D = pi * rho_b.

PROPOSITION (consistency). pi-hat -> pi iff rho_b = rho_D, which holds
iff the truth is exchangeable with the members, which holds iff,
conditional on the members' information set, the members are iid draws
from the TRUE conditional of the state given that information.

## 3. Where EDA-in-the-loop IS asymptotically correct

(a) Linear-Gaussian with the CORRECT R: perturbed-observation EDA
members are exact posterior samples (the classical randomized-MAP
result); the loop is trivially consistent -- and pi is Gaussian, so
there is nothing to learn.

(b) The dense-observation limit sigma_b / sigma_o -> 0: as the posterior
contracts, BOTH rho_b and rho_D collapse to a point mass. EDA's
miscalibration is a constant RATIO (its ensemble expresses
(B^-1 + H'R_A^-1 H)^-1 while its actual error carries the sandwich
covariance), but a constant ratio of a vanishing quantity vanishes: the
kernel error goes to zero in absolute terms and pi-hat -> pi for ANY
provider. This is the regime the benign sandbox configuration
approached, and why no provider ordering appeared there.

## 4. Where it is NOT: many cycles at fixed observation density

EDA's stationary ensemble law is calibrated to (R_A, B), not to reality;
rho_b differs from rho_D by a constant offset that accumulated n never
touches. The estimate converges -- to the wrong limit, at a distance set
by the kernel bias. Measured: the 40-cycle stressed run's EDA plateau at
L1 ~ 0.13 while the exact provider rode 1/sqrt(n) through it
(0.29 -> 0.065 over n = 400 -> 8000).

## 5. The loop as a fixed-point iteration

Define T: pi-hat -> (cycle the filter under likelihood pi-hat) -> (DOEE
on the archive).

Fixed point: T(pi) = pi for the exact-posterior provider -- at the true
density, Bayes calibrates the ensemble to reality, exchangeability is
exact, and DOEE is consistent. EDA's map has no fixed point at pi:
T_EDA(pi) != pi; its fixed point sits at the bias floor of section 4.

Order of the error, CORRECTED: away from the fixed point, members follow
p-hat while truth follows p, so law(b) = p (-) p-hat against
law(Delta) = p-hat (-) p-hat -- a mismatch FIRST order in (p - p-hat).
(An earlier handoff called this second order; that was too generous.)
What makes the iteration converge is DAMPING, not order: the contraction
factor is roughly

    c ~ (posterior sensitivity to the likelihood) x a^2
        x (DOEE sensitivity to kernel error, ~ sigma_b / sigma_o),

with a the persistence (fresh model noise launders a (1 - a^2) share of
any miscalibration per cycle). Local convergence requires c < 1; no
global theorem is claimed. Both sides measured: contraction in the
stressed feedback run (L1 at the statistical rate for 40 cycles), and
the basin's edge in the sparse-observation run, where a bad early pi-hat
fed back into the ensembles and the honest Gaussian outperformed the
"exactly wrong" pi-hat posterior. The feedback archive gate
(--feedback-min-n) is therefore a convergence condition, not a
heuristic: start the iteration inside the basin.

## 6. Summary of the convergence conditions

pi-hat -> pi after many cycles iff:

1. observation errors independent across time and of the state (makes
   prior innovations a convolution; forces filtering in the innovation
   view);
2. the provider's analysis step is calibrated at the fixed point (exact
   posterior qualifies; the componentwise particle flow approximately,
   with ~1.05 inflation covering its measured residual; EDA does not,
   outside regimes 3a-3b);
3. deconvolution identifiability (kernel characteristic function without
   zeros; location up to ensemble-mean bias);
4. initialization inside the contraction basin (in practice: do not
   adopt the estimate into the analysis until the archive supports a
   decent one).

## 7. Deterministic dynamics: sampling versus density

Lorenz-96 proper is a deterministic ODE; the sandbox's additive Gaussian
increment is a construction, and operational systems inject stochastic
physics (SPPT/SKEB) for the same reason the sandbox does. The
deterministic case splits cleanly:

- The ESTIMATION side never evaluates a prior density. Consistency needs
  members that are SAMPLES of the true filtering prior, and deterministic
  dynamics preserve exact sampling for free (push exact posterior
  samples through M). Sections 1-4 survive deterministic dynamics
  untouched.
- The ANALYSIS side (MALA, the flow) needs log-prior and score, which no
  provider has exactly under deterministic nonlinear M; every provider
  approximates (the shared Gaussian fit). Additive model noise with a
  KNOWN density q is the one construction that restores an analytic
  prior: given the previous posterior particles, the filtering prior is
  the exact mixture (1/K) sum_j q(x - M(x_j^a)) -- density and score in
  closed form, q not necessarily Gaussian. As Q -> 0 the fitted-prior
  approximation binds and the "exact" provider's exactness claim erodes:
  a measurable robustness axis, not a philosophical one.

## 8. The ideal samples: the leave-one-out principle

What should the members be, ideally, for DOEE? Two conditions on the
samples used to score observation y_{t,i}:

  (i)  INDEPENDENCE: the members may condition on anything EXCEPT
       y_{t,i} itself (keeps eps_{t,i} independent of the members, so
       the innovation is a genuine convolution);
  (ii) CALIBRATION: the members must be iid draws from the TRUE
       conditional of the state given whatever they condition on
       (exchangeability, correct kernel).

Condition (i) constrains the conditioning set; condition (ii) says be
Bayesian about it. More conditioning means a tighter conditional, a
narrower kernel, less deconvolution burden -- so the optimum conditions
maximally subject to (i):

    THE IDEAL SAMPLES ARE DRAWS FROM THE LEAVE-ONE-OUT SMOOTHING
    POSTERIOR  p(x | all observations except y_{t,i}).

The hierarchy, ordered by kernel width:

- Full smoothing posterior (conditions on y_{t,i} too): INVALID --
  fitted residuals, shrunk toward the noise. In the linear-Gaussian case
  the influence identity r_loo = r/(1 - A_ii) shows the
  degrees-of-freedom correction turns full-posterior residuals into the
  leave-one-out ones; "residuals up to O(DFS/n)" is the corrected
  version of an invalid construction.
- Leave-one-out smoothing posterior: valid, narrowest possible kernel.
  Nearly free implementation: importance-reweight the full smoothing
  ensemble, w_k proportional to 1/pi-hat(r_{t,i}^k), per observation
  (single-observation removal keeps the weights mild; leave-slice-out is
  the sturdier fallback if they degenerate).
- Filtering prior (conditions on past only): valid, wider than
  necessary; the ideal's causal approximation. Cycling keeps its kernel
  bounded by re-anchoring each window.
- Climatological prior (Stage A): valid, widest kernel; zero
  circularity; right first test, wrong final answer.

Calibration is not bought at any rung: the LOO posterior is computable
only under pi-hat, so the fixed-point structure of section 5 applies
unchanged. LOO minimizes the kernel GIVEN calibration.

CAVEAT (measured on mirrored_gamma): the importance weights 1/pi(r) are
UNBOUNDED whenever the density can vanish at an observed residual --
support walls are the worst case -- and then a few wall-violating
members absorb all the weight (ESS collapsed to ~3/400 and the resampled
kernel degenerated). The finite-variance condition is E[1/pi(r)] finite
under the full posterior; this is exactly the known IS-LOO failure mode
of the Bayesian cross-validation literature. Implemented remedy:
Ionides-style weight truncation at mean(w) sqrt(Kref) (slightly biased,
stable); principled upgrades are Pareto-smoothed importance sampling,
leave-slice-out, or exact per-observation refits for flagged
observations. Steep-tailed and compact-support noise densities should be
run with the truncated weights and the reported ESS watched.

Smoothing-view consequence: in a strong-constraint 4D window the
trajectory has d degrees of freedom against n observations, so corrected
residuals differ from the true errors by O(DFS/n) and the estimation
problem sheds its dependence on ensemble calibration as DFS/n -> 0 --
the provider ordering of section 4 is a property of the filtering regime
(sigma_b/sigma_o order one), not of estimation as such. The price is the
perfect-model assumption (model error lands in the residual pool) and
the Lyapunov cap on window length. stage_c_smoothing.py measures the
ladder at matched n.
