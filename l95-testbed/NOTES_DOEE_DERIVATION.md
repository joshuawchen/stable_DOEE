# NOTES: the DOEE master derivation

The precise mathematics of when and why DOEE recovers the observation
error density, as a function of the sampling law the members are drawn
from. Companion to NOTES_CONVERGENCE.md (which treats the cycled loop);
this note treats a single identifiable record. Setting: perfect known
model, truth x_t = M_t(x_0*), one observation per step
y_t = h_t(x_0*) + eps_t with h_t = H o M_t, eps_t iid ~ pi, independent
of everything, pi time-invariant, t = 1..n. Members x_0^{t,k} ~ q_t iid
(K per step), q_t conditioned on an information set I_t. Innovation
d_t^k = eps_t + beta_t^k with beta_t^k = h_t(x_0*) - h_t(x_0^{t,k});
member difference Delta_t^{jk} = h_t(x_0^j) - h_t(x_0^k). Write
p_t = law(x_0* | I_t) (the true conditional), P_t = h_t # p_t,
Q_t = h_t # q_t (obs-space pushforwards), phi_X = characteristic
function.

## 1. The two hypotheses

(H1) INDEPENDENCE: y_t not in I_t and q_t is sigma(I_t)-measurable.
Then eps_t is independent of (x_0*, members) given I_t and

    law(d_t) = pi * law(beta_t)        (a true convolution).

(H2) SAMPLING STRUCTURE: given I_t, truth ~ p_t and members ~ q_t, all
independent. Then

    phi_beta = phi_P . conj(phi_Q),    phi_Delta = |phi_Q|^2.

Classical exchangeability is the special case q_t = p_t.

## 2. The estimator

DOEE (Hu, van Leeuwen & Geer 2024): histogram the pooled innovations
(f_d) and the pooled same-location member differences (f_kappa) on a
grid; the grid convolution is linear, f_d ~ A p with A Toeplitz in
f_kappa; estimate

    p-hat = argmin_{p >= 0} ||A p - f_d||^2 + lambda ||D2 p||^2,

a positivity-constrained penalized quadratic program (NNLS structure),
normalized to unit mass. Pooling must group observations sharing the
same pi (innovation_groups); the kernel mixture then matches the
innovation mixture location by location. Characteristic functions below
are analysis devices only; the algorithm never performs Fourier
division.

## 3. The master formula

Population limit (n -> infinity, lambda -> 0): p-hat solves
pi-hat * law(Delta) = pi * law(beta), i.e.

    phi_pihat . |phi_Q|^2 = phi_pi . phi_P . conj(phi_Q)
    =>  PHI:   phi_pihat = phi_pi . phi_P / phi_Q,

and in cumulants, every order j >= 1:

    (*)   kappa_j(pi-hat) = kappa_j(pi) + kappa_j(P_t) - kappa_j(Q_t).

Consequences:
- j = 2: sigma-hat^2 - sigma^2 = Var(P) - Var(Q): the sigma_b^2 leakage
  identity in general form.
- j = 3: any SYMMETRIC sampling law (EDA's Gaussian, symmetric
  perturbations) has kappa_3(Q) = 0, so the entire skewness of the true
  conditional lands in pi-hat. Member-difference kernels are symmetric
  always; asymmetric background error is invisible to them.
- Under-dispersion by factor gamma (Q ~ P scaled): kappa_2 mismatch
  (1 - gamma^2) Var(P), zeroed by anomaly inflation 1/gamma. Measured
  componentwise-PFF gamma ~ 0.95 => inflation ~ 1.05.
- EDA at fixed observation density: Q Gaussian with the asserted
  covariance, P the non-Gaussian true conditional with the sandwich
  second moment; the (*) mismatch is O(R error) + O(non-Gaussianity),
  CONSTANT in n: the measured bias floor.
- The estimator is exact iff Q_t = P_t IN OBS SPACE: full state-space
  calibration is sufficient, not necessary.

## 4. The rate: why a high-entropy prior fails with zero bias

Take q_t = p_t (zero bias by (*)). The inversion divides by |phi_Q|^2;
with m = nK effective samples and pi of smoothness b, classical
deconvolution minimax gives

    ordinary-smooth kernel  |phi_Q| ~ |w|^{-a}:
        error ~ m^{-b/(2b + 2a + 1)}          (polynomial, degraded)
    supersmooth kernel (Gaussian-like, width s = O(1)):
        error ~ (log m)^{-b/2}                (LOGARITHMIC).

A wide climatological prior is therefore consistent and unusably slow:
the entropy of Q sets the rate; (*) sets the bias; they are separate
objects.

## 5. The contraction regime and the optimum

Let I_t = { y_s : s != t } and assume identifiability:
lambda_min(J_n) -> infinity for the Fisher-type information
J_n = sum_t E[psi'(eps)] grad h_t grad h_t'. Posterior contraction /
Bernstein-von Mises gives P_t ~ N(mu_t, s_t^2) with s_t^2 = O(1/n) and
mu_t - h_t(x_0*) = O_p(n^{-1/2}). Then:

- BIAS channel: any q_t with obs-space mean mu_t + o(n^{-1/2}) and
  variance O(1/n) contributes O(1/n) ||pi''|| to (*): second order,
  vanishing.
- VARIANCE channel: |phi_Q|^2 -> 1, amplification -> 1, and the
  estimator attains the ORACLE rate of direct density estimation from
  iid draws of eps.

THEOREM (informal). Among sampling laws satisfying H1, the error is
[amplification(width of Q)] x (statistical rate) + |(*) mismatch|.
The unique law zeroing the mismatch identically at every n while
minimizing amplification among zero-bias laws is the LEAVE-ONE-OUT
posterior q_t = p(x_0 | y_{-t}), where y_{-t} = all observations EXCEPT
y_t. (P_t is itself the tightest true conditional excluding y_t, and
(*) = 0 forces Q_t = P_t.)

Worked n = 3, to fix the notation: to score y_2, draw members from
p(x_0 | y_1, y_3); to score y_1, from p(x_0 | y_2, y_3); to score y_3,
from p(x_0 | y_1, y_2). n barely-different posteriors, each conditioning
on n - 1 observations -- almost everything -- omitting exactly the one
observation it will judge. Linear-model specialization: LOO residuals
r_t / (1 - A_tt); "never score an observation against a state estimate
that ingested it" is the Desroziers instinct completed at density level.

FIRST-ORDER-EQUIVALENT CHEAP TWIN: a point mass at any sqrt(n)-
consistent leave-one-out estimator. Kernel = delta (no deconvolution,
oracle amplification); (*) bias = s_t^2 + shift smoothing = O(1/n),
the same order as the calibrated optimum's finite-K kernel noise. The
calibrated LOO posterior removes the O(1/n) automatically through the
member-difference kernel; the point mass removes it analytically through
the influence/DFS correction. A point mass at the FULL-posterior mean
violates H1 and inherits the (I - A) shrinkage: the invalid "residual"
arm, derived. Absolute continuity with the prior is a red herring: PHI
shows the criterion is phi_P/phi_Q -> 1 on pi's frequency band --
low-cumulant matching of h#q to h#p -- which a point mass satisfies
asymptotically and a wide absolutely-continuous prior fails in rate.

## 6. Taxonomy (instances of (*) + section 4)

  q                          (*) bias                    kernel/rate
  climatological prior       0                           O(1), log rate
  filtering prior (cycled)   0 at pi-hat = pi;           O(sigma_b),
                             O(pi-hat - pi) damped        polynomial
  EDA                        O(R err)+O(nonGauss),        moderate;
                             const in n; kappa_3 leaked   bias floor
  PFF componentwise,         (1-gamma^2)Var(P) at fixed   ~ posterior
    converged                point; inflation 1/gamma
  full-posterior samples     H1 FAILS (not a convolution; (I - A) width
                             fitted residuals)            bias)
  LOO posterior              0 identically                minimal: OPTIMAL
  point mass at sqrt(n)-     O(1/n)                       zero kernel
    consistent LOO estimate                               (oracle)

## 7. The derivative-based realization (score form; PFF-ready)

Full-posterior score under pi-hat, strong constraint, prior p_0:

    S_full(x_0) = grad log p_0(x_0)
                  + sum_s dnll(eps_s(x_0)) . grad h_s(x_0),
    grad h_s = (dM_s/dx_0)' H'      (the 4D-Var adjoint chain),

and since p(x_0 | y_{-t}) is proportional to
p(x_0 | y_{1:n}) / pi-hat(eps_t(x_0)):

    S_{-t}(x_0) = S_full(x_0) - dnll(eps_t(x_0)) . grad h_t(x_0).

The LOO score is the full score minus one known term. Ingredient list:
prior score (analytic at the initial time; the additive-model-noise
mixture in cycled windows), likelihood score under pi-hat (the Format A
/ CostJoNonGaussian machinery), the adjoint. NO forecast density
appears: score-based samplers need pointwise log-derivatives only, and
the strong-constraint x_0 formulation folds the dynamics into h_s. Any
asymptotically exact posterior sampler driven by S_{-t} sits at the
optimum: MALA, or the componentwise particle flow with inflation
1/gamma; the choice is engineering.

Cost ladder (p_{-t} is an O(1/n) KL perturbation of p_full):
  0. importance reweighting of the full-posterior ensemble,
     w ~ 1/pi-hat(eps_t): zero extra sampling; requires
     E[1/pi-hat] < infinity (fails at support walls; truncation / PSIS /
     refits -- see NOTES_CONVERGENCE.md section 8 caveat);
  1. ONE preconditioned flow/Newton step from the full-posterior
     particles along the subtracted-score direction: the
     influence-function / one-step-CV approximation, first-order exact
     in 1/n; repairs the support-wall failure by MOVING particles;
  2. few-step warm-started flow per t (start is O(n^{-1/2}) from the
     target);
  3. full per-t flows (gold standard, unnecessary at these sizes).

## 8. Boundary conditions

(i) Identifiability: with chaotic M the posterior filaments beyond the
Lyapunov horizon; realize n by many re-anchored windows (cycling), each
window's s^2 = O(1/n_window), the archive supplying the total.
(ii) Everything is GIVEN pi-hat: the fixed-point, first-order-with-
damping loop structure of NOTES_CONVERGENCE.md section 5 governs;
LOO optimality is optimality of the kernel given calibration.
(iii) Perfect model assumed; model error lands in the residual pool
(weak constraint reintroduces transition densities).
