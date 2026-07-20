# l95 closed-loop testbed

Runs the whole chain in Lorenz-95, where the truth is known: inject an
observation-error density, assimilate, estimate the density back from ensemble
innovations, feed it to the cost function, repeat. The point is that the
recovered density can be compared against the one that went in, which is
impossible with real observations.

## What it is meant to answer

  * does the loop converge, or oscillate
  * does it converge to the RIGHT density, or to a self-consistent wrong one
  * does the analysis improve as the density improves
  * is the ensemble reliable enough for the deconvolution, checked directly
    against the true background error rather than assumed

## Design decisions, and why

**3D-Var, not long-window 4D-Var.** l95 is chaotic, but that does not by itself
create multiple minima within a cycle: the observation operator is linear, so a
Gaussian 3D-Var cost is exactly quadratic. Non-convexity enters through the
non-Gaussian Jo, and the evolving-Gaussian construction keeps each inner loop
quadratic anyway, so the risk is outer-loop oscillation rather than a local
minimum. Long-window 4D-Var is where chaos genuinely produces multiple minima,
and there is no reason to take that on here.

**Particle flow, not EDA.** EDA draws its observation perturbations from the
ASSUMED R, and the premise of this work is that the assumed R is wrong. The
analysis ensemble then does not sample the true posterior, its spread is
calibrated to the assumption rather than to reality, and the deconvolution runs
on a spread the assumption produced. The bias direction is at least benign --
assume sigma_o too small, perturbations too small, ensemble under-dispersive,
deconvolution attributes the shortfall to observation error, sigma_o estimated
too large, negative feedback -- but that is a hypothesis, not a guarantee. EDA
is asymptotically correct only under linear-Gaussian assumptions with the right
R and B, and l95 is chaotic and nonlinear.

oops already contains a better option: src/oops/assimilation/PFF.h, a particle
flow filter, selected with `minimizer: algorithm: PFF` and exercised by the
eda_3dvar_pff tests. It is a Minimizer and calls J_.computeGradientFG, so the
flow is driven by the cost function gradient -- which in CostJoNonGaussian is
the true score g(d), not R^-1 d. The particles therefore move along the actual
non-Gaussian log-likelihood gradient, so the shape of the density informs the
ensemble and not merely a Gaussian variance.

It also avoids what rules out a plain particle filter here. In forty dimensions
with forty observations a bootstrap filter degenerates, the required number of
particles growing exponentially with the variance of the log weights.
Resampling makes the survivors unweighted but duplicates them, collapsing
ensemble diversity -- and diversity is exactly what the deconvolution consumes,
so an under-dispersed ensemble biases sigma_o high. Particle flow moves
particles rather than weighting and resampling them, which is the point of it.

Two caveats. The filter gives the posterior, so the prior ensemble for the next
cycle comes from forecasting the particles forward. And the flow depends on the
current density through the gradient, so the circularity does not vanish; what
changes is that it becomes exactly an EM structure -- sampling with the current
estimate, then updating the estimate -- rather than that plus a Gaussian
approximation layered on top.

**Background error matters more than usual.** The deconvolution removes the
background spread, so if the ensemble is under-dispersive the shortfall is
attributed to observation error and sigma_o comes out biased high. In l95 the
true background error is background minus truth, so ensemble reliability can be
checked rather than assumed. Do that every cycle.

## Two stages, deliberately separated

**Stage A: no filter at all.** Construct the ensemble directly -- draw members
and the truth from the same distribution, push them through the observation
operator, inject an error density, estimate it back. Exchangeability then holds
by construction rather than depending on a filter, so the estimator provably
should recover the injected density as the ensemble and sample sizes grow, and
any failure is the estimator or the plumbing. Needs no filter, no cycling, no
generated ensemble files.

**Stage B: cycled particle flow.** Now the ensemble comes from the assimilation
using the current density, circularity included. This is the actual research
question. Stage A supplies the reference answer, so when Stage B drifts it is
clear whether the feedback or the estimator is responsible. Doing Stage B first
would conflate the two.

## Step 1: injecting a known error density  (done)

l95 writes observations as H(truth) with no noise -- `make obs: true` is
`hofx.save("ObsValue")`, and `obs_error` in the generate block only fills the
ObsError column with the assumed error. So the generated file is the truth in
observation space and any density can be added to it.

    python3 inject_obs_error.py truth3d.obt noisy.obt \
        --density heavy --seed 1 --assumed-error 0.4 --truth-out truth.npz

Densities: gaussian, heavy (85/15 variance mixture), laplace, mirrored_gamma
(the density from the reference's idealised experiment). The draws and the
density parameters are saved to the npz so a later comparison does not depend on
regenerating the same random numbers.

`--assumed-error` sets what a Gaussian control run will believe, which is how
the control and treatment are made to differ in exactly one thing.

## Step 2: one cycle, does the density come back  (done for Stage A)

Before building any cycling: generate observations, run EDA once with the
Gaussian assumption, collect member H(x), estimate the density, and compare
against the injected one. If a single pass does not recover it there is no point
iterating, and the reason will be much easier to find here than inside a loop.

`collect_ensemble.py` does the reading. Each member writes its own observation
file and the cost function saves the first-guess departure there as `ombg`, so

    H(x_b^k) = ObsValue - ombg_k

and `collect` returns the observations and the member values. Those go to
`histograms_from_ensemble`, which pairs at the SAME LOCATION to build the
innovation histogram and the kernel, and `estimate_from_histograms` deconvolves
them.

WHAT STAGE A ALREADY CAUGHT. Feeding the estimator innovations and
perturbations as its X and Y is wrong. The innovation is already a difference,
so the estimator differences it again and one background variance survives:

    d = obs - member    variance sigma_o^2 + 2 sigma_b^2
    p = member - mean   variance sigma_b^2
    d_i - p_j           variance sigma_o^2 + 3 sigma_b^2
    kernel p_i - p_j    variance 2 sigma_b^2
    recovered           sigma_o^2 + sigma_b^2

Measured: recovered width 0.69 against a true 0.45, unchanged from 2000 to
20000 observations and 5 to 40 members. Passing raw observations and raw member
values is correct in principle but the estimator pairs at random, which puts the
field variability back into both histograms; that returned 2.15. Pairing at the
same location is both correct and well conditioned, and converges: 0.474 at
2000 observations, 0.454 at 8000 with 40 members, with kurtosis tracking the
truth to within about one unit.

The error was only visible because Stage A makes the truth EXCHANGEABLE with
the members rather than their centre. An earlier check with members centred on
the truth appeared to validate the wrong feed, because member - truth then has
variance sigma_b^2 instead of 2 sigma_b^2 and the mistake cancels.

THE SHIPPED l95 EDA CANNOT BE USED. eda_3dvar_1..4.yaml give every member the
same background file and vary only `obs perturbations seed`, so H(x_b^k) is
identical across members, the kernel collapses to a delta and there is nothing
to deconvolve. That ensemble samples analysis uncertainty arising from
observation error, not background error. `collect` refuses this case.

So the testbed writes its own member configurations, each pointing at its own
background from a real ensemble. The pff configurations show both the pattern
and the file names: each member reads
Data/forecast.ens.<N>.2010-01-01T00:00:00Z.P1D.l95, written by genenspert. That
is what makes them a genuine background ensemble where eda_3dvar_1..4 are not.

`reliability()` is the diagnostic that justifies the whole exercise. If members
and truth are exchangeable draws then RMSE(ensemble mean) = spread sqrt((K+1)/K).
Tested on constructed ensembles it returns a ratio of 1.10 for a reliable one,
0.41 when the members are too tightly clustered and 1.76 when too spread, with
the corresponding warning about which way sigma_o will be biased.

## The artifact floor, and when a tail can be believed at all

Stage A also found something that changes how any recovered density should be
read. Deconvolution under a non-negativity constraint cannot perform the
cancellations exact inversion needs, so the solver returns a density peaked in
the middle with tails that are not in the data. Run on a GAUSSIAN truth the
estimator reports excess kurtosis around +4, and that does not go away with
more observations, more members, more smoothing or finer bins. It is bias, not
variance: 2.5 times the observations with twice the members moved it slightly
up.

`null_calibration.py` measures it for a given configuration:

    sigma_o/sigma_b   floor mean   floor p95   width bias
        2.00             +0.66       +0.88        +4.5%
        0.80             +3.77       +4.28       +14.0%
        0.50            +20.49      +28.99       +33.7%

So whether a tail can be measured at all is decided by how much of the
innovation is observation error. Where the observation error is twice the
background spread the floor is near +1 and a recovered shape means something.
Where they are comparable, only strong non-Gaussianity clears the floor: the
85/15 mixture (true +8) does, a Laplace error (true +3) does not, and Stage A
reported +4.5 for Laplace against a floor of +4.3. Where the observation error
is half the background spread the estimator reports +29 on Gaussian data and
nothing about the shape can be believed.

Consequences worth carrying into the RRFS work: every exported density should
report its floor alongside it, and the threshold for acting on a heavy tail is
the floor rather than zero. A stratum whose resolvability is below about 0.7
should be used for a variance estimate only, if at all.

## Step 3: cycling

A driver alternating forecast and EDA over many cycles, since the existing l95
tests are single-cycle. Diagnostics per cycle: analysis error against truth,
ensemble spread against actual background error, and the estimated density
against the injected one.

## Step 4: the loop

Re-estimate the density from the accumulated innovations, write it into the
configuration, re-run, repeat. Track the distance between successive densities
and between each density and the truth.
