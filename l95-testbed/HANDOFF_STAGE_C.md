# HANDOFF: Stage C (stage_c_smoothing.py) -- state as of this session's end

## What Stage C is

The smoothing-view program: strong-constraint 4D windows in anomaly
coordinates (deterministic linear evolution a R inside the window, prior
N(0, C), T obs times x m locations), the LOO principle
(NOTES_DOEE_DERIVATION.md section 5) made executable, and end-to-end
pipelines scored by 4D MAP regret against the true-density MAP. Three
modes: the ladder (--T-list, prior/residual/loo arms), the single-window
pipeline (--pipeline: oracle Gaussians, prior baseline, two ITERATED
pipelines sharing one posterior<->LOO<->estimate loop differing only in
what they retain -- gaussI keeps variance, loo keeps shape), and the
prequential multi-window mode (--windows: re-anchored windows, one
archive, window w analyzed under the density from windows 1..w-1; H1
exact).

## Settled results (all with fixed seeds, reproducible)

1. LADDER (heavy): loo < residual < prior in L1 at every n; naive
   residuals shrink as predicted; convergence as d/n -> 0. Section 8 of
   the derivation confirmed empirically.
2. SINGLE-WINDOW PIPELINE, heavy, T=10 (n=400), 16 reps: loo 0.0131 vs
   oracle Gaussian 0.0459 and iterated Gaussian 0.0465 nats/ob; paired
   margin +0.0334 CI [+0.0244, +0.0422], 15/16. THE headline. T=3:
   positive, not significant (estimation floor). T=25: margin narrows
   (+0.0075, CI grazing zero) -- the loo floor does not fall with
   within-window n while the Gaussian ceiling does; lam-vs-n scaling
   suspected, unresolved.
3. SINGLE-WINDOW PIPELINE, skewed, 16 reps: loo LOSES significantly
   (-0.027, -0.039). NOT a machinery failure: the Gaussian ceiling on
   this density is tiny (0.016 at T=10, 0.007 at T=25 -- gaussM's own
   regret IS the ceiling) and sits below the estimation floor. The
   operational decision rule: estimate shape where the shape moves the
   weights; heavy and skewed bracket the two sides. (Registered
   prediction "skew gives the largest margin" FAILED: regret ceiling is
   set by the score's nonlinearity over the data's range, not by
   cumulant size; this skewmix's log-density is nearly parabolic where
   the mass lives.)
4. GAUSSIAN NULL (selftest-pinned): gaussM regret 7e-17 (scoring chain
   machine-exact); gaussI converges to sd 0.389 in 2 iterations; loo
   insurance premium 0.033 nats/ob.
5. MIRRORED_GAMMA: retired from all experiment menus. Compact support is
   outside DOEE's design class; every wall pathology was the machinery
   correctly refusing an out-of-scope input. Wall lessons recorded
   (IS-LOO weight blowup, MALA wall handling, regret formally infinite
   for wall-violating analyses -- scored by rmse + violation fraction if
   ever run again). Replaced by "skewed" (skewmix: 0.8 N(-0.1, 0.25^2) +
   0.2 N(0.4, 0.45^2), zero mean, sd 0.361, skewness ~0.97, Gaussian
   tails, analytic everything).

## THE LIVE INVESTIGATION: the prequential shape loop diverges on skewed

Sequence of measured facts:
- pooled archive: loo wins windows 1-7, then slow runaway (L1 0.10 ->
  0.23 rising WITH the archive, regret to ~0.10 by w=19; trailing 0/10).
- regen archive (all rows regenerated under the current density each
  window) diverges FASTER (blowup by w~11, L1 to 0.67, regret to 0.39).
  This FALSIFIED the stale-statistics (online-EM-memory) diagnosis as
  primary.
- Control: gaussI uses the IDENTICAL LOO machinery (same kref=400, same
  reweighting, ESS 340-373) and is stable with L1 falling -- LOO
  sampling/accuracy exonerated by control.
- Smoking gun ordering in the regen run: the EXPORT GAP (new `exp`
  column: L1 between the raw recovered density and what the Format A
  export delivers) jumps to 0.26 at w=10 BEFORE the blowup; the loop
  feeds back the EXPORT (spec_nll), not the raw estimate; the gauss pipe
  only feeds back a scalar sd and is immune. Working diagnosis:
  feedback THROUGH the export projection (enforce_unimodal +
  gaussian_tails, built on symmetric densities) is the unstable element
  under accumulating skew; regen diverging faster is consistent
  (self-consistent archive = full-strength coupling to the mangled
  density).
- Theory correction on record: "regen = batch EM so runaway impossible"
  was WRONG as stated -- regen is the E-step, but the M-step here is
  DOEE + EXPORT, which carries no EM ascent guarantee. True penalized
  batch EM (M-step fits pi to full-posterior residuals; in
  linear-Gaussian its E-step spread performs the shrinkage correction
  exactly: R-hat = r r' + H Sigma H', the Desroziers identity) remains
  the principled formulation; the M-step choice (EM-implicit vs explicit
  LOO kernel) is an open, testable question.

PENDING (launched at session end, results not yet seen):

    python3 stage_c_smoothing.py --windows 20 --density skewed \
        --T-list 10 --persistence 0.97 --feedback raw
    python3 stage_c_smoothing.py --windows 20 --density skewed \
        --T-list 10 --persistence 0.97 --feedback raw --relax 0.5

--feedback raw bypasses the export in the loop (raw_nll_from_estimate:
interpolated log-density, decay-enforced linear log-tails, score by
differentiation); the export is still computed so the exp column shows
it diverging harmlessly if the raw loop is stable. Decision tree:
raw stable -> export convicted; fix = asymmetric tail handling in
gaussian_tails (per-side junction/curvature) -- top engineering item for
the JEDI branch either way, since the export is the operational
interface. raw diverges but relax 0.5 rescues -> expansive map, damping
is the cure. both diverge -> estimator-level (lam-3 NNLS under skew
feedback), investigate with plots.

## Also pending / queued

- PFF shakedown never ran: --pipeline --density heavy --T-list 10
  --replicates 8 --sampler pff. Read: pff-sdr column quiet in 0.95-1.02
  (free per-window calibration vs the analytic iteration-0 Gaussian
  posterior) and regrets matching the MALA heavy T=10 block => PFF
  certified; then optionally the full heavy table under pff.
- skewed multi-window with a stabilized loop: the crossover-with-archive
  demonstration (economics: Gaussian pays a CONSTANT total per window --
  ~19 nats heavy, ~6.5 skewed at T=10 -- while the shape cost falls like
  1/W; predicted crossover W ~ 3-4 for skewed).
- The loo floor at large within-window n (result 2, T=25): lam-vs-N
  scaling for the pooled estimate.
- --center-innovations (VarBC-style offset pinning) implemented, not yet
  decisive-tested.
- Consolidate everything into the final handoff set once the live
  investigation closes.

## Tooling notes for the next session

Repo: stable_DOEE, branch jedi-density-export; Mac checkout under
~/Downloads/scas-paper/rrfs-ufo-update/stable_DOEE (bridge-writable);
VM runs via git pull; commits/pushes from J's terminal (bridge has no
git identity); python3 blocked on the Mac bridge, all execution on the
VM (venv ~/jedi/venv: numpy quadprog matplotlib). Fixed seeds
throughout: every table in this handoff reproduces exactly.
