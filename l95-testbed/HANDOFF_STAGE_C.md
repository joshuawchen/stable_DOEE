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

Loop-stability knobs added this session (all default-off; every pinned
table reproduces exactly at defaults):
- --loo-defense DELTA: LOO weights under the defensive mixture
  (1-delta) pi-hat + delta N(0, (3 s)^2) (Hesterberg), pi-hat
  normalized numerically; bounds 1/pi, protects the ESS.
- --feedback-smooth SIG: the SAMPLING copy of the raw fed-back density
  is Gaussian-smoothed at bandwidth SIG; the analysis MAP keeps the
  full estimate. Decouples ESS stability (needs smoothness) from
  estimation accuracy (needs adaptivity).
- Hardening (always on, inert for healthy runs): export failures are
  nonfatal everywhere (spec None; in raw feedback the export is
  diagnostic-only and the loop proceeds); interior exact zeros in the
  raw feedback are floored at the span's own 1e-6 cut before log; a
  non-finite fed-back score is refused (previous density kept).

Working configuration for the headline prequential runs:
--feedback raw --adaptive --loo-defense 0.01 --feedback-smooth 0.1

## Settled results (fixed seeds, reproducible)

1. LADDER (heavy): loo < residual < prior in L1 at every n; naive
   residuals shrink as predicted; convergence as d/n -> 0.
2. SINGLE-WINDOW PIPELINE, heavy, T=10 (n=400), 16 reps: loo 0.0131 vs
   oracle Gaussian 0.0459 / iterated Gaussian 0.0465 nats/ob; margin
   +0.0334 CI [+0.0244, +0.0422], 15/16. T=3 positive not significant;
   T=25 margin narrowed (the "plateau", RESOLVED below -- it was lam).
3. SINGLE-WINDOW PIPELINE, skewed, 16 reps: loo loses (-0.027, -0.039)
   because this density's Gaussian ceiling (0.016 at T=10, 0.007 at
   T=25) sits below the single-window estimation floor. Superseded in
   the archive regime by result 7 -- the static half of the decision
   rule, not the final word.
4. GAUSSIAN NULL (selftest-pinned): gaussM regret 7e-17; gaussI
   converges to sd 0.389 in 2 iterations; loo insurance premium 0.033;
   defended loo (delta 0.01) inert in the null (sd 0.356 vs 0.355,
   ESS 350).
5. PFF CERTIFIED END TO END: single-window pipeline, heavy T=10, 8 reps
   under --sampler pff (kref 400, componentwise kernel, inflation
   1.05): pff-sdr 1.04-1.05 every window (analytic iteration-0
   calibration, band [0.90, 1.08], zero warnings); loo 0.0120, margins
   +0.0320 CI [+0.0150, +0.0484] (vs gaussI) and +0.0314 (vs gaussB),
   7/8 -- statistically on top of the MALA reference block. The
   operational sampler (oops PFF.h analog) runs the entire honest
   pipeline.
6. PLATEAU RESOLVED -- IT WAS LAM: under fixed lam 3 the prequential
   loo/gaussI L1 sat ~0.15 while N grew 400 -> 8000; under --adaptive
   the gaussI column rides the archive down to L1 0.03 at N=8000. The
   estimation floor falls with the archive when smoothing scales with
   the data; the crossover economics stand.
7. PREQUENTIAL CROSSOVER, BOTH DENSITIES (20 windows x n=400, working
   configuration above, MALA):
   - heavy: trailing-10 loo 0.0024 vs gaussM 0.0518 / gaussB 0.0506 /
     gaussI 0.0518; margin +0.0494 CI [+0.0438, +0.0547], 10/10,
     crossover from window 0. loo sits at 0.001-0.006 from w2 onward:
     ~95% of the Gaussian ceiling captured, near-true-MAP analyses,
     archive ~800 obs at crossover. gaussB ~ gaussM everywhere: oracle
     variance tuning recovers nothing; the entire 0.05 is shape. THE
     central figure.
   - skewed: trailing-10 loo 0.0086 vs Gaussians 0.0165-0.0168; margin
     +0.0082 CI [+0.0046, +0.0115], 9/10, crossover from window 0.
     Even the near-parabolic boundary case crosses once the archive
     pushes the floor under its small ceiling.
   Decision rule, final dynamic form: the ceiling (score nonlinearity
   over the data's range, computable from the estimate) decides WHETHER
   shape can pay; the archive decides WHEN it starts paying.
8. MIRRORED_GAMMA retired from experiment menus (compact support is
   outside DOEE's design class); replaced by "skewed" (skewmix,
   analytic, Gaussian tails). Wall lessons kept in the notes.

## The closed investigation: shape-loop instability, fully attributed

Two distinct instabilities were isolated and fixed this session, each
convicted by the gaussI control (identical LOO machinery, stable
throughout):

A. EXPORT PROJECTION under accumulating skew (prior session's suspect,
   convicted): with --feedback export the skewed loop ran away (pooled
   by w~7, regen faster); with --feedback raw both runs stayed bounded
   for 20 windows while the exp column showed the export diverging
   harmlessly (gaps to 2.0 late in the relax run). Relaxation (0.5) is
   unnecessary and mildly harmful once the export is out of the loop
   (adds memory; trailing margin -0.0047 vs -0.0010 raw-only).
   => Engineering item: per-side junction/curvature in gaussian_tails.
   Heavy's export gaps stay 0.01-0.07 all run: the defect is
   skew-specific.

B. ROUGHNESS-WEIGHTS SPIRAL, adaptive x loop (new this session): the
   adaptive estimator is licensed to be rough exactly where 1/pi
   weights are most sensitive. Rough fed-back density -> ESS collapse
   (34-74) -> archive rows resampled from few effective members ->
   duplication violates the adaptive noise model -> under-smoothing ->
   rougher. Also two hard failures on the way: interior exact zeros
   (NNLS positivity) -> log -inf -> nan scores -> singular Laplace
   solve; and the export's find_mode throwing on a mangled estimate.
   All fixed (see knobs/hardening above). With the working
   configuration: ESS 318-374 across every window of both prequential
   runs.

## Open / queued

- Late-run skewed wrinkle: after w~13 the loo L1 drifts 0.05 -> 0.10
  and regret ~0.002 -> ~0.011 (still winning 9/10). Absent on heavy.
  Suspects: stale pooled rows generated under early wrong densities
  (--archive-mode recent isolates this in one run), or loop noise
  scale at this L1. Diagnostic, not blocking.
- Export per-side tail fix (item A above): THE remaining engineering
  step between the sandbox result and the operational path, since JEDI
  consumes Format A while the working loop currently feeds back raw.
  Closing test: repaired export back in the loop reproduces the raw
  runs' stability and margins.
- Prequential replication under --sampler pff: the certification
  (result 5) is single-window; one run each of heavy/skewed prequential
  under pff ties both headline figures to the operational sampler.
- Rung-1 influence-step LOO (derivation note's ladder): the transport
  upgrade to importance reweighting; less urgent now that defense +
  smoothing hold the ESS, still the principled fix where reweighting
  starves.
- Consolidate the final figure set (crossover curves for both
  densities) when the above close.

## Tooling notes for the next session

Repo: stable_DOEE, branch jedi-density-export; Mac checkout under
~/Downloads/scas-paper/rrfs-ufo-update/stable_DOEE (bridge-writable);
VM runs via git pull; commits/pushes from J's terminal (bridge has no
git identity); python3 blocked on the Mac bridge, all execution on the
VM (venv ~/jedi/venv: numpy quadprog matplotlib). numpy 2.x on the VM:
np.trapz is gone (one past crash). Fixed seeds throughout: every table
in this handoff reproduces exactly at default knob settings.
