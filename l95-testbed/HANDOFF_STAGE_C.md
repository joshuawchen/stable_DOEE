# HANDOFF: Stage C (stage_c_smoothing.py) -- state as of this session's end

## What Stage C is

The smoothing-view program: strong-constraint 4D windows in anomaly
coordinates (deterministic linear evolution a R inside the window, prior
N(0, C), T obs times x m locations), the LOO principle
(NOTES_DOEE_DERIVATION.md section 5) made executable, and end-to-end
pipelines scored by 4D MAP regret against the true-density MAP. Three
modes: the ladder (--T-list, prior/residual/loo arms), the single-window
pipeline (--pipeline), and the prequential multi-window mode
(--windows: re-anchored windows, one archive, window w analyzed under
the density from windows 1..w-1; H1 exact; --archive-mode pooled or
recent(5)).

RECOMMENDED CONFIGURATION (wins on both densities through the
operational interface, MALA and PFF):
--feedback export --adaptive --loo-defense 0.01 --feedback-smooth 0.1
--archive-mode recent --export-gap-max 0.4

Loop-stability knobs (all default-off; pinned tables reproduce exactly
at defaults): --loo-defense DELTA (defensive-mixture LOO weights,
Hesterberg, pi-hat normalized numerically); --feedback-smooth SIG (the
SAMPLING copy of the fed-back density is Gaussian-smoothed; the
analysis keeps the full estimate -- in BOTH raw and export feedback, so
the two modes differ only in the analysis density). Always-on
hardening: export failures nonfatal everywhere; interior exact zeros in
raw feedback floored at the span's 1e-6 cut; non-finite fed-back scores
refused (previous density kept).

## Settled results (fixed seeds, reproducible)

1. LADDER (heavy): loo < residual < prior in L1 at every n.
2. SINGLE-WINDOW PIPELINE, heavy T=10, 16 reps: loo 0.0131 vs oracle
   Gaussian 0.0459; margin +0.0334 CI [+0.0244, +0.0422], 15/16.
   Single-window skewed loses (ceiling below the floor) -- superseded
   in the archive regime by result 6.
3. GAUSSIAN NULL (selftest-pinned): gaussM 7e-17; gaussI converges in
   2 iterations; loo premium 0.033; defended loo inert (sd 0.356 vs
   0.355, ESS 350).
4. PFF CERTIFIED, heavy, single-window AND prequential:
   - single-window 8 reps: pff-sdr 1.04-1.05, margins +0.0320 (gaussI)
     / +0.0314 (gaussB), 7/8, on top of the MALA block.
   - prequential 20 windows (recommended config, raw): trailing loo
     0.0018 vs ceiling 0.0518, margin +0.0501 CI [+0.0439, +0.0555],
     10/10, crossover from w0, w0 analytic calibration silent. THE
     CENTRAL FIGURE, on the operational sampler.
5. PLATEAU RESOLVED -- IT WAS LAM: under --adaptive the estimation
   floor falls with the archive (gaussI L1 0.15 -> 0.03 at N=8000).
5b. SAMPLER FIDELITY INSTRUMENT (--pff-fidelity): one window, true
   analytic density, MALA reference vs PFF over a bandwidth/budget
   grid, carried through LOO and the estimator (pooled-innovation
   mean/sd/skew, then L1 and ESS at matched seeds). This is the check
   an sd-ratio-only readout cannot do (odd-moment blindness). Also the
   JEDI PFF.h audit (jedi_export/patches/): three implementation
   defects found and patched (repulsion sign, frozen particle
   geometry, learning-rate branch clobber); the sandbox mirror never
   had them, so all sandbox PFF results certify the PAPER algorithm
   and transfer to JEDI only once the patch lands. DONE: branch
   pff-paper-conformance on joshuawchen/oops, b6065682 (conformance
   patch) + cfce66fa (reference regen; 'test output filename' kept in
   the yaml for future regens), pushed; oops_l95_eda_3dvar_pff green,
   133/133 l95 suite green with the patch. Acceptance read of the
   conformant flow: norm 100 -> 15.2 over ten steps, decelerating to a
   plateau, zero backtracks, eps steady at 0.05, inflation 1.00; Jo
   1003 -> 261 with Jb settling at ~218. CAVEAT for the calibration
   ctest: the PFF norm diagnostic sums updates over particles, and the
   repulsion term is pairwise-ANTISYMMETRIC, so it cancels exactly in
   that sum -- the norm is constitutionally blind to the sign bug the
   patch fixed (old and new iteration-0 norms agree to 14 digits).
   Member trajectories, pinned by the regenerated reference, carry the
   correction; a spread-vs-closed-form-posterior ctest is the proper
   correctness gate (Phase 0 list). Merge to nongaussian-costjo at
   J's discretion.
6. PREQUENTIAL CROSSOVER, skewed (MALA):
   - raw pooled: +0.0082 CI [+0.0046, +0.0115], 9/10, with a late-run
     wrinkle (stale rows, see 7).
   - raw recent(5): +0.0145 CI [+0.0120, +0.0168], 10/10, trailing loo
     0.0023, wrinkle gone.
   Even the near-parabolic boundary case crosses from w0. Decision
   rule, dynamic form: the ceiling decides WHETHER shape can pay; the
   archive decides WHEN.
7. ARCHIVE CURATION IS A DESIGN ELEMENT: recent(5) (2000 obs) BEATS
   pooled (8000 obs) on skewed under MALA -- rows generated under
   early wrong densities carry stale kernels that outweigh their data
   value. Cheap approximation of the self-consistent (batch-EM/regen)
   archive. CAVEAT: curation interacts with sampler fidelity (see open
   item on PFF skew).
8. EXPORT ACQUITTED AT THE RECOMMENDED CONFIGURATION: export feedback,
   skewed, recent(5), MALA: +0.0147 CI [+0.0126, +0.0167], 10/10,
   trailing loo 0.0021 -- IDENTICAL to raw's +0.0145. The full
   operational chain (estimate -> Format A -> non-Gaussian analysis)
   reproduces the raw loop on the hardest density. Pooled export run:
   +0.0059, 8/10 (sag = the stale-row wrinkle, not the export). The
   original export-feedback runaway is reattributed: the export
   AMPLIFIES spiral-degraded estimates (its failure is a CLIFF: exp
   pegs at 2.0 on estimates only ~0.13-0.15 from truth) but is
   faithful (gap <= 0.05) on clean ones.
9. INTERFACE PARITY LAYER: jedi_export/make_parity_fixtures.py +
   parity_fixtures.json -- deterministic exported specs from the
   ANALYTIC menu densities plus 23-point score/variance reference
   tables per density covering every evaluator branch, floats at 12
   significant digits (regeneration byte-stable across platforms;
   --check verified green on the VM). JSON is valid YAML: the C++
   ctest reads it with eckit::YAMLConfiguration and compares
   oops::NonGaussianDensity at tol 1e-9. Building this caught and
   fixed a real defect: sigma_at_mode was grid-fragile (left-Riemann
   reconstruction error aliasing through a float-asymmetric fit
   window; 0.348 instead of 0.400 on an exact Gaussian at dx =
   sigma/10). Fixed by trapezoid reconstruction + one-ulp window
   guard; sandbox pinned tables untouched (score semantics never reads
   sigma at mode); the C++ variance path consumed the wrong value
   until now. C++ SIDE DONE: branch format-a-parity-ctest (0a12c229)
   on joshuawchen/oops, oops_assimilation_formata_parity GREEN ON
   FIRST RUN -- all 184 comparisons (4 densities x 23 points x
   score+variance) at 1e-9, first compile, no adjustment on either
   side: the mirror and the evaluator were in exact agreement, and
   drift on either side now fails a ctest. Fixtures canonical in
   stable_DOEE; regenerate there, copy via make_parity_branch.sh.
10. MIRRORED_GAMMA retired (compact support out of design class);
    "skewed" replaces it.

## Instabilities: attributed and fixed

A. EXPORT-AMPLIFIED SPIRAL (historical): export feedback diverged on
   skewed under the UNSTABILIZED loop; with the loop stabilized the
   export reproduces raw (result 8). Remaining export work is
   HARDENING, not a blocker: per-side tail treatment so degraded
   skewed estimates degrade the export gracefully instead of tripping
   the cliff; plus the C++ parity ctest.
B. ROUGHNESS-WEIGHTS SPIRAL: adaptive estimates are rough exactly
   where 1/pi weights are most sensitive; rough density -> ESS
   collapse -> duplicated archive rows -> adaptive under-smooths ->
   rougher. Fixed by the smoothing decouple + defense + sanitation;
   ESS 318-374 in every stabilized run. Two hard failures fixed on the
   way: interior exact zeros -> log -inf -> nan scores; export
   find_mode throwing on mangled estimates.

## Open / queued

- PFF SKEW SENSITIVITY: CLOSED, three-way. (i) Flow-bias hypothesis
  refuted by the fidelity diagnostic: under the true skewed density,
  PFF matches MALA's posterior marginals to dskew <= 0.014 and sd
  ratio 1.000-1.004, FLAT across bandwidth (0.3-1.2) and budget
  (300-1200) -- no tuning even needed. (ii) Member-statistics
  hypothesis refuted by the estimation-stage columns: PFF ensembles
  give L1 0.056-0.067 vs MALA's 0.071 at matched LOO seeds (ESS
  361-363 vs 365) -- at least as good for the estimator. (iii) Seed
  replication: at seed 11, PFF-recent WINS +0.0157 CI [+0.0141,
  +0.0175] 10/10, slightly ahead of MALA's +0.0148 on the same
  realization. The seed-7 PFF loss (-0.0172) was a LOOP-REALIZATION
  EVENT, not a sampler property. Residual (low priority): the
  boundary-density loop has a small per-realization excursion
  probability (1 of 4 recent-archive skewed realizations),
  sampler-agnostic as far as measured, visible in-flight via the
  ESS/L1/exp columns; a seed ladder would estimate the rate.
  Operationally: the recommended configuration wins on both densities
  with BOTH samplers at the replicated level, and the fidelity grid
  being flat means there is nothing to tune.
- Export hardening: DONE as the SELF-GAP REFUSAL GATE
  (--export-gap-max, default off, 0.4 recommended): the export
  measures its own L1 disagreement with the raw estimate and refuses
  to ship past the threshold, routing into the loop's existing
  spec-None handling (keep the previous density). Grounded in the
  measured cliff: healthy gaps <= 0.24 in every pinned run, failures
  0.53-2.0 only on degraded estimates. Guarded in test_loop_guards
  (3b). Deeper per-side tail work is DEMOTED to optional: the export
  is acquitted on clean estimates and now refuses bad ones, and the
  stabilized loop makes bad ones rare (excursions), during which
  refusing is the right behavior anyway. Port note: the same gate
  belongs in the Orion export pipeline (run_doee_export) at Phase 4.
- PFF CALIBRATION CTEST: DONE. Branch pff-calibration-ctest
  (6ef3251c + 03e2d2ae), four-test chain all green: ens-mean
  background (l95_ens_mean_variance.x) -> exact DRPCG 3D-Var from the
  mean background (= the conformant flow's target posterior mean;
  prior anchor is x_bar_b) -> shared-likelihood PFF at obs
  perturbation amplitude 0 (the paper's configuration and the Phase 3
  loop template) -> bespoke checker (l95/test/lorenz95/
  PFFCalibration.cc). FIRST MEASURED CALIBRATION OF THE CONFORMANT
  FLOW: ensemble mean rms deviation from the exact posterior mean
  0.2307 (21-step x eps 0.05 budget residual: the norm plateaus at
  ~15% of initial, so ~85% of the mean signal is captured -- budget,
  not correctness; the converged sandbox flow matches to ~0.001);
  ensemble SPREAD 0.5564 -- contracted from the ~0.6 prior, NOT
  collapsed: the anti-collapse gate the summed-update norm cannot be,
  with hard yaml-configurable bands (mean tolerance 0.30 documented as
  the budget bound; spread in [0.01, 0.80]). Member analysis filenames
  are DATELESS (Data/pff_calibration.mem00N.a.l95) -- learned at first
  contact.
- PHASE 1 4D GATES: DONE. Branch nongaussian-4dvar (3d8b5d43 ..
  f4be3db0), both tests green.
  (a) 4dvar_gaussequiv: the non-Gaussian machinery with a DEGENERATE
  Gaussian Format A spec inside full 4D-Var reproduces stock
  4dvar_dripcg to 7-8 significant digits at every evaluation (J
  124.1839566 vs 124.1839529 -> 3.2835359 vs 3.2835359 -> 3.0311036
  vs 3.0311036); the residual is EXACTLY float32(0.4) vs double 0.4 in
  the obt obs-error storage. Strict-generalization PROVEN at the 4D
  application level.
  (b) 4dvar_nongaussian: the heavy Format A density in full 4D-Var
  with EvolvingSigma saved -- the first genuinely non-Gaussian 4D
  reference (DRPCG, ninner 10, 20-line reference from a completed
  run).
  FINDING on the way (jedi_export/patches/jojc_warning.patch, on the
  branch): the DR minimizers' checkQuadraticCostFunction asserted
  quadratic JoJc >= 0 -- a Gaussian sum-of-squares premise. Small
  theorem: with the evolving-variance SECANT Hessian, W^-1 g = (d - m)
  exactly, so the quadratic model's Jo minimum is EXACTLY ZERO --
  except inside the mode window and at the sigma floor, where the
  variance is deliberately not (d-m)/g and small negative excursions
  are legitimate (measured: -0.111 and -0.718 against J0 = 33,
  identical under DRIPCG and DRPCG). Fix: JoJc-negative demoted to a
  warning; Jb-negative, inf, NaN remain fatal. Compatibility note for
  Phase 3/4: expect these warnings in non-Gaussian minimizations;
  they are diagnostics, not faults.
- C++ TRUNK MERGED: nongaussian-costjo at 11ff16cd carries all four
  branches (--no-ff merges, conflict-free); the merged trunk builds
  and passes 153/153 l95+assimilation tests with ZERO reference
  regeneration -- the composition proof. Branch history (for the
  record): pff-paper-conformance b6065682+cfce66fa;
  format-a-parity-ctest 0a12c229; pff-calibration-ctest
  6ef3251c+03e2d2ae; nongaussian-4dvar 3d8b5d43..f4be3db0 (carries the
  jojc warning fix).
- Phase 3 (IN PROGRESS): jedi_export/phase3_cycle.py -- rung 1, the
  first closed DOEE loop through real JEDI. Fixed l95 window, iterated:
  N-member amplitude-0 ensemble analysis under the current Format A
  density (iteration 0 = the degenerate Gaussian spec, so only the
  non-Gaussian code path runs) -> oman departures collected from the
  member obt outputs -> defended LOO -> adaptive DOEE -> self-gated
  export -> next iteration's block. Errors INJECTED from a known menu
  density (inject_obs_error.py), so per-iteration L1 against truth is
  measured.
- THE PFF STEPPING/NORM DIAGNOSTIC ARC (first production contact at
  N=40 found two real defects; five wrong theories were killed by
  one-command measurements before the instrument that ended it -- a
  VERBATIM numpy transcription of PFF.h's update law + controller
  (exact B from ErrorCovarianceL95's spectral-Gaussian circulant,
  exact kernel h^2 = SD^2/N, exact eps schedule) that reproduces JEDI
  behavior offline and made fixes testable without VM round-trips):
  (1) TRANSPORT: the update carries 1/N with fixed eps and a fixed
  iteration budget, so cumulative transport goes like eps*T/N -- N=40
  at the stock 21-iteration budget is starved 10x vs N=4. Workaround
  (yaml): --pff-ctcheck huge (the x1.5 growth ratchet has NO ceiling
  and the norm check is blind to the damage; disable it) plus
  --pff-outer ~ 21*N/4. Principled fix: AdaGrad-lineage stepping (the
  SVGD default, what the sandbox mirror uses; why 400 sandbox
  particles never saw any of this). Message for Chih-Chi.
  (2) THE PARTICLE-0 CATAPULT (BUG, FIXED): computeNorm summed the
  per-particle updates INTO rank 0's update variable (by-reference
  clobber) to form the collective norm -- rank 0 then applied the
  N-fold summed update as its own step every iteration, and each
  rank's controller watched a DIFFERENT norm (rank 0 the sum, the
  rest their own). Explains the measured mem001 pump-peak-recover
  (sd 1.67 -> 7.9 @ k~90 -> recovery at exactly the (eps/N)
  contraction rate), the impossible ensemble-mean drift, and why
  every eps/SD tuning failed. Transcription reproduces it exactly
  (bug on: mem0 explodes, ensemble contaminated; bug off: healthy).
  Fix: jedi_export/patches/pff_norm_fix.patch (copy-based summation +
  collective norm broadcast), branch pff-norm-fix; references
  regenerated; CALIBRATION IMPROVED: meanDev 0.231 -> 0.174, spread
  0.556 -> 0.395 (the old numbers included the corrupted member).
  Post-fix N=40 Gaussian control (ctcheck 999, outer 210): sd 0.334
  (obs-error floor), ESS 33/40, L1(heavy truth) 0.119 --
  sandbox-grade from 120 obs x 40 members through real JEDI.
  Acquitted along the way (each by direct measurement): EffectiveError
  /R drift (constant 0.4000 all iterations), ObsBias coordinate (0.000
  flat), position bookkeeping (getFirstGuess exact), eps stability per
  se, bandwidth-spread balance, common-background-offset coupling.
- RUNG 1 CLOSED (depth-2 protocol, matching the sandbox pipeline):
  it0 L1 0.119 (ESS 33, sd 0.334 at the obs-error floor), it1 under
  the fed-back heavy density L1 0.158, gate-accepted, flow healthy
  (norm ~1-2%, Jo monotone) -- the closed DOEE loop demonstrated on
  real JEDI. TRANSPORT LAW VALIDATED AT A THIRD POINT: N=100 at
  eps*T/N matched (outer 525) gives sd 0.316, L1 0.132, ESS 82 --
  identical convergence at matched transport, ESS fraction preserved;
  N=100 at outer 210 (transport-starved) fails as the law predicts
  (sd 0.670, ESS 9). All rows in the VM's phase3_table.log.
- FIXED-DATA ITERATION DRIFT (characterized, gain > 1): depth-3+ on
  one window's 120 obs drifts superlinearly (0.119 -> 0.158 -> 0.433)
  with the FLOW healthy every cycle (the monitor separates DA health
  from estimate health). Refresh + recent-archive pooling + smoothing
  (rung 2 as built) SLOWS it (settles L1 ~0.33-0.42 over 8 cycles)
  but gain stays > 1. Two live suspects, discriminators queued for
  next session (container-harness-tested before shipping, per the new
  workflow rule): (1) TIME-DISPLACEMENT CONTAMINATION, ranked first --
  the 120 obs sit at THREE times (22:30/00:00/01:30) all compared to
  the 00:00 state; the displacement is FIXED across refreshes (same
  truth), so the pooled archive accumulates the same structured
  residual; discriminator: estimate from the 40 synchronous obs only.
  (2) The ~20% member-spread deficit (measured diversity 0.172 vs
  theoretical 0.216 -- NOT near-copies, the near-copy theory is
  retired) interacting density-dependently; discriminator: small
  perturbed-obs spread inflation. Also retired post-fix: the
  kernels-alive regime at N=40 under fixed eps (SD 1.6 control still
  over-disperses, sd 0.845) -- dead-kernel descent is the only
  working regime without AdaGrad, sharpening the Chih-Chi case.
- WORKFLOW (established this session): the assistant's container
  holds a live clone + full dependency stack; guard suite + selftest
  green there (one benign scipy delta in a non-gated diagnostic);
  jedi_export/test_phase3_driver.py runs the ENTIRE driver against a
  stub build tree offline -- no driver edit ships without it passing.
  TODO: package the verbatim PFF transcription (the instrument that
  found the catapult) as jedi_export/pff_transcription.py.
- pff-norm-fix PUSHED (fd90c9da); merge into nongaussian-costjo
  pending -- NOTE the trunk still carries the catapult until then.
- SESSION CLOSE (post-rung-2 characterization; chronological):
  1. K ACQUITTED: N=100 at matched transport (outer 525) reproduces
     N=40's feedback trajectory (it1 0.209/it2 0.259 vs 0.205/0.231)
     with ESS 77 -- the core deficit is STRUCTURAL (LOO weight
     contrast 1/pi is flat in a peaked core; no K fixes it). Remedy
     dial: the exported spec's MODE WINDOW (extends the
     uniform-shrinkage zone the LOO provably handles). N=100 it1 flow
     went briefly unstable (norm 444%, heavy-core stiffness at the
     long budget) -- caught by the monitor, logged for the memo.
  2. SMOOTHING CONVICTED AND EXPLAINED: --feedback-smooth redirects
     LOO weights to the smoothed raw estimate while JEDI analyzes
     under the exported spec -- an analysis/weighting DENSITY MISMATCH
     the sandbox structurally never has (its st["nll"] drives both
     sampling and weighting). Importance consistency requires weights
     under the density the members were generated under. Default now
     0.0; this is the CORRECT configuration, not a workaround.
  3. THE MIRROR PROGRAM (J-directed: run everything off-VM, VM only
     for JEDI ground truth + one-time data export):
     - jedi_export/pff_transcription.py (COMMITTED): verbatim offline
       PFF twin; selftest reproduces the catapult, the contamination,
       the transport law. The instrument that cracked the case.
     - jedi_export/test_phase3_driver.py (COMMITTED): full driver
       executes against a stub build tree offline. WORKFLOW RULE: no
       driver edit ships without it passing in the container.
     - jedi_export/extract_mirror_data.py + vm_data/phase3_fields.npz
       (COMMITTED): one-time VM export of real fields (y, H(truth) at
       all 3 times, 40 member backgrounds, oman). Snapshot notes: the
       obs are the REFRESHED final-cycle draw (r sd 0.417, not seed
       7's 0.345), ombg 1.664, common offset +0.215, within-point
       scatter 0.329 (= JEDI's it0 floor, scalar theory validated),
       truth displacement 0.052 (4x smaller than synthesized worlds
       produce -- the reason synthesized calibration failed).
     - jedi_export/l95_mirror.py (COMMITTED): the full-loop twin --
       real L96, exact B (construction verified line-by-line against
       ErrorCovarianceL95.cc incl. the Eigen FFT convention), 3-time
       obs, verbatim flow, native estimator chain, closed-form 3DVar
       control. FieldWorld runs on the real npz. CALIBRATION: it0
       GATE GREEN on real fields (ESS 33 exact, gaussian fingerprint
       core 1.09/tail 0.33, sd 0.406 consistent with this draw's
       floor). OPEN: it1 fingerprint -- JEDI's fed-back analysis
       releases tails (0.97), the mirror's stays near-Gaussian;
       suspect is the it0-exported spec's tail slopes on this draw;
       next step is evaluating the mirror's exported spec sigma
       structure and, if needed, extending the extractor to pull the
       VM's phase3_it0_density.npz for direct comparison.
     - Estimation-chain twin (statistical member model): ABANDONED by
       its own calibration gate (3.7x off). Boundary mapped:
       update-law mirrors calibrate (self-contained arithmetic);
       statistical models of analysis output do not (member stats
       inherit the full analysis geometry). Full-loop computation is
       the right architecture, and it works.
  4. NEXT SESSION OPENS WITH: the it1 fingerprint diagnosis, then the
     mode-window sweep in the calibrated mirror, then ONE JEDI
     confirmation run of whatever the mirror concludes. Also queued:
     merge pff-norm-fix into the trunk; the Chih-Chi memo (AdaGrad +
     transport law + catapult + the N=100 stiffness event); the
     smoothing-port line audit is CLOSED (see 2).
- Rung-1 influence-step LOO: the transport upgrade; the natural home
  for skew if PFF's bias is structural.
- Figure set: crossover curves (heavy PFF, skewed export) when the
  above close.

## Tooling notes

Repo stable_DOEE, branch jedi-density-export; Mac checkout
~/Downloads/scas-paper/rrfs-ufo-update/stable_DOEE (bridge-writable;
python3 BLOCKED on the bridge -- execution on the VM or in the
assistant sandbox); commits/pushes from J's terminal. VM venv
~/jedi/venv (numpy 2.x: np.trapz is gone). Fixed seeds: every table
here reproduces exactly at default knob settings; prequential headers
now print sampler, feedback, archive mode, and all active knobs.
