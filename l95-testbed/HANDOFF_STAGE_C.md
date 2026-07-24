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
- PFF CALIBRATION CTEST (next oops branch, needs VM iteration):
  amplitude-0 configuration (all particles share ONE likelihood --
  the paper's PFF, and the template for Phase 3's loop invocation;
  the existing per-member-perturbed harness is an EDA-flow hybrid,
  design question flagged), a bespoke checker in the
  ObsErrorDiagZeroMeanPerturbations.cc pattern reading the 4 member
  analyses, comparing the ensemble MEAN to the DRIPCG 3dvar analysis
  (exact posterior mean, linear-Gaussian) and pinning the SPREAD --
  the correctness gate the norm diagnostic cannot be (antisymmetry
  caveat in 5b).
- Phase 1: 4D Gaussian-equivalence ctest on JEDI l95 (nongaussian-
  costjo fork), then non-Gaussian 4D reference.
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
