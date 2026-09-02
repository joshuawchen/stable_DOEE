# HANDOFF_STAGE_C update -- splice over the previous SESSION CLOSE block

- SESSION CLOSE (it1 fingerprint diagnosis; drift convicted; remedy
  certified; chronological):
  1. IT1 FINGERPRINT CLOSED: the mirror's it0 export is near-compact
     (tail sigmas 0.104/0.118; implied tail density 0.00 against heavy
     truth), so its it1 could never show tails. Single-variable tail
     dial (interior fixed, continuation curvature varied): recovered
     tail crosses 1.0 near fed sigma 1.0-1.4; departure kurtosis jumps
     +2.2 -> +7 as soon as tails open to sigma 0.35. The snapshot's
     real oman (pooled sd 0.515, kurtosis +6.17) sits in the open-tail
     rows. JEDI-vs-mirror was a PHASE OFFSET on one drift trajectory,
     not a twin miscalibration.
  2. FIXED-DATA DRIFT REPRODUCED OFFLINE (depth-6 mirror): 0.158 ->
     0.123 -> 0.260 -> 0.409 -> 0.416 -> 0.459, flow healthy every
     cycle. Anatomy: the it1 export opens the RIGHT tail only, sigma
     0.12 -> 0.98 in one cycle -- gaussian_tails' per-side polyfit
     over the outer 8 percent of mass is high-variance noise and the
     [0.3 sd, 3 sd] clip window admits an 11x jump; it2 releases;
     damage then migrates into an asymmetric interior while tails
     re-close. Tail-curvature positive feedback is the mechanism; the
     gap gate makes it worse by freezing the last accepted open spec
     (spec-None keeps the previous density).
  3. REMEDY CERTIFIED: per-side slew-rate limit on the export's tail
     sigmas -- exported sigma at most 1.3x the FED spec's per cycle,
     narrowing unconstrained -- plus an absolute sigma floor 0.1
     (0.25 x assumed error) guarding the separate narrow-ratchet cliff
     (a near-compact fed side detonates the 1/pi LOO weights; measured
     E6, sigma -> 0.05, ESS 12). Implementation: apply_tail_guards in
     stage_c_smoothing.py, dd-native with conditional assignment so
     untouched curvatures pass through bit-identical. Driver knobs
     --tail-rate-max / --tail-sigma-floor (defaults 0 = off; pinned
     tables intact; test_phase3_driver ALL PASS); mirror run_loop and
     CLI carry the same knobs. The guard reproduces the wrapper
     experiment's E4 trajectory to every printed digit. Seed
     replication (refresh + archive 5, depth 8, seeds 7/11/21):
     remedy trailing-3 L1 0.140 / 0.293 / 0.173 with ZERO refusals;
     stock 1.37 (late divergence) / 1.863 (5 refusals) / 1.676 (3
     refusals). 3/3, decisive.
  4. COUNTER-EXPERIMENTS: a symmetric two-sided RATE limit is the
     wrong shape (blocks the fast re-closing the loop needs after an
     excursion; E7 degrades). Steep-symmetric adds nothing on top of
     the rate limit (E5). Fixed-data deep iteration still collapses at
     it6 with the rate limit alone or with the floor: on frozen data
     the widening pressure is persistent, the limit converts the jump
     into a 1.3x-per-cycle ratchet at the fence, and the asymmetric
     ~0.3/~0.12 fed spec detonates the it6 estimate -- fixed data
     remains the diagnostic regime, gain above 1 is fundamental there;
     the remedy targets the operational (refresh) regime and holds it.
  5. TRANSPORT LAW, DENSITY DIMENSION: feeding the ANALYTIC heavy spec
     from wide backgrounds is transport-STARVED, not stiff -- initial
     flow norm 0.6 vs 34.5 under the near-compact it0 export (the
     heavy score is ~50x weaker) -- so adequacy depends on the fed
     density's score scale. Explains why every drifting JEDI cycle
     read flow-healthy. For the Chih-Chi memo.
  6. LOOP CHAOS NOTE (method): a sigma-space round trip in an early
     guard variant perturbed the fed curvature at the last ulp and the
     loop amplified it into visibly different trajectories by it1.
     Loop-level claims require seed replication; single-trajectory
     comparisons are draws.
- NEXT SESSION OPENS WITH: ONE JEDI confirmation run of the certified
  remedy -- rung-2 configuration plus --tail-rate-max 1.3
  --tail-sigma-floor 0.1 (needs the updated phase3_cycle.py and
  stage_c_smoothing.py on the VM). Then the mode-window sweep in the
  calibrated mirror (the structural core deficit; untouched). The
  extractor extension for per-iteration density npz is DEMOTED: the
  drift reproduced offline without it. Still queued: merge
  pff-norm-fix (fd90c9da) into nongaussian-costjo (trunk still
  carries the catapult); the Chih-Chi memo (AdaGrad + transport law,
  both dimensions + catapult + N=100 stiffness event).
- HOUSEKEEPING: Mac and GitHub diverged -- Mac holds unpushed de56e1c
  + f76cd38 (the two share a commit message; one looks accidental),
  GitHub holds 44c5501 (vm_data npz, pushed from the VM). git pull
  --rebase on the Mac, then push. New files this session (container
  origin, delivered in the bundle): jedi_export/diag_it1_fingerprint
  .py, diag_tail_dial.py, diag_tail_remedy.py; modified:
  jedi_export/l95_mirror.py (archive_windows + tail-guard knobs),
  jedi_export/phase3_cycle.py, l95-testbed/stage_c_smoothing.py
  (apply_tail_guards).
