Placement (repo root = stable_DOEE, branch jedi-density-export).
First reconcile git: pull --rebase on the Mac, then push.

  l95-testbed/stage_c_smoothing.py   -> replace (adds apply_tail_guards)
  jedi_export/phase3_cycle.py        -> replace (imports guard, applies at
                                        the accept site, adds
                                        --tail-rate-max, --tail-sigma-floor)
  jedi_export/l95_mirror.py          -> replace (archive_windows pooling +
                                        the same two guard knobs; default
                                        settings reproduce every pinned row)
  jedi_export/diag_*.py (3 files)    -> new diagnostic scripts
  HANDOFF_STAGE_C_UPDATE.md          -> splice over the SESSION CLOSE block
                                        of l95-testbed/HANDOFF_STAGE_C.md
  tail_guards_tracked.patch          -> alternative to the two replacements
                                        above for the tracked files only
                                        (git apply from repo root); the
                                        mirror and diag files still copy in
                                        as full files

Verification run in the container before packaging: test_phase3_driver
ALL PASS; mirror pinned baseline exact at default knobs (0.158/0.123,
ESS 33/31); guard reproduces the E4 experiment digit-for-digit; seeds
7/11/21 remedy-vs-stock table in the handoff update.

VM confirmation command (rung-2 configuration plus the remedy):
  python3 jedi_export/phase3_cycle.py --build ~/jedi/src/build/oops/l95/test \
      --oops ~/jedi/src/oops --members 40 --iters 8 --refresh-obs \
      --archive-windows 5 --tail-rate-max 1.3 --tail-sigma-floor 0.1
