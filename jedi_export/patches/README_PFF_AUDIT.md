Apply on the joshuawchen/oops fork, branch nongaussian-costjo (verified
to apply cleanly on 79388e4):

    git apply pff_paper_conformance.patch

Three correctness fixes that make PFF.h match Hu & van Leeuwen (2021)
and standard SVGD -- corrections TOWARD the paper, no algorithmic
additions. The l95 eda_3dvar_pff ctest references will need
regeneration (spread and increments change; with the repulsion sign
fixed, the collapse pressure is gone and 'inflation factor' should stay
at 1.0).

1. REPULSION SIGN. The kernel-gradient term must be grad_{x_j} K
   (gradient with respect to the SUMMED particle) = +(x_i - x_j)/h^2 K;
   the code implemented -(x_i - x_j) (gradient with respect to the
   wrong argument), turning repulsion into attraction -- a systematic
   ensemble-collapse pressure. With h^2 = alpha*B, the B-preconditioning
   cancels and the correct term is +(x_i - x_j) o K / alpha.

2. FROZEN PARTICLE GEOMETRY. Kernel, repulsion, and the prior anchor
   were evaluated at getBackground() -- the STATIC prior members -- for
   the entire pseudo-time flow; only the Jo gradient followed the moving
   particles. The paper evaluates the interaction terms at the CURRENT
   particles each step. Fix: xxCur = background + first-guess increment
   feeds dxi; xxMean stays the FIXED prior-member mean (the paper's
   x_bar_b). Companion fix: drop addGradientFG so rr is the Jo-only
   gradient -- with current positions in the explicit prior term,
   keeping addGradientFG would double-count the prior score from outer
   iteration 1 onward. (In the ORIGINAL frozen-geometry code the two
   pieces happened to reconstruct the correct prior term
   -(x_current - x_bar_b); once the geometry moves, the explicit term
   carries it alone.)

3. LEARNING-RATE CLOBBER. The unconditional trailing 'dx = dxij;'
   overwrote every branch of the adaptive-eps logic: on the
   norm-increased backtrack branch and the minimum-learning-rate branch
   the RAW UNSCALED step was applied -- the largest possible step,
   exactly when instability was detected. Each branch now owns its dx
   and the trailing assignment is removed. (Note also that 'iter_--'
   in the backtrack branch decrements a local copy; the outer loop is
   not rewound and the previous step is not undone -- unchanged by this
   patch, but worth knowing: 'redo pff' is a log message, not a redo.)

Verified correct as-is (for the record): the componentwise Schur-product
kernel structure matches the paper's dimension-wise kernel; the
bandwidth convention h^2 = standard_deviation^2 / Np is the paper's
B/Np with the YAML knob as the tunable scale; the Jo gradient is
evaluated at the current first guess; and the non-Gaussian CostJo
gradient flows into the likelihood term unchanged, so the Format A
densities drive the flow correctly once the above are fixed.
