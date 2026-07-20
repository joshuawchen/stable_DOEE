#!/usr/bin/env python3
"""ONE-TIME migration: lift the estimator out of the notebook into a module.

    python3 migrate_notebook.py ../stable_DOEE.ipynb ../stable_doee.py

Run this once, commit the generated stable_doee.py, and then delete this script.
From that point the module is the source of truth and the notebook imports from
it:

    from stable_doee import estimate_noise_pmf, finalize_pdf_cache

which is the right way round: exploration and production cannot drift apart,
and the estimator is importable without executing a notebook.

Only function definitions are lifted. Top-level statements between them are
demo and plotting code, and would otherwise run on import -- the first version
of this script did exactly that, leaving stray X, Y, N and grid_vals in the
module namespace and re-running the estimator whenever anything imported it.
Docstrings containing backslashes are made raw so the module imports without
SyntaxWarning.
"""
import re
import sys
import json

WANTED = {"estimate_noise_pmf", "finalize_pdf_cache", "logpdf_scalar",
          "dlogpdf_scalar", "pdf_scalar", "pdf", "_logsumexp",
          "_ensure_negative", "_quad_log_integral",
          "_segment_log_integral_linear_log"}

HEADER = '''"""Deconvolution-based Observation-Error Estimation (DOEE).

Recovers a non-parametric observation-error density from ensemble innovations,
following Hu, van Leeuwen & Geer (2024, QJRMS), by matching the innovation
histogram to the convolution of the estimated noise with the empirical
difference histogram, and solving a nonnegative, mass-constrained,
smoothness-regularised quadratic programme.

This module is the source of truth for the estimator; the notebook imports from
here.

Beyond the published method this implementation adds numerical stability: a
trimmed interior on which the log density is piecewise linear, quadratic-log
tails with curvature forced negative so the density stays integrable, and
smoothness regularisation on the first differences of the estimate.

Requires numpy and quadprog.
"""

import math

import numpy as np
import quadprog


'''


def extract(nb_path, out_path):
    nb = json.load(open(nb_path))
    src = "\n\n".join("".join(c.get("source", []))
                      for c in nb["cells"] if c["cell_type"] == "code")
    lines = src.splitlines()

    blocks, i = [], 0
    while i < len(lines):
        m = re.match(r"^(?:def|class)\s+(\w+)", lines[i])
        if m and m.group(1) in WANTED:
            j, last = i + 1, i
            while j < len(lines):
                ln = lines[j]
                if ln.strip() == "":
                    j += 1
                    continue
                if not ln.startswith((" ", "\t")):
                    break                     # back at top level: function ended
                last = j
                j += 1
            blocks.append("\n".join(lines[i:last + 1]))
            i = j
        else:
            i += 1

    missing = [w for w in WANTED
               if not any(re.match(rf"^def {w}\b", b) for b in blocks)]
    body = "\n\n\n".join(b.rstrip() for b in blocks)
    # raw docstrings where backslashes appear, to avoid SyntaxWarning
    body = re.sub(r'( {4})"""(\n\s*log \\)', r'\1r"""\2', body)

    with open(out_path, "w") as f:
        f.write(HEADER + body + "\n")

    print(f"wrote {out_path}: {len(blocks)} functions")
    if missing:
        print("  NOT FOUND (check the notebook):", ", ".join(sorted(missing)))
    print("\nNow: import it from the notebook, commit the module, delete this "
          "script.")
    return blocks


if __name__ == "__main__":
    nb = sys.argv[1] if len(sys.argv) > 1 else "../stable_DOEE.ipynb"
    out = sys.argv[2] if len(sys.argv) > 2 else "../stable_doee.py"
    extract(nb, out)
