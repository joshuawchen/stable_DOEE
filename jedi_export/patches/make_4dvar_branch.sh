#!/usr/bin/env bash
# Create the nongaussian-4dvar branch: Phase 1's two 4D gates on JEDI
# l95. (1) 4dvar_gaussequiv: the non-Gaussian machinery carrying a
# DEGENERATE Gaussian Format A spec (sigma 0.4, exact tails) in the
# full 4D-Var cost -- its cost trajectory must reproduce the stock
# 4dvar_dripcg reference, which is the application-level proof that the
# non-Gaussian path is a strict generalization. (2) 4dvar_nongaussian:
# the heavy Format A density in 4D, the first genuinely non-Gaussian 4D
# reference. Both derived from the committed 4dvar_dripcg.yaml.
#
#   bash make_4dvar_branch.sh [path-to-oops-checkout]
set -euo pipefail

OOPS="${1:-}"
if [ -z "$OOPS" ]; then
  OOPS="$(find "$HOME" -maxdepth 9 -name PFF.h \
          -path "*src/oops/assimilation*" 2>/dev/null \
          | head -1 | sed 's#/src/oops/assimilation/PFF.h##')"
fi
[ -n "$OOPS" ] && [ -d "$OOPS/.git" ] || {
  echo "oops checkout not found; pass it explicitly"; exit 1; }
cd "$OOPS"
git config user.email > /dev/null 2>&1 || {
  echo "set a git identity first"; exit 1; }

git fetch origin
git checkout -f -B nongaussian-4dvar origin/nongaussian-costjo

python3 - << 'PYEOF'
TI = "l95/test/testinput/"
TO = "l95/test/testoutput/"
base = open(f"{TI}4dvar_dripcg.yaml").read()

GAUSS = """      non gaussian cost:
        mode: 0.0
        # degenerate interior: all departures fall in the exact tails
        grid spacing: 2.0e-6
        stable min: -1.0e-6
        stable max: 1.0e-6
        log slopes: [0.0]
        # exact Gaussian tails for sigma^2 = 0.16
        left log slope: 6.25e-6
        left curvature: -6.25
        right log slope: -6.25e-6
        right curvature: -6.25
        sigma at mode: 0.4
        mode window: 0.0
"""

HEAVY = """      non gaussian cost:
        mode: 0.0
        grid spacing: 0.25
        stable min: -3.0
        stable max: 3.0
        log slopes: [0.669724, 0.728092, 0.796896, 0.878894, 0.977676, 1.097741,
                     1.243992, 1.418999, 1.611975, 1.759015, 1.628223, 0.744879,
                     -0.744879, -1.628223, -1.759015, -1.611975, -1.418999, -1.243992,
                     -1.097741, -0.977676, -0.878894, -0.796896, -0.728092, -0.669724]
        left log slope: 0.6437768240
        left curvature: -0.05
        right log slope: -0.6437768240
        right curvature: -0.05
        sigma at mode: 0.4
        mode window: 0.25
        save evolving sigma: true
        evolving sigma group: EvolvingSigma
"""


def make(name, block):
    y = base.replace("  observations:\n    observers:",
                     "  observations:\n    jo type: evolving gaussian\n"
                     "    observers:")
    y = y.replace("      obs operator: {}",
                  "      obs operator: {}\n" + block.rstrip("\n"), 1)
    y = y.replace("4dvar_dripcg", name)
    y = y.replace(f"test:\n  reference filename: testoutput/{name}.test",
                  f"test:\n  reference filename: testoutput/{name}.test\n"
                  f"  test output filename: testoutput/{name}.test.out")
    open(f"{TI}{name}.yaml", "w").write(y)
    open(f"{TO}{name}.test", "w").write("")


make("4dvar_gaussequiv", GAUSS)
make("4dvar_nongaussian", HEAVY)

cm = open("l95/test/CMakeLists.txt").read()
a1 = "  testinput/4dvar_dripcg.yaml\n"
assert a1 in cm
cm = cm.replace(a1, a1 + "  testinput/4dvar_gaussequiv.yaml\n"
                          "  testinput/4dvar_nongaussian.yaml\n", 1)
a2 = "  testoutput/4dvar_dripcg.test\n"
assert a2 in cm
cm = cm.replace(a2, a2 + "  testoutput/4dvar_gaussequiv.test\n"
                          "  testoutput/4dvar_nongaussian.test\n", 1)
a3 = """ecbuild_add_test( TARGET oops_l95_4dvar_dripcg
                  COMMAND l95_4dvar.x
                  ARGS testinput/4dvar_dripcg.yaml
                  TEST_DEPENDS oops_l95_forecast oops_l95_makeobs4d )
"""
assert a3 in cm
cm = cm.replace(a3, a3 + """
ecbuild_add_test( TARGET oops_l95_4dvar_gaussequiv
                  COMMAND l95_4dvar.x
                  ARGS testinput/4dvar_gaussequiv.yaml
                  TEST_DEPENDS oops_l95_forecast oops_l95_makeobs4d )

ecbuild_add_test( TARGET oops_l95_4dvar_nongaussian
                  COMMAND l95_4dvar.x
                  ARGS testinput/4dvar_nongaussian.yaml
                  TEST_DEPENDS oops_l95_forecast oops_l95_makeobs4d )
""", 1)
open("l95/test/CMakeLists.txt", "w").write(cm)
print("files written, cmake wired")
PYEOF

git add l95/test
git commit -m "Phase 1 4D gates: Gaussian equivalence and the first non-Gaussian 4D-Var

4dvar_gaussequiv: the non-Gaussian machinery with a degenerate Gaussian
Format A spec (sigma 0.4, exact tails) in full 4D-Var; its cost
trajectory must reproduce the stock 4dvar_dripcg reference -- the
application-level strict-generalization proof. 4dvar_nongaussian: the
heavy Format A density in 4D with EvolvingSigma saved. References
empty: regenerate on first run."

cat << 'EOT'

Branch nongaussian-4dvar ready. Build, regenerate, and RUN THE
EQUIVALENCE CROSS-CHECK (the actual Phase 1 gate):

  cd ~/jedi/src/build/oops
  make -j4 2>&1 | tail -5
  ctest -R "oops_l95_4dvar_gaussequiv|oops_l95_4dvar_nongaussian" --output-on-failure
  # both FAIL vs empty references; then:
  cp l95/test/testoutput/4dvar_gaussequiv.test.out \
     ~/jedi/src/oops/l95/test/testoutput/4dvar_gaussequiv.test
  cp l95/test/testoutput/4dvar_nongaussian.test.out \
     ~/jedi/src/oops/l95/test/testoutput/4dvar_nongaussian.test
  ctest -R "oops_l95_4dvar_gaussequiv|oops_l95_4dvar_nongaussian"   # green

  # THE GATE: the gaussequiv cost trajectory must match stock 4D-Var.
  # Compare the cost/J lines; agreement = equivalence proven at the
  # 4D application level (paste both into the chat for the read):
  grep -E "Quadratic cost|Nonlinear J|CostJo|CostJb" \
       ~/jedi/src/oops/l95/test/testoutput/4dvar_gaussequiv.test | head -20
  grep -E "Quadratic cost|Nonlinear J|CostJo|CostJb" \
       ~/jedi/src/oops/l95/test/testoutput/4dvar_dripcg.test | head -20

  cd ~/jedi/src/oops
  git add l95/test/testoutput
  git commit -m "Pin 4D gaussequiv and nongaussian references"
  git push -u origin nongaussian-4dvar

Note: the degenerate spec assumes the 4D obs error sd is 0.4 (matching
the 3D case). If the gaussequiv/dripcg cost lines disagree
STRUCTURALLY (not last-digit), check the ObsError column of truth4d
and set 'sigma at mode' and the tail curvatures (-1/sd^2) accordingly.
EOT
