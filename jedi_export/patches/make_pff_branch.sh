#!/usr/bin/env bash
# Create the pff-paper-conformance branch on the oops fork and apply the
# audit patch. Run on the machine that holds the oops checkout (the VM).
#
#   bash make_pff_branch.sh [path-to-oops-checkout]
#
# The branch is cut from origin/nongaussian-costjo, so fetch works from
# any state of the local checkout. After this script: build, run the
# ctest (it FAILS against the old reference by design -- the old
# reference pins the collapsing flow), inspect, regenerate the
# reference, commit, push. The printed checklist covers each step.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PATCH="$HERE/pff_paper_conformance.patch"
[ -f "$PATCH" ] || { echo "patch not found next to this script"; exit 1; }

OOPS="${1:-}"
if [ -z "$OOPS" ]; then
  OOPS="$(find "$HOME" -maxdepth 6 -name PFF.h -path "*src/oops/assimilation*" \
          2>/dev/null | head -1 | sed 's#/src/oops/assimilation/PFF.h##')"
fi
[ -n "$OOPS" ] && [ -d "$OOPS/.git" ] || {
  echo "oops checkout not found; pass it explicitly:"
  echo "  bash make_pff_branch.sh /path/to/oops"
  exit 1
}

cd "$OOPS"
echo "oops checkout: $OOPS"

git config user.email > /dev/null 2>&1 || {
  echo "git identity is not set on this machine; set one first:"
  echo "  git config --global user.name  'Your Name'"
  echo "  git config --global user.email 'you@example.edu'"
  exit 1
}

git fetch origin
git checkout -f -B pff-paper-conformance origin/nongaussian-costjo
git apply --check "$PATCH"
git apply "$PATCH"
git add src/oops/assimilation/PFF.h
git commit -m "PFF paper conformance: repulsion sign, moving particle geometry, learning-rate branch clobber

Three fixes that make PFF.h match Hu & van Leeuwen (2021) and standard
SVGD; audit and derivations in stable_DOEE
jedi_export/patches/README_PFF_AUDIT.md. The eda_3dvar_pff ctest
reference regeneration follows in the next commit (the old reference
pins the pre-fix trajectory)."

cat << 'EOT'

Branch pff-paper-conformance is ready. Remaining steps, in your jedi
build tree:

  1. rebuild oops (your bundle's usual build command)
  2. ctest -R oops_l95_eda_3dvar_pff       # EXPECTED TO FAIL vs the
                                           # old reference
  3. sanity-read the run log: norms decreasing under the eps schedule,
     no runaway, ensemble spread holding rather than collapsing
  4. find <build>/l95/test -name 'eda_3dvar_pff*test.out*'
     cp it over <oops>/l95/test/testoutput/eda_3dvar_pff.test
  5. ctest -R oops_l95_eda_3dvar_pff       # now green
  6. cd <oops>
     git add l95/test/testoutput/eda_3dvar_pff.test
     git commit -m "Regenerate eda_3dvar_pff reference for the conformant flow"
     git push -u origin pff-paper-conformance
EOT
