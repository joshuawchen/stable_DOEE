#!/usr/bin/env bash
# Create the pff-calibration-ctest branch: the correctness gate for the
# conformant PFF that the norm diagnostic cannot be (the repulsion term
# is pairwise-antisymmetric and cancels exactly in the summed-update
# norm, so the norm is blind to spread pathologies).
#
#   bash make_calibration_branch.sh [path-to-oops-checkout]
#
# Cut from origin/pff-paper-conformance (the test exercises the FIXED
# flow; on the unfixed trunk the spread gate would rightly fail). The
# chain: bgmean (ens-mean background via l95_ens_mean_variance.x) ->
# exact (DRPCG 3D-Var from the mean background = the exact posterior
# mean, linear-Gaussian) -> pff run (amplitude 0: all particles share
# ONE likelihood, the paper's PFF and the Phase 3 template) -> checker
# (ensemble mean vs exact, spread vs configured band).
#
# Everything derivable is DERIVED from committed files (member yamls
# from eda_3dvar_pff_N, the exact yaml from 3dvar.yaml) so the
# configurations stay in sync with their sources by construction.
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
git checkout -f -B pff-calibration-ctest origin/pff-paper-conformance

TI=l95/test/testinput
TO=l95/test/testoutput

# member yamls: amplitude 0, renamed outputs, derived from the pff EDA set
for n in 1 2 3 4; do
  sed -e "s/obs perturbations amplitude: 0.2/obs perturbations amplitude: 0.0/" \
      -e "s/mem00$n.eda_3dvar_pff/mem00$n.pff_calibration/" \
      -e "s/exp: eda_3dvar_pff.mem00$n/exp: pff_calibration.mem00$n/" \
      "$TI/eda_3dvar_pff_$n.yaml" > "$TI/pff_calibration_$n.yaml"
done

cat > "$TI/pff_calibration.yaml" << 'EOF'
files:
- testinput/pff_calibration_1.yaml
- testinput/pff_calibration_2.yaml
- testinput/pff_calibration_3.yaml
- testinput/pff_calibration_4.yaml

test:
  reference filename: testoutput/pff_calibration.test
  test output filename: testoutput/pff_calibration.test.out
EOF

cat > "$TI/pff_calibration_bgmean.yaml" << 'EOF'
ensemble:
  members from template:
    template:
      date: 2010-01-02T00:00:00Z
      filename: Data/forecast.ens.%mem%.2010-01-01T00:00:00Z.P1D.l95
    pattern: '%mem%'
    nmembers: 4

geometry:
  resol: 40

mean output:
  datadir: Data
  exp: pff_calibration.bgmean
  type: an
  date: 2010-01-02T00:00:00Z
EOF

# the exact companion: stock 3dvar with the mean background and its own
# reference; derived from 3dvar.yaml
python3 - << 'PYEOF'
import re
src = open("l95/test/testinput/3dvar.yaml").read()
src = src.replace(
    "filename: Data/forecast.fc.2010-01-01T00:00:00Z.P1D.l95",
    "filename: Data/pff_calibration.bgmean.an.2010-01-02T00:00:00Z.l95")
src = src.replace("Data/3dvar.2010-01-02T00:00:00Z.obt",
                  "Data/pff_calibration_exact.2010-01-02T00:00:00Z.obt")
src = src.replace("exp: 3dvar", "exp: pff_calibration_exact")
src = re.sub(r"test:\n  reference filename: [^\n]+\n",
             "test:\n"
             "  reference filename: testoutput/pff_calibration_exact.test\n"
             "  test output filename: "
             "testoutput/pff_calibration_exact.test.out\n", src)
open("l95/test/testinput/pff_calibration_exact.yaml", "w").write(src)
PYEOF

cat > "$TI/pff_calibration_check.yaml" << 'EOF'
geometry:
  resol: 40

member analyses:
- date: 2010-01-02T00:00:00Z
  filename: Data/pff_calibration.mem001.a.2010-01-02T00:00:00Z.l95
- date: 2010-01-02T00:00:00Z
  filename: Data/pff_calibration.mem002.a.2010-01-02T00:00:00Z.l95
- date: 2010-01-02T00:00:00Z
  filename: Data/pff_calibration.mem003.a.2010-01-02T00:00:00Z.l95
- date: 2010-01-02T00:00:00Z
  filename: Data/pff_calibration.mem004.a.2010-01-02T00:00:00Z.l95

exact analysis:
  date: 2010-01-02T00:00:00Z
  filename: Data/pff_calibration_exact.an.2010-01-02T00:00:00Z.l95

# rms(ensemble mean - exact posterior mean) must be under this; the
# flow's 21-step budget leaves a residual (norm plateau ~15% of the
# initial), so the bound is generous -- tighten after first green
mean tolerance: 0.15
# ensemble spread (rms member deviation from the mean): neither
# collapsed nor runaway; band refined at reference regeneration
spread min: 0.01
spread max: 0.80

test:
  reference filename: testoutput/pff_calibration_check.test
  test output filename: testoutput/pff_calibration_check.test.out
EOF

cat > l95/test/lorenz95/PFFCalibration.cc << 'EOF'
/*
 * (C) Copyright 2026 Colorado State University
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/// \file PFFCalibration.cc
///
/// Calibration gate for the conformant PFF (Hu & van Leeuwen 2021): with
/// unperturbed observations all particles share ONE likelihood, so in the
/// linear-Gaussian case the flow targets the posterior of N(mean background,
/// B) x likelihood -- whose mean is exactly the 3D-Var analysis started from
/// the mean background (pff_calibration_exact). Checks the ensemble MEAN
/// against that analysis and the ensemble SPREAD against a configured band.
/// This is the correctness gate the PFF norm diagnostic cannot be: the
/// repulsion term is pairwise-antisymmetric and cancels exactly in the
/// summed-update norm, so the norm is blind to spread pathologies.

#include <cmath>
#include <string>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "eckit/testing/Test.h"
#include "lorenz95/Resolution.h"
#include "lorenz95/StateL95.h"
#include "oops/runs/Run.h"
#include "oops/runs/Test.h"
#include "oops/util/Expect.h"
#include "oops/util/Logger.h"
#include "test/TestEnvironment.h"

namespace test {

CASE("test_pff_calibration") {
  const eckit::Configuration & conf = TestEnvironment::config();
  lorenz95::Resolution resol(eckit::LocalConfiguration(conf, "geometry"),
                             oops::mpi::myself());

  const auto memberConfs = conf.getSubConfigurations("member analyses");
  EXPECT(memberConfs.size() >= 2);
  std::vector<std::vector<double>> members;
  for (const auto & mc : memberConfs) {
    lorenz95::StateL95 xx(resol, mc);
    members.push_back(xx.getField().asVector());
  }
  lorenz95::StateL95 xexact(resol,
                            eckit::LocalConfiguration(conf, "exact analysis"));
  const std::vector<double> & exact = xexact.getField().asVector();

  const size_t nx = exact.size();
  const size_t nm = members.size();
  std::vector<double> mean(nx, 0.0);
  for (const auto & m : members) {
    EXPECT_EQUAL(m.size(), nx);
    for (size_t i = 0; i < nx; ++i) mean[i] += m[i];
  }
  for (size_t i = 0; i < nx; ++i) mean[i] /= static_cast<double>(nm);

  double meandev2 = 0.0;
  for (size_t i = 0; i < nx; ++i) {
    const double d = mean[i] - exact[i];
    meandev2 += d * d;
  }
  const double meanDev = std::sqrt(meandev2 / static_cast<double>(nx));

  double spread2 = 0.0;
  for (const auto & m : members)
    for (size_t i = 0; i < nx; ++i) {
      const double d = m[i] - mean[i];
      spread2 += d * d;
    }
  const double spread =
      std::sqrt(spread2 / static_cast<double>(nx * (nm - 1)));

  oops::Log::test() << "PFF calibration: ensemble mean rms deviation from the "
                    << "exact posterior mean: " << meanDev << std::endl;
  oops::Log::test() << "PFF calibration: ensemble spread: " << spread
                    << std::endl;

  EXPECT(meanDev <= conf.getDouble("mean tolerance"));
  EXPECT(spread >= conf.getDouble("spread min"));
  EXPECT(spread <= conf.getDouble("spread max"));
}

class PFFCalibration : public oops::Test {
 public:
  PFFCalibration() {}
 private:
  std::string testid() const override {return "test::PFFCalibration";}
  void register_tests() const override {}
  void clear() const override {}
};

}  // namespace test

int main(int argc, char **argv) {
  oops::Run run(argc, argv);
  test::PFFCalibration tests;
  return run.execute(tests);
}
EOF

touch "$TO/pff_calibration.test" "$TO/pff_calibration_exact.test" \
      "$TO/pff_calibration_check.test"

python3 - << 'PYEOF'
cm = open("l95/test/CMakeLists.txt").read()
a1 = "  testinput/eda_3dvar_pff.yaml\n"
assert a1 in cm
cm = cm.replace(a1, a1 + """  testinput/pff_calibration_1.yaml
  testinput/pff_calibration_2.yaml
  testinput/pff_calibration_3.yaml
  testinput/pff_calibration_4.yaml
  testinput/pff_calibration.yaml
  testinput/pff_calibration_bgmean.yaml
  testinput/pff_calibration_exact.yaml
  testinput/pff_calibration_check.yaml
""", 1)
a2 = "  testoutput/eda_3dvar_pff.test\n"
assert a2 in cm
cm = cm.replace(a2, a2 + """  testoutput/pff_calibration.test
  testoutput/pff_calibration_exact.test
  testoutput/pff_calibration_check.test
""", 1)
a3 = """ecbuild_add_test( TARGET oops_l95_eda_3dvar_pff
                  MPI 4
                  COMMAND l95_eda.x
                  ARGS testinput/eda_3dvar_pff.yaml
                  TEST_DEPENDS oops_l95_genenspert oops_l95_makeobs3d )
"""
assert a3 in cm
cm = cm.replace(a3, a3 + """
ecbuild_add_test( TARGET oops_l95_pff_calibration_bgmean
                  COMMAND l95_ens_mean_variance.x
                  ARGS testinput/pff_calibration_bgmean.yaml
                  TEST_DEPENDS oops_l95_genenspert )

ecbuild_add_test( TARGET oops_l95_pff_calibration_exact
                  COMMAND l95_4dvar.x
                  ARGS testinput/pff_calibration_exact.yaml
                  TEST_DEPENDS oops_l95_pff_calibration_bgmean oops_l95_makeobs3d )

ecbuild_add_test( TARGET oops_l95_pff_calibration
                  MPI 4
                  COMMAND l95_eda.x
                  ARGS testinput/pff_calibration.yaml
                  TEST_DEPENDS oops_l95_genenspert oops_l95_makeobs3d )

ecbuild_add_test( TARGET oops_l95_pff_calibration_check
                  SOURCES lorenz95/PFFCalibration.cc
                  ARGS testinput/pff_calibration_check.yaml
                  LIBS lorenz95
                  TEST_DEPENDS oops_l95_pff_calibration oops_l95_pff_calibration_exact )
""", 1)
open("l95/test/CMakeLists.txt", "w").write(cm)
print("cmake wired")
PYEOF

git add l95/test
git commit -m "PFF calibration gate: amplitude-0 flow vs the exact mean-background posterior

Chain: ens-mean background -> exact DRPCG 3D-Var -> shared-likelihood
PFF (amplitude 0, the paper's configuration and the loop template) ->
bespoke checker (ensemble mean vs exact posterior mean, spread band).
The correctness gate the summed-update norm cannot be (pairwise-
antisymmetric repulsion cancels in it). References empty: regenerate
on first run."

cat << 'EOT'

Branch pff-calibration-ctest ready. Build and regenerate:

  cd ~/jedi/src/build/oops
  make -j4 2>&1 | tail -5
  ctest -R "oops_l95_pff_calibration" --output-on-failure

Expected first pass: bgmean green; exact/pff FAIL vs empty references;
check runs and PRINTS meanDev and spread (the calibration verdict --
read those numbers) and fails vs its empty reference. Then:

  ls Data/pff_calibration*          # verify the analysis filenames
                                    # match pff_calibration_check.yaml;
                                    # adjust the yaml if the naming
                                    # convention differs
  cp l95/test/testoutput/pff_calibration_exact.test.out \
     ~/jedi/src/oops/l95/test/testoutput/pff_calibration_exact.test
  cp l95/test/testoutput/pff_calibration.test.out \
     ~/jedi/src/oops/l95/test/testoutput/pff_calibration.test
  cp l95/test/testoutput/pff_calibration_check.test.out \
     ~/jedi/src/oops/l95/test/testoutput/pff_calibration_check.test
  ctest -R "oops_l95_pff_calibration" --output-on-failure   # all green
  cd ~/jedi/src/oops
  git add l95/test/testoutput
  git commit -m "Pin pff calibration references"
  git push -u origin pff-calibration-ctest

Iteration risks (each a one-line yaml fix if hit): the mean-output
config keys of l95_ens_mean_variance.x; the member analysis filename
convention; paths relative to the build test dir.
EOT
