#!/usr/bin/env python3
"""Stage A end to end: the first test that pulls the whole chain at once.

    inject a known error density into l95 observations
    -> genenspert background ensemble, one first-guess pass per member
    -> collect member H(x), check ensemble reliability against the truth
    -> DOEE (third-difference regularised), compare against the injected density
    -> null floor at the measured configuration
    -> export to the `non gaussian cost` block, check(spec) as the C++ will
    -> run a Gaussian control and an evolving-Gaussian treatment 3D-Var
    -> read back EvolvingSigma and score both analyses against the noise-free
       observable under the injected density itself

Every link above exists and is tested separately; this is the first place they
are exercised as one chain. It stays inside Stage A: the ensemble never sees
the estimated density, so there is no feedback loop and any failure is a joint,
not the circularity.

Two modes.

REAL MODE needs an oops build (see RUNBOOK_ORBSTACK_JEDI.md) and runs the
model:

    python3 stage_a_end_to_end.py --build ~/jedi/src/build/oops \\
        --density heavy --members 20 --check-gaussian-diag

REAL MODE PREREQUISITES are run automatically when their artifacts are missing
(truth and forecast via l95_forecast.x, observations via l95_hofx.x, the
ensemble via l95_genpert.x). All testbed artifacts are prefixed `testbed_` so
nothing collides with ctest outputs.

SYNTHETIC MODE needs only numpy and quadprog and fabricates what the model
would have produced -- an exchangeable truth-and-members construction written
through the same .obt files -- so every Python step of the chain, including the
file plumbing and the YAML generation, runs anywhere:

    python3 stage_a_end_to_end.py --synthetic

The model runs and the read-back are then skipped, and the generated control
and treatment configurations are left in the work directory for inspection.

VERDICTS. Mechanical failures are fatal: a solver that does not run, a spec
that check() rejects, more than --max-nfixed slopes projected, EvolvingSigma
missing or not finite and positive, control and treatment outputs that are not
row-aligned. Scientific outcomes -- recovered width and kurtosis, floor
clearance, the treatment beating the control -- are printed with verdicts and
are fatal only under --strict, because a single l95 cycle at 40 gridpoints is
a small sample and the point of Stage A is to measure, not to assume. The
control-versus-treatment verdict uses the log score of each analysis's implied
predictive density at the NOISE-FREE observable -- the injected density as the
loss on analysis error, reported as regret in nats per observation below a
perfect analysis. This is an OSSE, so the metric can be chosen from the truth
rather than from either competitor's assumption; RMSE is the loss the Gaussian
control itself optimises, so it is reported alongside MAE but decides nothing.
Scoring against the noisy observations would be worse still: the analysis is
correlated with the draws it assimilated, and fits are not verification.

--check-gaussian-diag additionally runs the stock 3dvar_evolvinggaussian
configuration and asserts every EvolvingSigma value equals 0.4, closing the
"diagnostic content is still unchecked" item in PROCESS_NONGAUSSIAN_JO.md:
that configuration is exactly Gaussian, so any other value is a bug in the
flag, the group naming, the loop index or the missing-value handling.
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))              # collect_ensemble, inject_obs_error, ...
sys.path.insert(0, str(HERE.parent))       # stable_doee_reg, jedi_export

import collect_ensemble as CE              # noqa: E402
import inject_obs_error as IJ              # noqa: E402
import null_calibration as NC              # noqa: E402
import stable_doee_reg as R                # noqa: E402
from jedi_export import doee_to_yaml as DY  # noqa: E402

DATE = "2010-01-02T00:00:00Z"

# ---------------------------------------------------------------------------
# configuration templates (l95). Literal YAML braces are doubled for .format.
# ---------------------------------------------------------------------------

GENENSPERT_TPL = """\
members: {members}
geometry:
  resol: 40
model:
  name: L95
  f: 8.0
  tstep: PT1H30M
perturbed variables: [x]
background error:
  covariance model: L95Error
  date: 2010-01-01T00:00:00Z
  length_scale: 1.0
  standard_deviation: 0.6
forecast length: PT27H
initial condition:
  date: 2010-01-01T00:00:00Z
  filename: Data/forecast.an.2010-01-01T00:00:00Z.l95
output:
  datadir: Data
  date: 2010-01-01T00:00:00Z
  exp: testbed
  frequency: PT1H30M
  type: ens
"""

MEMBER_TPL = """\
# testbed member {k}: one outer loop only. The purpose of this run is the ombg
# column (y - H(x_b^k)) written at the first cost evaluation; the analysis is
# discarded.
cost function:
  cost type: 3D-Var
  time window:
    begin: 2010-01-01T21:00:00Z
    length: PT6H
  geometry:
    resol: 40
  analysis variables: [x]
  background:
    date: 2010-01-02T00:00:00Z
    filename: Data/testbed.ens.{k}.2010-01-01T00:00:00Z.P1D.l95
  background error:
    covariance model: L95Error
    date: 2010-01-02T00:00:00Z
    length_scale: 1.0
    standard_deviation: 0.6
  observations:
    observers:
    - obs operator: {{}}
      obs space:
        obsdatain:
          obsfile: Data/testbed_noisy.obt
        obsdataout:
          obsfile: Data/testbed_mem{k:03d}.obt
      obs error:
        covariance model: diagonal
variational:
  minimizer:
    algorithm: DRPCG
  iterations:
  - ninner: 1
    gradient norm reduction: 1e-10
    geometry:
      resol: 40
output:
  datadir: Data
  exp: testbed_mem{k:03d}
  times: ["2010-01-02T00:00:00Z"]
  type: an
"""

ANALYSIS_TPL = """\
# testbed {label}: {comment}
cost function:
  cost type: 3D-Var
  time window:
    begin: 2010-01-01T21:00:00Z
    length: PT6H
  geometry:
    resol: 40
  analysis variables: [x]
  background:
    date: 2010-01-02T00:00:00Z
    filename: Data/forecast.fc.2010-01-01T00:00:00Z.P1D.l95
  background error:
    covariance model: L95Error
    date: 2010-01-02T00:00:00Z
    length_scale: 1.0
    standard_deviation: 0.6
  observations:
{jo_type}    observers:
    - obs operator: {{}}
      obs space:
        obsdatain:
          obsfile: Data/testbed_noisy.obt
        obsdataout:
          obsfile: Data/testbed_{label}.obt
      obs error:
        covariance model: diagonal
{density}variational:
  minimizer:
    algorithm: DRPCG
  iterations:
  - ninner: 10
    gradient norm reduction: 1e-10
    geometry:
      resol: 40
  - ninner: 10
    gradient norm reduction: 1e-10
    geometry:
      resol: 40
output:
  datadir: Data
  exp: testbed_{label}
  times: ["{date}"]
  type: an
"""


def control_yaml():
    return ANALYSIS_TPL.format(
        label="control", date=DATE, jo_type="", density="",
        comment="Gaussian 3D-Var believing the ObsError column")


def treatment_yaml(spec):
    block = DY.to_yaml(spec, indent=6) + "\n"
    return ANALYSIS_TPL.format(
        label="treatment", date=DATE,
        jo_type="    jo type: evolving gaussian\n",
        density=block,
        comment="evolving-Gaussian 3D-Var using the estimated density")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

class Report:
    """Collects hard failures and soft verdicts, prints as it goes."""

    def __init__(self, strict=False):
        self.hard, self.soft, self.strict = [], [], False
        self.strict = strict

    def head(self, txt):
        print(f"\n=== {txt}")

    def info(self, txt):
        print(f"    {txt}")

    def fail(self, txt):
        print(f"    FAIL  {txt}")
        self.hard.append(txt)

    def verdict(self, ok, txt):
        print(f"    {'ok    ' if ok else 'WORSE ' }{txt}")
        if not ok:
            self.soft.append(txt)

    def finish(self):
        print("\n=== summary")
        for f in self.hard:
            print(f"    hard failure: {f}")
        for s in self.soft:
            print(f"    adverse outcome: {s}")
        if not self.hard and not self.soft:
            print("    every link held and every outcome came out the "
                  "right way")
        fatal = bool(self.hard) or (self.strict and bool(self.soft))
        print(f"    exit {'1 (FAIL)' if fatal else '0 (PASS)'}")
        return 1 if fatal else 0


def run_bin(rep, bindir, exe, cfg, cwd, logdir, tag):
    logdir.mkdir(exist_ok=True)
    log = logdir / f"{tag}.log"
    cmd = [str(bindir / exe), cfg]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=cwd, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = "".join(open(log).readlines()[-15:])
        rep.fail(f"{exe} {cfg} exited {rc}; last lines:\n{tail}")
    else:
        rep.info(f"{exe} {cfg}  ->  ok  (log {log.name})")
    return rc == 0


def moments_on(grid, pi):
    dx = grid[1] - grid[0]
    p = np.maximum(np.asarray(pi, float), 0.0)
    tot = p.sum() * dx
    if tot <= 0:
        return np.nan, np.nan
    p = p / tot
    mu = (p * grid).sum() * dx
    sd = np.sqrt((p * (grid - mu) ** 2).sum() * dx)
    ku = (p * (grid - mu) ** 4).sum() * dx / sd ** 4 - 3.0
    return float(sd), float(ku)


def analytic_density(spec, grid):
    """The injected density, evaluated on `grid`. For mirrored_gamma the
    rescaling uses the population standard deviation (2*sqrt(2)) where the
    injection used the sample one, so the comparison is approximate there."""
    g = lambda s: np.exp(-grid ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
    k = spec["kind"]
    if k == "gaussian":
        return g(spec["sigma"])
    if k == "mixture":
        return spec["w"] * g(spec["sigma1"]) \
            + (1 - spec["w"]) * g(spec["sigma2"])
    if k == "laplace":
        b = spec["b"]
        return np.exp(-np.abs(grid) / b) / (2 * b)
    if k == "mirrored_gamma":
        c = spec["rescaled_to_sigma"] / (2.0 * np.sqrt(2.0))
        t = 2.0 - grid / c
        f = np.where(t >= 0, t * np.exp(-t / 2.0) / 4.0, 0.0)
        return f / c
    raise ValueError(f"unknown injected density kind '{k}'")


def true_log_density(spec, x):
    """log f of the injected density at x. The same (for mirrored_gamma,
    population-scale approximate) function scores every run, so comparisons
    stay paired; the floor keeps a support violation finite and enormous
    rather than -inf."""
    f = analytic_density(spec, np.asarray(x, float))
    return np.log(np.maximum(f, 1e-300))


def read_columns(path, prefix=None):
    rec = CE.read_obt(str(path))
    if prefix is None:
        return rec
    return {k: v for k, v in rec.items() if k.startswith(prefix)}


# ---------------------------------------------------------------------------
# synthetic substitutes for the model runs
# ---------------------------------------------------------------------------

def synthesize(work, n_obs, K, sigma_b, field_amp, seed):
    """Fabricate what truth + genenspert + member first-guess runs would leave
    on disk, with truth EXCHANGEABLE with the members (the Stage A
    construction), writing through the same .obt files the real runs use."""
    rng = np.random.default_rng(seed)
    x = np.arange(n_obs)
    field = field_amp * (np.sin(x / 37.0) + 0.5 * np.cos(x / 11.0))
    truth = field + rng.normal(0.0, sigma_b, n_obs)
    members = field[:, None] + rng.normal(0.0, sigma_b, (n_obs, K))
    rows = [[str(j), DATE, repr(float(j % 40)), repr(float(truth[j])),
             repr(0.4)] for j in range(n_obs)]
    IJ.write_obt(str(work / "testbed_truth3d.obt"),
                 ["ObsValue", "ObsError"], rows)
    return members


def write_synthetic_members(work, members, assumed_error):
    names, rows, _ = IJ.read_obt(str(work / "testbed_noisy.obt"))
    iv = 3 + names.index("ObsValue")
    obs = np.array([float(r[iv]) for r in rows])
    K = members.shape[1]
    for k in range(K):
        ombg = obs - members[:, k]
        mrows = [[r[0], r[1], r[2], r[iv], repr(float(assumed_error)),
                  repr(float(ombg[j]))] for j, r in enumerate(rows)]
        IJ.write_obt(str(work / f"testbed_mem{k:03d}.obt"),
                     ["ObsValue", "ObsError", "ombg"], mrows)


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=None,
                    help="oops build directory (real mode)")
    ap.add_argument("--synthetic", action="store_true",
                    help="fabricate the model outputs; no build needed")
    ap.add_argument("--work", type=Path, default=None,
                    help="work directory for synthetic mode "
                         "(default ./testbed_work)")
    ap.add_argument("--density", default="heavy",
                    choices=["gaussian", "heavy", "laplace", "mirrored_gamma"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--assumed-error", type=float, default=0.4)
    ap.add_argument("--n-obs", type=int, default=2000,
                    help="synthetic mode only; real mode takes what "
                         "makeobs3d generated")
    ap.add_argument("--sigma-b", type=float, default=0.5,
                    help="synthetic mode only")
    ap.add_argument("--floor-trials", type=int, default=6)
    ap.add_argument("--max-nfixed", type=int, default=5)
    ap.add_argument("--reuse-ensemble", action="store_true",
                    help="real mode: skip genenspert and the member runs if "
                         "their outputs already exist")
    ap.add_argument("--check-gaussian-diag", action="store_true",
                    help="real mode: also run the stock Gaussian-equivalence "
                         "configuration and assert EvolvingSigma == 0.4")
    ap.add_argument("--strict", action="store_true",
                    help="adverse scientific outcomes are fatal too")
    a = ap.parse_args()

    if bool(a.build) == bool(a.synthetic):
        ap.error("exactly one of --build or --synthetic is required")

    rep = Report(strict=a.strict)
    real = a.build is not None

    if real:
        bindir = (a.build / "bin").resolve()
        testdir = (a.build / "l95" / "test").resolve()
        data = testdir / "Data"
        cfgdir = testdir / "testinput"
        logdir = testdir / "testbed_logs"
        for p, what in ((bindir / "l95_4dvar.x", "l95 executables"),
                        (cfgdir, "l95 test inputs"),
                        (data, "l95 test Data directory")):
            if not p.exists():
                print(f"{p} not found ({what}); is --build an oops build "
                      "directory with the l95 model compiled?")
                return 2
    else:
        data = (a.work or Path("testbed_work")).resolve()
        data.mkdir(parents=True, exist_ok=True)

    # ---- 1. base artifacts -------------------------------------------------
    rep.head("base artifacts (truth, forecast, observations)")
    if real:
        prereqs = [
            ("truth.fc.2010-01-01T00:00:00Z.PT21H.l95",
             "l95_forecast.x", "testinput/truth.yaml", "truth"),
            ("forecast.fc.2010-01-01T00:00:00Z.P1D.l95",
             "l95_forecast.x", "testinput/forecast.yaml", "forecast"),
            ("truth3d.2010-01-02T00:00:00Z.obt",
             "l95_hofx.x", "testinput/makeobs3d.yaml", "makeobs3d"),
        ]
        for artifact, exe, cfg, tag in prereqs:
            if (data / artifact).exists():
                rep.info(f"{artifact} present")
            elif not run_bin(rep, bindir, exe, cfg, testdir, logdir, tag):
                return rep.finish()
        truth_obt = data / "truth3d.2010-01-02T00:00:00Z.obt"
        members = None
    else:
        members = synthesize(data, a.n_obs, a.members, a.sigma_b,
                             field_amp=4.0, seed=a.seed + 1)
        rep.info(f"synthetic truth and {a.members} members constructed "
                 f"(n={a.n_obs}, sigma_b={a.sigma_b})")
        truth_obt = data / "testbed_truth3d.obt"

    # ---- 2. inject the known density --------------------------------------
    rep.head(f"inject '{a.density}' (seed {a.seed}, assumed error "
             f"{a.assumed_error})")
    npz = data / "testbed_truth.npz"
    spec_inj = IJ.inject(str(truth_obt), str(data / "testbed_noisy.obt"),
                         density=a.density, scale=a.scale, seed=a.seed,
                         assumed_error=a.assumed_error, truth_path=str(npz))
    rep.info(f"n={spec_inj['n']}  sample sigma {spec_inj['sample_sigma']:.3f}"
             f"  sample exkurt {spec_inj['sample_excess_kurtosis']:+.2f}")
    saved = np.load(npz)
    truth_obs = saved["truth_obs"]
    inj = json.loads(str(saved["spec"]))

    # ---- 3. ensemble and member first-guess passes -------------------------
    rep.head(f"ensemble of {a.members} and one first-guess pass per member")
    if real:
        ens_file = data / f"testbed.ens.{a.members}.2010-01-01T00:00:00Z.P1D.l95"
        if a.reuse_ensemble and ens_file.exists():
            rep.info("reusing the existing testbed ensemble")
        else:
            (cfgdir / "testbed_genenspert.yaml").write_text(
                GENENSPERT_TPL.format(members=a.members))
            if not run_bin(rep, bindir, "l95_genpert.x",
                           "testinput/testbed_genenspert.yaml",
                           testdir, logdir, "genenspert"):
                return rep.finish()
        for k in range(1, a.members + 1):
            out = data / f"testbed_mem{k:03d}.obt"
            if a.reuse_ensemble and out.exists():
                continue
            (cfgdir / f"testbed_mem{k:03d}.yaml").write_text(
                MEMBER_TPL.format(k=k))
            if not run_bin(rep, bindir, "l95_4dvar.x",
                           f"testinput/testbed_mem{k:03d}.yaml",
                           testdir, logdir, f"mem{k:03d}"):
                return rep.finish()
    else:
        write_synthetic_members(data, members, a.assumed_error)
        rep.info("member .obt files written through the real file format")

    # ---- 4. collect and check the ensemble ---------------------------------
    rep.head("collect member H(x) and check reliability against the truth")
    pattern = str(data / "testbed_mem*.obt")
    obs, hofx, meta = CE.collect(pattern)
    rep.info(f"{meta['members']} members, {meta['n_obs']} observations, "
             f"mean background spread {meta['mean_background_spread']:.3f}")
    rel = CE.reliability(pattern, truth_obs)
    rep.info(f"reliability ratio {rel['ratio']:.2f}  ({rel['note']})")

    # ---- 5. estimate the density -------------------------------------------
    rep.head("DOEE, third-difference regularised, lambda by cross-validation")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid, f_d, f_k, innov = R.histograms_from_ensemble(obs, hofx,
                                                           seed=a.seed + 2)
        groups = R.innovation_groups(len(obs), hofx.shape[1])
        xg, pi, cache = R.estimate_from_histograms(grid, f_d, f_k,
                                                   innov=innov, groups=groups)
    sd, ku = moments_on(xg, pi)
    fine = np.arange(-10.0, 10.0001, 0.02)
    tru = analytic_density(inj, fine)
    p_i = np.interp(fine, xg, pi, left=0.0, right=0.0)
    tot = p_i.sum() * (fine[1] - fine[0])
    l1 = float(np.abs(p_i / tot - tru).sum() * (fine[1] - fine[0])) \
        if tot > 0 else np.nan
    rep.info(f"lambda {cache['lambda']:.0e}  resolvability "
             f"{cache['resolvability']:.2f}")
    rep.info(f"recovered sigma {sd:.3f} (injected sample "
             f"{spec_inj['sample_sigma']:.3f})  exkurt {ku:+.2f} (injected "
             f"sample {spec_inj['sample_excess_kurtosis']:+.2f})  L1 {l1:.3f}")
    rep.verdict(abs(sd / spec_inj['sample_sigma'] - 1.0) <= 0.25,
                f"recovered width within 25% of the injected one "
                f"({100 * (sd / spec_inj['sample_sigma'] - 1):+.1f}%)")

    # ---- 6. the null floor at this configuration ---------------------------
    rep.head(f"null floor at the measured configuration "
             f"({a.floor_trials} trials)")
    sigma_b_est = float(np.sqrt(np.mean(np.var(hofx, axis=1, ddof=1))))
    sigma_o_est = float(cache["resolvability"] * sigma_b_est)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        floor = NC.null_floor(sigma_o_est, sigma_b_est, len(obs),
                              hofx.shape[1], trials=a.floor_trials)
    rep.info(f"sigma_o~{sigma_o_est:.3f} sigma_b~{sigma_b_est:.3f}  floor "
             f"mean {floor['kurtosis_mean']:+.2f}  p95 "
             f"{floor['kurtosis_p95']:+.2f}")
    rep.info("verdict on the recovered tail: " + NC.verdict(ku, floor))
    if inj["kind"] != "gaussian":
        rep.verdict(ku > floor["kurtosis_p95"],
                    "a non-Gaussian injection should clear the floor")
    else:
        rep.verdict(ku <= floor["kurtosis_p95"],
                    "a Gaussian injection should sit under the floor")

    # ---- 7. export ---------------------------------------------------------
    rep.head("export to the `non gaussian cost` block")
    clamped = 0
    for side in ("left_dd", "right_dd"):
        if cache[side] > 0:
            cache[side] = -1e-12
            clamped += 1
    if clamped:
        rep.info(f"{clamped} tail curvature(s) clamped to -1e-12 to keep the "
                 "tails integrable (finalize_pdf_cache convention)")
    spec, nfixed = DY.to_spec(cache, save_sigma=True)
    rep.info(f"mode {spec['mode']:+.3f}  sigma at mode "
             f"{spec['sigma at mode']:.3f}  nfixed {nfixed}")
    if nfixed > a.max_nfixed:
        rep.fail(f"nfixed={nfixed} exceeds --max-nfixed={a.max_nfixed}: the "
                 "estimate is too noisy to assimilate")
    bad = DY.check(spec)
    if bad:
        rep.fail(f"check(spec) rejected {len(bad)} departure(s), first "
                 f"{bad[0]}")
    else:
        rep.info("check(spec): effective variance finite and positive "
                 "everywhere the C++ will look")
    block_path = (data if not real else cfgdir) / "testbed_density_block.yaml"
    block_path.write_text(DY.to_yaml(spec, indent=6) + "\n")
    rep.info(f"density block written to {block_path}")

    # ---- 8. control and treatment configurations ---------------------------
    rep.head("control and treatment configurations")
    cdir = cfgdir if real else data
    (cdir / "testbed_control.yaml").write_text(control_yaml())
    (cdir / "testbed_treatment.yaml").write_text(treatment_yaml(spec))
    rep.info(f"written to {cdir}/testbed_control.yaml and "
             "testbed_treatment.yaml")
    if not real:
        rep.info("synthetic mode: stopping before the model runs; point "
                 "--build at an oops build to take the chain the rest of "
                 "the way")
        return rep.finish()

    # ---- 9. run both -------------------------------------------------------
    rep.head("run the control and the treatment")
    okc = run_bin(rep, bindir, "l95_4dvar.x", "testinput/testbed_control.yaml",
                  testdir, logdir, "control")
    okt = run_bin(rep, bindir, "l95_4dvar.x",
                  "testinput/testbed_treatment.yaml", testdir, logdir,
                  "treatment")
    if not (okc and okt):
        return rep.finish()

    # ---- 10. read back -----------------------------------------------------
    rep.head("read back the diagnostic and the analyses")
    tre = CE.read_obt(str(data / "testbed_treatment.obt"))
    sig_cols = sorted(k for k in tre if k.startswith("EvolvingSigma"))
    if not sig_cols:
        rep.fail("no EvolvingSigma columns in the treatment output; the "
                 "'save evolving sigma' path did not write")
    for c in sig_cols:
        v = tre[c]
        if not (np.all(np.isfinite(v)) and np.all(v > 0)):
            rep.fail(f"{c} contains non-finite or non-positive values")
        else:
            rep.info(f"{c}: mean {v.mean():.3f}  min {v.min():.3f}  max "
                     f"{v.max():.3f}  std {v.std():.3f}")
    if sig_cols:
        spread = max(float(tre[c].std()) for c in sig_cols)
        rep.verdict(spread > 1e-6,
                    "sigma should vary with the departure for a genuinely "
                    "non-Gaussian density")

    ctl = CE.read_obt(str(data / "testbed_control.obt"))
    # The verdict metric is chosen from the TRUTH, not from either
    # competitor's assumption. RMSE is the loss the Gaussian control itself
    # optimises, so deciding by RMSE hands the control home advantage; this
    # is an OSSE, the injected density is known exactly, so the analysis is
    # scored by the log score of its implied predictive density at the
    # NOISE-FREE observable: mean log f_true(H(truth) - H(x_a)). Scoring
    # against the noisy observations instead would reward fitting the very
    # draws that were assimilated. Every injected density has its mode at
    # zero, so the score is reported as a regret in nats per observation
    # below a perfect analysis, and RMSE and MAE are printed for continuity
    # without deciding anything.
    eps = saved["errors"]
    ll0 = float(true_log_density(inj, np.zeros(1))[0])
    out = {}
    for name, rec in (("control", ctl), ("treatment", tre)):
        if "oman" not in rec:
            rep.fail(f"{name} output has no oman column")
            continue
        if len(rec["ObsValue"]) != len(eps):
            rep.fail(f"{name} output has {len(rec['ObsValue'])} rows but "
                     f"{len(eps)} errors were injected; outputs are not "
                     "aligned with the injection")
            continue
        # oman = y - H(x_a) and y = H(truth) + eps, so H(truth) - H(x_a)
        # = oman - eps, and likewise for the background through ombg.
        d_an = rec["oman"] - eps
        d_bg = rec["ombg"] - eps
        out[name] = {
            "regret_an": ll0 - float(np.mean(true_log_density(inj, d_an))),
            "regret_bg": ll0 - float(np.mean(true_log_density(inj, d_bg))),
            "rmse_an": float(np.sqrt(np.mean(d_an ** 2))),
            "mae_an": float(np.mean(np.abs(d_an))),
        }
        rep.info(f"{name}: regret {out[name]['regret_an']:.4f} nats/ob "
                 f"(background {out[name]['regret_bg']:.4f})  rmse "
                 f"{out[name]['rmse_an']:.4f}  mae {out[name]['mae_an']:.4f}")
    if len(out) == 2:
        # both runs share the background and the observations, so the
        # background regrets must agree exactly; a difference means the two
        # output files are not row-aligned and every comparison is void
        mis = abs(out["control"]["regret_bg"] - out["treatment"]["regret_bg"])
        if mis > 1e-9:
            rep.fail(f"background regrets differ by {mis:.2e}; the control "
                     "and treatment outputs are not aligned")
        rep.verdict(out["treatment"]["regret_an"] <= out["control"]["regret_an"],
                    "the treatment analysis should beat the Gaussian control "
                    "under the true-density log score")

    # ---- 11. optional: the Gaussian-equivalence diagnostic check -----------
    if a.check_gaussian_diag:
        rep.head("stock Gaussian-equivalence run: EvolvingSigma must be 0.4")
        if run_bin(rep, bindir, "l95_4dvar.x",
                   "testinput/3dvar_evolvinggaussian.yaml", testdir, logdir,
                   "gaussian_equiv"):
            g = CE.read_obt(str(
                data / "3dvar_evolvinggaussian.2010-01-02T00:00:00Z.obt"))
            cols = sorted(k for k in g if k.startswith("EvolvingSigma"))
            if not cols:
                rep.fail("no EvolvingSigma columns in the Gaussian-"
                         "equivalence output")
            for c in cols:
                dev = float(np.max(np.abs(g[c] - 0.4)))
                if dev > 1e-6:
                    rep.fail(f"{c} deviates from 0.4 by up to {dev:.2e}; the "
                             "diagnostic content is wrong")
                else:
                    rep.info(f"{c}: every value is 0.4 (max deviation "
                             f"{dev:.1e})")

    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
