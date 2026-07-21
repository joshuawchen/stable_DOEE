#!/usr/bin/env python3
"""Stage A end to end: the first test that pulls the whole chain at once.

    inject a known error density into l95 observations
    -> genenspert background ensemble, one first-guess pass per member
    -> collect member H(x), check ensemble reliability against the truth
    -> DOEE (third-difference regularized), compare against the injected density
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
control itself optimizes, so it is reported alongside MAE but decides nothing.
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
  standard_deviation: {pert_sd}
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

MAKEOBS_TPL = """\
# testbed observation generation: H(truth) with no noise, at a configurable
# density. l95 interpolates the observation operator, so obs_density may
# exceed the 40 gridpoints. The ObsError column carries the assumed error.
geometry:
  resol: 40
model:
  f: 8.0
  name: L95
  tstep: PT1H30M
initial condition:
  date: 2010-01-01T21:00:00Z
  filename: Data/truth.fc.2010-01-01T00:00:00Z.PT21H.l95
forecast length: PT6H

time window:
  begin: 2010-01-01T21:00:00Z
  length: PT4H30M
observations:
  observers:
  - obs operator: {{}}
    obs space:
      generate:
        obs_density: {obs_density}
        obs_error: {obs_error}
        obs_frequency: PT1H30M
      obsdataout:
        obsfile: Data/testbed_truth3d.obt
make obs: true
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


def save_density_plot(path, obs, hofx, xg, pi, inj, seed, adaptive,
                      cache=None):
    """Three-panel picture of the run: the recovered density against the
    injected truth (linear and log -- the log panel is where tail lobes
    and satellite maxima are visible), and the innovation space showing
    the raw histogram next to the fitted and true innovation densities.
    Written on every run, including gate failures, which is when it is
    most wanted. Returns the path, or None when matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    nb = None
    if adaptive:
        _, _, _, iv = R.histograms_from_ensemble(obs, hofx, seed=seed)
        nb = R.adaptive_bin_count(iv)
    grid, f_d, f_k, _ = R.histograms_from_ensemble(obs, hofx, seed=seed,
                                                   n_bins=nb)
    dx = grid[1] - grid[0]
    p = np.interp(grid, xg, pi, left=0.0, right=0.0)
    tot = p.sum() * dx
    if tot > 0:
        p = p / tot
    tru = analytic_density(inj, grid)
    Eta = R._conv_matrix(np.asarray(f_k, float), grid.size, dx, grid[0])
    fit_in = dx * (Eta @ p)
    tru_in = dx * (Eta @ tru)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(grid, tru, "k-", lw=1.8, label="injected truth")
    ax[0].plot(grid, p, "-", color="tab:red", lw=1.4, label="estimate")
    ax[0].set_title("noise density")
    ax[0].set_xlabel("obs error")
    ax[0].legend(frameon=False)
    ax[1].semilogy(grid, np.maximum(tru, 1e-7), "k-", lw=1.8)
    ax[1].semilogy(grid, np.maximum(p, 1e-7), "-", color="tab:red", lw=1.4)
    if cache is not None:
        for edge, slope, dd, sgn in (
                (cache["stable_min"], cache["left_log_slope"],
                 cache["left_dd"], -1.0),
                (cache["stable_max"], cache["right_log_slope"],
                 cache["right_dd"], +1.0)):
            j = int(np.argmin(np.abs(grid - edge)))
            xt = grid[(grid - edge) * sgn >= 0]
            if xt.size > 1 and p[j] > 0:
                lt = np.log(p[j]) + slope * (xt - edge) \
                    + 0.5 * dd * (xt - edge) ** 2
                ax[1].semilogy(xt, np.maximum(np.exp(lt), 1e-7), "--",
                               color="tab:blue", lw=1.2,
                               label="exported tail" if sgn > 0 else None)
        ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_ylim(1e-6, None)
    ax[1].set_title("noise density, log scale (tails and satellites)")
    ax[1].set_xlabel("obs error")
    ax[2].fill_between(grid, f_d, step="mid", alpha=0.35,
                       color="tab:gray", label="innovation histogram")
    ax[2].plot(grid, tru_in / dx, "k-", lw=1.8, label="truth * kernel")
    ax[2].plot(grid, fit_in / dx, "-", color="tab:red", lw=1.4,
               label="estimate * kernel (what the fit matched)")
    ax[2].set_title("innovation space")
    ax[2].set_xlabel("innovation  y - H(x)")
    ax[2].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.set_xlim(grid[0], grid[-1])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def gaussian_tails(cache, xg, pi, sd, rep, mass_frac=0.92,
                   sigma_widest=3.0):
    """Replace the two-bin tail curvatures with windowed fits and enforce
    at-least-Gaussian decay in the export.

    The cache's left_dd/right_dd are finite differences over the two
    outermost trimmed bins -- maximally noisy -- and the old policy
    clamped any wrong sign to -1e-12, so the assimilated density had
    EXPONENTIAL tails by convention whatever the truth did. Here the
    log-density curvature is fit per side over the outer (1 - mass_frac)
    of probability mass, and the exported curvature is
        dd = clip(dd_fit, -1/(0.3 sd)^2, -1/(sigma_widest * sd)^2)
    so beyond the data every tail closes at least as fast as a Gaussian
    of sigma_widest recovered sigmas (and no faster than 0.3 of one,
    against spiky edge fits). Tails the data measure as thinner keep
    their measured curvature; this is export policy, not estimation --
    the recovered density itself is untouched."""
    lo, hi = cache["stable_min"], cache["stable_max"]
    sel = (xg >= lo) & (xg <= hi) & (pi > 0)
    x_in, p_in = xg[sel], pi[sel]
    dxg = xg[1] - xg[0]
    c = np.cumsum(p_in) * dxg
    c = c / max(c[-1], 1e-300)
    dd_cap = -1.0 / (sigma_widest * max(sd, 1e-6)) ** 2
    dd_flr = -1.0 / (0.3 * max(sd, 1e-6)) ** 2
    for side, mask in (("left_dd", c <= 1.0 - mass_frac),
                       ("right_dd", c >= mass_frac)):
        xs, ps = x_in[mask], p_in[mask]
        if xs.size < 6:
            k = min(8, x_in.size)
            xs, ps = (x_in[:k], p_in[:k]) if side == "left_dd"                 else (x_in[-k:], p_in[-k:])
        dd_fit = np.nan
        if xs.size >= 3:
            try:
                dd_fit = 2.0 * float(np.polyfit(xs, np.log(ps), 2)[0])
            except Exception:
                pass
        dd = dd_fit if np.isfinite(dd_fit) else dd_cap
        dd = float(np.clip(dd, dd_flr, dd_cap))
        sig_t = float(np.sqrt(-1.0 / dd))
        rep.info(f"{side}: fitted log-curvature "
                 f"{dd_fit if np.isfinite(dd_fit) else float('nan'):+.3f} "
                 f"over {xs.size} outer bins -> exported {dd:+.3f} "
                 f"(tail sigma {sig_t:.2f}; widest allowed "
                 f"{sigma_widest:.0f} x recovered sd {sd:.2f})")
        cache[side] = dd


def oracle_cache(spec_inj):
    """A stable_DOEE-style cache built from the injected density itself
    (grid-point convention: stable min/max are the first and last point).
    Regions where the density vanishes (mirrored_gamma support edge) are
    trimmed to the contiguous run around the mode."""
    s = spec_inj["sample_sigma"]
    dx = s / 25.0
    half = 6.0 * s
    n = 2 * int(round(half / dx)) + 1
    xg = (np.arange(n) - (n - 1) / 2) * dx
    f = analytic_density(spec_inj, xg)
    keep = f > 1e-12
    padded = np.r_[0, keep.astype(int), 0]
    d = np.diff(padded)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    k = int(np.argmax([f[s0:e0].sum() for s0, e0 in zip(starts, ends)]))
    xg, f = xg[starts[k]:ends[k]], f[starts[k]:ends[k]]
    logf = np.log(f)
    slopes = np.empty_like(logf)
    slopes[0] = (logf[1] - logf[0]) / dx
    slopes[-1] = (logf[-1] - logf[-2]) / dx
    slopes[1:-1] = (logf[2:] - logf[:-2]) / (2 * dx)
    return {"dx": dx, "stable_min": float(xg[0]), "stable_max": float(xg[-1]),
            "slopes_log": slopes,
            "left_log_slope": float(slopes[0]),
            "left_dd": float((slopes[1] - slopes[0]) / dx),
            "right_log_slope": float(slopes[-1]),
            "right_dd": float((slopes[-1] - slopes[-2]) / dx),
            "lambda": float("nan"), "resolvability": float("inf"),
            "kept_mass": 1.0}


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
    ap.add_argument("--pert-sd", type=float, default=0.6,
                    help="real mode: genenspert initial perturbation standard "
                         "deviation. The initial spread grows through 27h of "
                         "l95 dynamics, so what matters is the reliability "
                         "ratio the run reports; at the stock 0.6 the first "
                         "measurement gave 2.27 (over-dispersive by more than "
                         "a factor of two)")
    ap.add_argument("--assumed-error", type=float, default=0.4)
    ap.add_argument("--n-obs", type=int, default=2000,
                    help="synthetic mode only; real mode is set by "
                         "--obs-density")
    ap.add_argument("--obs-density", type=int, default=40,
                    help="real mode: observation locations per observation "
                         "time (3 times in the window, so n_obs = 3x this). "
                         "l95 interpolates, so this may exceed the 40 "
                         "gridpoints; raise it together with --members to "
                         "scale the innovation sample")
    ap.add_argument("--sigma-b", type=float, default=0.5,
                    help="synthetic mode only")
    ap.add_argument("--floor-trials", type=int, default=6)
    ap.add_argument("--max-nfixed", type=int, default=5,
                    help="export gate, in PERCENT of probability mass the "
                         "unimodality projection may touch (the raw bin "
                         "count is reported but does not gate: at large "
                         "sample sizes many near-empty tail bins get "
                         "cosmetic sign repairs)")
    ap.add_argument("--reuse-ensemble", action="store_true",
                    help="real mode: skip genenspert and the member runs if "
                         "their outputs already exist")
    ap.add_argument("--lam", type=float, default=None,
                    help="force the smoothing strength, skipping selection; "
                         "the null floor is measured at the same value. On "
                         "the first record ensemble the flat criterion plus "
                         "the calibrated default under-smoothed at 347 bins "
                         "(sigma at mode 0.24 vs 0.50); lam 30 restored it")
    ap.add_argument("--adaptive", action="store_true",
                    help="estimate with estimate_adaptive (cross-split "
                         "one-SE smoothing selection); refuses --lam "
                         "because there is no knob to set, which is the "
                         "point")
    ap.add_argument("--oracle-density", action="store_true",
                    help="export the injected density instead of the "
                         "estimate; separates estimation quality from the "
                         "cost-function method (the estimate is still "
                         "computed and reported)")
    ap.add_argument("--check-gaussian-diag", action="store_true",
                    help="real mode: also run the stock Gaussian-equivalence "
                         "configuration and assert EvolvingSigma == 0.4")
    ap.add_argument("--strict", action="store_true",
                    help="adverse scientific outcomes are fatal too")
    a = ap.parse_args()

    if bool(a.build) == bool(a.synthetic):
        ap.error("exactly one of --build or --synthetic is required")
    if a.adaptive and a.lam is not None:
        ap.error("--adaptive selects its own smoothing; --lam has no "
                 "meaning under it")

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
        # the testbed generates its own truth observations so the density is
        # a knob; the stock truth3d stays untouched for the ctest suite and
        # for --check-gaussian-diag
        (cfgdir / "testbed_makeobs.yaml").write_text(
            MAKEOBS_TPL.format(obs_density=a.obs_density,
                               obs_error=a.assumed_error))
        if not run_bin(rep, bindir, "l95_hofx.x",
                       "testinput/testbed_makeobs.yaml", testdir, logdir,
                       "makeobs"):
            return rep.finish()
        truth_obt = data / "testbed_truth3d.obt"
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
                GENENSPERT_TPL.format(members=a.members, pert_sd=a.pert_sd))
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
    # exactly this run's members: a wildcard also sweeps in stale files from
    # earlier experiments at other sizes, and collect() rightly refuses the
    # mixture. Real-mode members are numbered from 1, synthetic from 0.
    base = 1 if real else 0
    pattern = [str(data / f"testbed_mem{k:03d}.obt")
               for k in range(base, base + a.members)]
    obs, hofx, meta = CE.collect(pattern)
    rep.info(f"{meta['members']} members, {meta['n_obs']} observations, "
             f"mean background spread {meta['mean_background_spread']:.3f}")
    rel = CE.reliability(pattern, truth_obs)
    rep.info(f"reliability ratio {rel['ratio']:.2f}  ({rel['note']})")
    rep.info(f"ensemble-mean bias {rel['bias_of_mean']:+.3f}: the "
             "member-difference kernel cancels any mean bias, so the "
             "recovered density's location is identified only up to this "
             "number and its mode should be read net of it")

    # ---- 5. estimate the density -------------------------------------------
    if a.adaptive:
        rep.head("DOEE, adaptive: whitened data term, cross-split one-SE mu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xg, pi, cache = R.estimate_adaptive_from_ensemble(
                obs, hofx, seed=a.seed + 2)
    else:
        rep.head("DOEE, third-difference regularized, lambda by "
                 "cross-validation")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid, f_d, f_k, innov = R.histograms_from_ensemble(
                obs, hofx, seed=a.seed + 2)
            groups = R.innovation_groups(len(obs), hofx.shape[1])
            xg, pi, cache = R.estimate_from_histograms(
                grid, f_d, f_k, lam=a.lam, innov=innov, groups=groups)
    sd, ku = moments_on(xg, pi)
    fine = np.arange(-10.0, 10.0001, 0.02)
    tru = analytic_density(inj, fine)
    p_i = np.interp(fine, xg, pi, left=0.0, right=0.0)
    tot = p_i.sum() * (fine[1] - fine[0])
    l1 = float(np.abs(p_i / tot - tru).sum() * (fine[1] - fine[0])) \
        if tot > 0 else np.nan
    if a.adaptive:
        rep.info(f"mu {cache['lambda']:.1e} (cross-split argmin "
                 f"{cache['mu_argmin']:.1e})  resolvability "
                 f"{cache['resolvability']:.2f}  kept interior mass "
                 f"{100 * cache['kept_mass']:.1f}%")
    else:
        rep.info(f"lambda {cache['lambda']:.0e}  resolvability "
                 f"{cache['resolvability']:.2f}  kept interior mass "
                 f"{100 * cache['kept_mass']:.1f}%")
    rep.info(f"recovered sigma {sd:.3f} (injected sample "
             f"{spec_inj['sample_sigma']:.3f})  exkurt {ku:+.2f} (injected "
             f"sample {spec_inj['sample_excess_kurtosis']:+.2f})  L1 {l1:.3f}")
    rep.verdict(abs(sd / spec_inj['sample_sigma'] - 1.0) <= 0.25,
                f"recovered width within 25% of the injected one "
                f"({100 * (sd / spec_inj['sample_sigma'] - 1):+.1f}%)")

    rep.verdict(cache["resolvability"] >= 0.7,
                f"resolvability {cache['resolvability']:.2f} should reach "
                "0.7 for the density shape to be usable (below that, "
                "variance only)")

    # Save the recovered density itself, so a surprising export can be
    # inspected rather than inferred from three summary numbers, and report
    # its local maxima with the mass they carry: a genuine density has one
    # dominant maximum near zero, and anything else names the problem.
    np.savez(data / "testbed_recovered.npz", xg=xg, pi=pi,
             slopes_log=np.asarray(cache["slopes_log"]),
             dx=cache["dx"], stable_min=cache["stable_min"],
             stable_max=cache["stable_max"], lam=cache["lambda"],
             resolvability=cache["resolvability"])
    dxg = xg[1] - xg[0]
    tot_pi = max(float(pi.sum() * dxg), 1e-300)
    interior = (pi[1:-1] >= pi[:-2]) & (pi[1:-1] >= pi[2:])
    peaks = np.where(interior)[0] + 1
    peaks = peaks[np.argsort(pi[peaks])[::-1][:5]]
    for p in peaks:
        lo_i, hi_i = max(0, p - 5), min(len(xg), p + 6)
        mass = float(pi[lo_i:hi_i].sum() * dxg) / tot_pi
        rep.info(f"local max at {xg[p]:+.3f}  density {pi[p]:.4f}  "
                 f"mass within 5 bins {100 * mass:.1f}%")

    # ---- 6. the null floor at this configuration ---------------------------
    rep.head(f"null floor at the measured configuration "
             f"({a.floor_trials} trials)")
    sigma_b_est = float(np.sqrt(np.mean(np.var(hofx, axis=1, ddof=1))))
    sigma_o_est = float(cache["resolvability"] * sigma_b_est)
    if cache["resolvability"] < 0.05:
        rep.info("skipped: the innovation histogram is no wider than the "
                 "kernel (resolvability ~0), so there is no recoverable "
                 "observation error to calibrate a floor for; the ensemble "
                 "spread has swallowed the signal")
        floor = None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            floor = NC.null_floor(sigma_o_est, sigma_b_est, len(obs),
                                  hofx.shape[1], trials=a.floor_trials,
                                  lam=(1e-1 if a.adaptive or a.lam is None
                                       else a.lam))
        if a.adaptive:
            rep.info("the floor is calibrated with the legacy instrument "
                     "at lam 1e-1: indicative for the adaptive estimate, "
                     "whose own null calibration is future work")
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
    if a.oracle_density:
        # Export the injected density itself. Everything downstream (mode
        # fit, sigma fit, reflection, validation, gates, the treatment run)
        # is identical, so the comparison isolates estimation quality from
        # the cost-function method.
        rep.info("ORACLE: exporting the injected density; the estimate "
                 "above is reported but not assimilated")
        cache = oracle_cache(spec_inj)
        xg = cache["stable_min"] + np.arange(len(cache["slopes_log"])) \
            * cache["dx"]
        pi = analytic_density(spec_inj, xg)
    gaussian_tails(cache, np.asarray(xg), np.asarray(pi), sd, rep)
    png = save_density_plot(data / "testbed_density.png", obs, hofx, xg, pi,
                            inj, a.seed + 2, a.adaptive, cache=cache)
    rep.info(f"density plot written to {png}" if png else
             "no density plot: matplotlib is not installed in this python")
    spec, nfixed = DY.to_spec(cache, save_sigma=True)
    rep.info(f"mode {spec['mode']:+.3f}  sigma at mode "
             f"{spec['sigma at mode']:.3f}  nfixed {nfixed}")
    export_ok = True
    # A fragmented solve leaves most of the mass outside the kept interior:
    # measured with an over-dispersed kernel (ratio 1.24) the QP answers a
    # sharper-than-realizable target with a picket fence of spikes, and no
    # choice of kept run makes that assimilable.
    if cache["kept_mass"] < 0.5:
        export_ok = False
        rep.fail(f"only {100 * cache['kept_mass']:.1f}% of the probability "
                 "mass lies in the kept interior: the solve fragmented "
                 "(known cause: an over-dispersed ensemble makes the kernel "
                 "wider than the innovations support; check the reliability "
                 "ratio)")
    # The gate on the unimodality projection weighs the probability MASS the
    # projection touched, not the bin count: at large sample sizes the kept
    # interior reaches far into the tails, where slope signs wiggle in bins
    # carrying next to no mass, and repairing those is cosmetic. Measured on
    # the synthetic testbed at n=1200, K=100: seventeen projected bins with
    # an L1 of 0.175 against the truth -- an essentially perfect recovery
    # that a raw count gate of five would refuse.
    c_bins = np.asarray(cache["stable_min"]
                        + (np.arange(len(cache["slopes_log"])) + 0.5)
                        * cache["dx"])
    s_raw = np.asarray(cache["slopes_log"], float)
    # the spec is exported in H(x) - y; the cache lives in y - H(x), so
    # reflect the mode back before classifying the cache's slopes.
    # Bins within two grid spacings of the mode are exempt: slope signs
    # across a flat top are mode-location chatter, not shape violations
    # (the C++ uses sigma at mode near there anyway), and they sit on the
    # densest bins, so counting them swamps the gate -- measured at lam 30
    # on the record ensemble, an L1 0.154 recovery was refused at 5.66%
    # with nearly all of it in two near-peak bins.
    mode_d = -spec["mode"]
    w = 2.0 * cache["dx"]
    bad_bins = ((c_bins < mode_d - w) & (s_raw < 0)) \
        | ((c_bins > mode_d + w) & (s_raw > 0))
    p_bins = np.interp(c_bins, xg, pi, left=0.0, right=0.0)
    mass_fixed = float(p_bins[bad_bins].sum()
                       / max(p_bins.sum(), 1e-300))
    rep.info(f"probability mass in projected bins {100 * mass_fixed:.2f}%")
    if mass_fixed > 0.01 * a.max_nfixed:
        export_ok = False
        rep.fail(f"the unimodality projection touched "
                 f"{100 * mass_fixed:.2f}% of the probability mass "
                 f"(gate {a.max_nfixed}%): the estimate is too noisy to "
                 "assimilate")
    # The mode gate is TESTBED POLICY, not a fundamental property: under the
    # unbiased-background anchor the estimator recovers genuinely shifted
    # error densities correctly (verified: a +0.6-mode injection comes back
    # at +0.54 with unbiased members). Every Stage A injection has its mode
    # at zero, so a far recovered mode is a truth-check failure here -- the
    # gauge says it measures ensemble bias, and the bias line above bounds
    # that -- and, separately, the Jo scalar the C++ reports is
    # 0.5 <d, g(d)>, which equals the evolving-Gaussian value
    # 0.5 <d - m, g(d)> only near m = 0, so a far mode also makes the
    # reported JoJc negative and oops refuses to minimize until the fork
    # carries the (d - m) fix.
    if abs(spec["mode"]) > 0.5 * max(sd, 1e-12):
        export_ok = False
        rep.fail(f"exported mode {spec['mode']:+.3f} sits "
                 f"{abs(spec['mode']) / max(sd, 1e-12):.1f} recovered sigmas "
                 "from zero; an obs-error density should peak near zero and "
                 "the C++ Jo scalar assumes it")
    bad = DY.check(spec)
    if bad:
        export_ok = False
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
    if not export_ok:
        rep.info("skipped: the density failed the export gates above, so "
                 "running it would only crash or mislead; fix the estimate "
                 "first (Data/testbed_recovered.npz holds it)")
        return rep.finish()
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
    # optimizes, so deciding by RMSE hands the control home advantage; this
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
