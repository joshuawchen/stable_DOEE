# jedi_export — DOEE density -> JEDI non-Gaussian Jo

Turns a stable_DOEE estimate into the `non gaussian cost` YAML block consumed by
`CostJoEvolvingGaussian` (Hu, Geer & van Leeuwen 2025, QJRMS 151:e5050), and
patches it into the JEDI configuration.

## Setup

The estimator lives in `stable_doee.py` at the repository root and is imported
by both this pipeline and the notebook. If it is not there yet, generate it once
from the notebook and commit it:

    python3 migrate_notebook.py ../stable_DOEE.ipynb ../stable_doee.py

Then change the notebook to import from it, and delete `migrate_notebook.py`:

    from stable_doee import estimate_noise_pmf, finalize_pdf_cache

That way round the module is the source of truth, exploration and production
cannot drift apart, and the estimator is importable without executing a
notebook.

Requires numpy, netCDF4 and quadprog.

## Use

    # survey first: what strata exist, and is any big enough?
    python3 run_doee_export.py --diags 'diags/jdiag_adpsfc_*.nc' \
        --variable airTemperature \
        --members hofx_mem001 hofx_mem002 hofx_mem003 --survey

    # then estimate and write, for ONE stratum
    python3 run_doee_export.py --diags 'diags/jdiag_adpsfc_*.nc' \
        --variable airTemperature \
        --members hofx_mem001 hofx_mem002 hofx_mem003 \
        --obstype 181 \
        --obtype-yaml .../obtype_config/adpsfc_airTemperature_181.yaml \
        --basic-yaml  .../basic_config/mpasjedi_hybrid3denvar.yaml

`--dry-run` prints the block instead of editing. Edits are textual, so comments
and everything else in the file survive; a backup is left alongside.

## What goes in

`estimate_noise_pmf(X, Y)` solves `Y = X + N` by matching `f_{Y-X}` to
`f_{X1-X2} * pi_N`. The inputs that make `N` the observation error are

    Y : ENSEMBLE INNOVATIONS      y - H(x_k)         = eps_o - eps_b
    X : ENSEMBLE PERTURBATIONS    H(x_k) - H(x_bar)  ~ -eps_b

because then `f_{Y-X} = f_{eps_o} * f_{-eps_b} * f_{eps_b}` and
`f_{X1-X2} = f_{-eps_b} * f_{eps_b}`, so the deconvolution returns `f_{eps_o}`.
This is DOEE on ensemble innovations. The estimator still treats its two arrays
as unpaired samples; the pairing is used only to form the innovations and
perturbations, which is ordinary DA bookkeeping. `samples.samples()` does this.

**Do not pass raw ObsValue and raw H(x).** Both carry the full variability of
the field, which dwarfs the errors: several K of weather against about 1 K of
observation error. Measured on synthetic data, the estimator then returned a
width roughly twice the truth when the field spread was 3x the noise, and failed
outright with a non-positive-definite QP at 6x. Forming innovations and
perturbations first cancels the field exactly, and the recovered width becomes
insensitive to how variable the field is: 6 K and 12 K field spread give the
same answer, for Gaussian and heavy-tailed errors alike.

**Analysis residuals (O-A) are not an input.** The analysis has already fitted
those observations, so the residual is shrunk by construction and the estimated
error comes out biased low. Feed that back into the assimilation and the bias
compounds each cycle: smaller error, heavier weight, smaller residual. The loop
converges on a self-consistent and wrong density, and Desroziers-style checks
computed from the same departures will not reveal it.

**At least two members are required**, and `load` refuses fewer. The
perturbations characterise the background error being deconvolved away; with one
member there is nothing to deconvolve. The `jdiag` files from the deterministic
jedivar carry only the control's `hofx`, so where member H(x) comes from is a
workflow question to settle first.

## Choosing what to pool

Pooling observations with different error characteristics manufactures apparent
non-Gaussianity: a mixture of Gaussians of different widths is heavy-tailed. The
reference warns about precisely this. DOEE will faithfully recover the mixture
and the evolving-Gaussian method will appear to help by down-weighting the
wings, but the right response is to separate the populations.

    always split on : variable, and ObsType (BUFR type: 181, 187, 120, ...)
    split if you can: pressure level or channel, and region if regimes differ
    pool across     : cycles -- the safe axis, and the sample-size lever
    or stratify by  : a state-dependent predictor, as the reference does with
                      symmetric cloud, fitting one density per bin

Pooling over cycles assumes the error density is stationary over the window
(plausible for instrument and representation error over days, questionable
across seasons) and treats repeated observations from the same platform as
independent when they are serially correlated, so the effective sample size is
below the raw count. Report the window and the counts alongside any density.

`--survey` lists the strata with counts and a usability verdict and estimates
nothing. The driver refuses to pool several strata into one density, and refuses
a stratum below `--min-n` (default 1000), since below roughly 1e3 the tails are
noise and the tails are the point of the method. `homogeneity()` splits a
stratum across the window and compares the innovation spread; a large difference
means it is still a mixture.

## The two YAML edits

They are in different files and do different things:

    density   -> the obtype template, e.g.
                 validated_yamls/templates/obtype_config/adpsfc_airTemperature_181.yaml
    switch on -> `jo type: evolving gaussian` under `observations:` in the basic
                 config, e.g. basic_config/mpasjedi_hybrid3denvar.yaml

Without the second the density is inert and the run stays Gaussian. The driver
does both when given `--obtype-yaml` and `--basic-yaml`, and says so when the
switch is omitted.

## Before an estimate is used

`to_spec` returns `nfixed`, the number of log slopes that had to be projected to
satisfy the unimodality the cost function requires. Zero means the estimate is
clean. More than a handful means it is too noisy to assimilate: widen the bins
or pool more cycles, rather than assimilating the projection.

`check(spec)` sweeps the effective variance exactly as the C++ evaluates it and
reports any point that is not finite and positive. Run it before an experiment.

The mode is located from the argmax of the reconstructed log density, not from
the last positive slope: a single ragged bin, which deconvolution readily
produces where it is ill-posed, moves the latter far from the true mode and
every downstream quantity inherits the error. Sigma at the mode is then fitted
to the cleaned slopes, so one bad bin cannot set the weight given to every
near-mode observation.

## Files

    ../stable_doee.py    the estimator (source of truth; notebook imports it)
    samples.py           read IODA diag files; innovations and perturbations,
                         stratification, usability and homogeneity checks
    doee_to_yaml.py      cache -> `non gaussian cost` spec and YAML block
    patch_yaml.py        edit the obtype template and the basic config
    run_doee_export.py   driver tying the above together
    selftest.py          checks for doee_to_yaml (numpy only)
    migrate_notebook.py  one-time lift of the estimator out of the notebook;
                         delete once stable_doee.py is committed
