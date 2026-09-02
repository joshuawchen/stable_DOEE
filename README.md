# stable_DOEE

Deconvolution of observation-error densities (DOEE) from departures and a
measured kernel, regularized so that the recovered density is usable as a
density: positive everywhere it is defined, smooth in the log, with tails
that decay, and with an honest account of what the data could not resolve.

The repository began as a notebook. It is now a small library with three
consumers, so the entry points and their contracts are described here
rather than in a cell.

## What is estimated

Given departures `d = eps - delta` and samples of the contaminating error
`delta` (the kernel), recover the density of `eps` by solving

    f_d = f_delta * f_eps

as a constrained quadratic program on a grid: `pi >= 0`, unit mass, and a
penalty on the THIRD DIFFERENCE OF log pi. The third difference is the
point. Its null space contains every log-quadratic and every log-linear
density, so it privileges neither Gaussian nor exponential tails, and
because it acts on log pi it keeps acting where pi is small. A penalty on
differences of pi returns an excess kurtosis near -0.6 whatever the truth
is; a penalty on the second difference of log pi imposes exponential
tails. Measured against known truths, this one tracks tail weight
conservatively rather than imposing a shape.

Two things are reported alongside the density and both should be read
before it is used:

- `resolvability`, sigma_o / sigma_b. Below about 0.5 the deconvolution is
  ill posed and no penalty rescues it. The estimator says so rather than
  returning a confident wrong answer.
- `kept_mass`, the fraction of the mass in the contiguous run the export
  keeps. Well below one means the solve fragmented.

## Entry points

    estimate_adaptive(grid, f_d, f_k, innov, groups, ...)
        The production path. Per-bin noise is MEASURED from grouped
        half-splits rather than assumed, the data term is whitened by it,
        the penalty is normalized by dx^-5 so the smoothing means the same
        thing at every bin count, and the smoothing is chosen by a
        cross-split one-standard-error rule. Returns (grid, pi, cache) with
        pi over the FULL grid, so a genuinely bimodal recovery is not
        truncated to its tallest mode; `cache["stable_min"]`,
        `cache["stable_max"]` bound the contiguous run the export trusts.

    estimate_from_histograms(grid, f_d, f_k, ...)
        The earlier path, kept: no noise model, smoothing cross-validated
        on the held-out innovation likelihood, with the documented flat
        criterion and its calibrated fallback.

    estimate_noise_pmf_reg(X, Y, n_members, ...)
        The original signature, taking perturbations and innovations.

    histograms_from_ensemble, innovation_groups, adaptive_bin_count
        Build the two histograms from an observation vector and a
        background ensemble, pairing AT THE SAME LOCATION so the field
        cancels before the deconvolution sees anything, and label the
        groups so that folds and splits do not leak one observation's
        error draw across the split.

    jedi_export/doee_to_yaml.py
        Turn a cache into the piecewise-log-linear spec that JEDI's
        non-Gaussian Jo reads, with the unimodality repair and the tail
        guards.

## Consumers

- `jedi_export/`, the JEDI non-Gaussian Jo density export and the l95
  mirror that calibrates it.
- The RRFS/UFO update work, through the same export.
- The observation-error intrinsic-dimension paper (Chen and van Leeuwen),
  whose pipeline wraps `estimate_adaptive` per channel and uses the
  recovered densities and their anamorphosis. That wrapper brackets the
  interval this repository publishes as stable, rather than re-deriving
  one.

## Tests

    python3 test_zero_bins.py             # the zero-bin fill
    python3 jedi_export/selftest.py       # the export: modes, repairs, tails
    python3 validate_doee_variants.py     # against known truths, with L1

`l95-testbed/` holds the null calibration the recovered kurtosis is judged
against, and `jedi_export/test_phase3_driver.py` exercises the driver
against a stub build tree.

## Requirements

numpy, scipy, quadprog.
