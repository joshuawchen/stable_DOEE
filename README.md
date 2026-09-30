# stable_DOEE

A numerically stable implementation of deconvolution-based observation error
estimation (DOEE; Hu, van Leeuwen and Geer, 2024). From ensemble innovations,
the differences between the observations and the prior ensemble members in
observation space, it recovers the probability density of the observation
error of one channel, without assuming that the error is Gaussian. The
recovered density is usable as a density: positive wherever it is defined,
smooth in the log, with tails that decay, and reported together with what
the data could not resolve.

## Install

    pip install git+https://github.com/joshuawchen/stable_DOEE@v1.0.1

This installs the two library modules, `stable_doee` and `stable_doee_reg`,
with their requirements numpy and quadprog (Python 3.9 or later).

## Example

```python
import numpy as np
from stable_doee_reg import estimate_adaptive_from_ensemble

rng = np.random.default_rng(0)
n_obs, K = 20_000, 40
truth = rng.normal(280.0, 5.0, n_obs)                      # the observed field
mean = truth + rng.normal(0.0, 0.5, n_obs)                 # background mean, with its error
hofx = mean[:, None] + rng.normal(0.0, 0.5, (n_obs, K))    # prior members, observation space
obs = truth + rng.gumbel(0.0, 0.8, n_obs)                  # observations with a skewed error

grid, pi, cache = estimate_adaptive_from_ensemble(obs, hofx)
print(cache["resolvability"], cache["kept_mass"])          # read these first (below)
```

`pi` is the recovered density of the observation error on `grid`. The
kernel is formed from differences between members, which assumes that the
truth is statistically indistinguishable from a member, as in the example.

## What is estimated

Given departures `d = eps - delta` and samples of the contaminating error
`delta` (the kernel, from the ensemble perturbations), the density of `eps`
solves

    f_d = f_delta * f_eps

We solve it as a constrained quadratic program on a grid: `pi >= 0`, unit
mass, and a penalty on the third difference of `log pi`. The null space of
that penalty contains every log-quadratic and every log-linear density, so it
favours neither Gaussian nor exponential tails, and because it acts on
`log pi` it keeps acting where `pi` is small.

Two quantities are reported with the density and should be read before it is
used:

- `resolvability`, sigma_o / sigma_b. Below about 0.5 the deconvolution is
  ill posed and no penalty rescues it.
- `kept_mass`, the fraction of the mass in the contiguous run that is kept.
  Well below one means the solution fragmented.

## Entry points

`stable_doee_reg`

    estimate_adaptive(grid, f_d, f_k, innov, groups, ...)
        The main estimator. The per-bin noise is measured from grouped
        half-splits, the data term is whitened by it, the penalty is scaled
        with the bin width so the smoothing means the same at every bin
        count, and the smoothing is chosen by a cross-split one-standard-error
        rule. Returns (grid, pi, cache), with pi over the full grid;
        cache["stable_min"] and cache["stable_max"] bound the contiguous run
        that is kept.

    estimate_adaptive_from_ensemble(obs, hofx, ...)
        The same from the observations and the prior members in observation
        space, with the histograms and the bin count built for you.

    histograms_from_ensemble, innovation_groups, adaptive_bin_count
        Build the two histograms from an observation vector and a background
        ensemble, pairing each observation with the members at the same
        location, so that the field cancels before the deconvolution, and
        label the groups so that splits do not separate one observation's
        innovations.

    estimate_from_histograms(grid, f_d, f_k, ...)
    estimate_noise_pmf_reg(X, Y, n_members, ...)
        The earlier regularized path, without a noise model, with the
        smoothing cross-validated on the held-out innovation likelihood.

`stable_doee`

    estimate_noise_pmf(X, Y, ...)
        The original estimator: a smoothness penalty on first differences of
        pi, with log-linear interior and quadratic-log tails.

    finalize_pdf_cache, pdf, logpdf_scalar, dlogpdf_scalar
        Evaluate a recovered density and its log gradient anywhere, with
        quadratic log tails beyond the grid (examples/gumbel_mixture.py).

## Tests, validation and examples

From a clone:

    pip install -e ".[test,examples]"
    pytest                                      # the zero-bin fill, end to end
    python3 scripts/validate_doee_variants.py   # original and regularized, against known truths
    python3 examples/gumbel_mixture.py          # the original estimator on a Gumbel mixture

## Related

The export of recovered densities to the non-Gaussian observation term of
JEDI, and the Lorenz-95 testbed that calibrates it, are on the
`jedi-density-export` branch. The code of Chen and van Leeuwen,
"Intrinsic dimension of observation-error dependence in data assimilation",
uses `estimate_adaptive` for the individual error pdfs of each channel.

## Reference

Hu, C.-C., van Leeuwen, P. J. and Geer, A. J. (2024) A non-parametric way to
estimate observation errors based on ensemble innovations. *Quarterly Journal
of the Royal Meteorological Society*, 150(761), 2296–2315.
https://doi.org/10.1002/qj.4710

## License

MIT, see [LICENSE](LICENSE).
