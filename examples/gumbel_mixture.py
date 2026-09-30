"""The original estimator on a known, strongly non-Gaussian noise.

    python3 examples/gumbel_mixture.py        # writes gumbel_mixture.png

Y = X + N with X standard Gaussian and N a three-component Gumbel mixture.
The numbers of X and Y samples need not be equal. For multivariate X and Y,
apply the method to each component. Needs matplotlib.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stable_doee import (dlogpdf_scalar, estimate_noise_pmf,  # noqa: E402
                         finalize_pdf_cache, logpdf_scalar, pdf)


def noise(rng, n):
    return np.concatenate((rng.gumbel(-1, 1, 5 * n // 8), rng.gumbel(5, 1, n // 4),
                           rng.gumbel(10, 1, n // 8)))


rng = np.random.default_rng(0)
N = 2_000_000
X = rng.normal(0, 1, N)
Y = X + noise(rng, N)

# deconvolve the histograms, then build the interpolator
xg, pi, cache = estimate_noise_pmf(X, Y, seed=0)
cache = finalize_pdf_cache(xg, cache)

x = np.linspace(-10, 20, 502)
s = np.linspace(-3, 15.5, 1000)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].hist(noise(rng, N), bins=400, density=True, color="0.8", label="true noise")
ax[0].plot(xg, pi, label="estimate on the histogram grid")
ax[0].plot(x, pdf(x, cache), ".", ms=2, label="estimate, interpolated")
ax[0].legend(fontsize=7)
ax[1].plot(s, [-logpdf_scalar(v, cache) for v in s])
ax[1].set_title("negative log pdf")
ax[2].plot(s, [-dlogpdf_scalar(v, cache) for v in s])
ax[2].set_title("its derivative")
fig.tight_layout()
fig.savefig("gumbel_mixture.png", dpi=120)
print("wrote gumbel_mixture.png")
