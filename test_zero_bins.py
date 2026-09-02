"""The zero-bin fill: what it does, what it leaves alone, and that the
result is still a normalized density.

    python3 test_zero_bins.py

Four checks on _fill_zero_bins itself, then one end to end through
estimate_adaptive on a Gaussian truth with a measured kernel, which is the
path the DA export uses.

Background. The QP is constrained pi >= 0 and quadprog's active set puts
exact zeros at isolated bins where a noisy histogram pushes a bin
negative. The third-difference-of-log penalty cannot itself reach zero, so
these are a solver boundary artifact rather than a claim that the density
vanishes, and a consumer that interpolates log pi across such a bin digs a
hole to log(1e-300) = -690 nats. Measured on cycled-filter archives with a
known error law: isolated zeros at 1.5 to 6 widths from the center caught
0.5 to 1.5 percent of one channel's true errors at -378 nats each, adding
2 to 6 nats per channel to the KL of a density whose width ratio and kept
mass both passed every gate.
"""
import numpy as np

from stable_doee_reg import (_fill_zero_bins, estimate_adaptive,
                             histograms_from_ensemble, innovation_groups,
                             adaptive_bin_count)

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))


print("unit: _fill_zero_bins")

# 1. an isolated interior zero is filled, and the result integrates to one
dx = 0.1
g = np.arange(-4, 4 + dx, dx)
pi = np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi)
pi /= pi.sum() * dx
hole_at = np.array([28, 55, 70])
pi_h = pi.copy()
pi_h[hole_at] = 0.0
pi_f, n = _fill_zero_bins(pi_h, dx)
check("isolated zeros filled", n == 3, f"n_filled {n}")
check("unit mass after filling", abs(pi_f.sum() * dx - 1.0) < 1e-12,
      f"mass {pi_f.sum() * dx:.15f}")
check("no zeros left inside the support", (pi_f[hole_at] > 0).all())
rel = np.abs(pi_f[hole_at] / pi[hole_at] - 1.0).max()
check("filled values near the truth", rel < 0.02, f"max relative error {rel:.4f}")
untouched = np.setdiff1d(np.arange(pi.size), hole_at)
drift = np.abs(pi_f[untouched] / pi_h[untouched] - 1.0).max()
check("other bins moved only by renormalization", drift < 5e-3,
      f"max relative change {drift:.2e}")

# 2. a long interior gap is NOT bridged: a body and a far lobe
pi2 = np.zeros_like(g)
body = np.abs(g) < 1.0
lobe = (g > 3.0) & (g < 3.4)
pi2[body] = np.exp(-0.5 * (g[body] / 0.4) ** 2)
pi2[lobe] = 0.01
pi2 /= pi2.sum() * dx
pi2_f, n2 = _fill_zero_bins(pi2, dx)
check("long gap left alone", n2 == 0, f"n_filled {n2}")
check("long-gap array unchanged", np.allclose(pi2_f, pi2))

# 3. leading and trailing zeros are the support edge, not holes
pi3 = np.zeros_like(g)
pi3[20:60] = 1.0
pi3 /= pi3.sum() * dx
pi3_f, n3 = _fill_zero_bins(pi3, dx)
check("edge zeros left alone", n3 == 0 and np.allclose(pi3_f, pi3))

# 4. a degenerate input does not raise
pi4 = np.zeros_like(g)
pi4[10] = 1.0 / dx
_, n4 = _fill_zero_bins(pi4, dx)
check("single-bin input handled", n4 == 0)

print("\nend to end: estimate_adaptive on a Gaussian truth")
rng = np.random.default_rng(0)
N, K, sd_o, sd_b = 4000, 20, 0.03, 0.03 * 0.32
truth = rng.normal(0, 1.0, N)                       # the field
hofx = truth[:, None] + rng.normal(0, sd_b, (N, K))
obs = truth + rng.normal(0, sd_o, N)
_, _, _, innov = histograms_from_ensemble(obs, hofx, seed=0)
nb = adaptive_bin_count(innov)
grid, f_d, f_k, innov = histograms_from_ensemble(obs, hofx, seed=0, n_bins=nb)
groups = innovation_groups(N, K)
g_out, pi_out, cache = estimate_adaptive(grid, f_d, f_k, innov, groups, seed=0)
dxo = g_out[1] - g_out[0] if g_out.size > 1 else 1.0
pos = np.flatnonzero(pi_out > 0)
interior_zeros = int((pi_out[pos[0]:pos[-1] + 1] <= 0).sum()) if pos.size > 1 else 0
print(f"  grid {grid.size} bins, kept run {g_out.size}, mu {cache['lambda']:.1e}, "
      f"kept mass {cache['kept_mass']:.3f}, zero bins filled {cache.get('zero_bins_filled')}")
check("cache reports the count", "zero_bins_filled" in cache)
check("no interior zeros survive in the returned pi", interior_zeros == 0,
      f"{interior_zeros} left")
check("returned pi integrates to one", abs(pi_out.sum() * dxo - 1.0) < 1e-9,
      f"{pi_out.sum() * dxo:.9f}")
check("kept mass is the main run's share of it",
      0.0 < cache["kept_mass"] <= 1.0 + 1e-9, f"{cache['kept_mass']:.6f}")
mass = max(pi_out.sum() * dxo, 1e-30)
mu_o = (pi_out * g_out).sum() * dxo / mass
w = np.sqrt((pi_out * (g_out - mu_o) ** 2).sum() * dxo / mass)
check("width within 25 percent of the truth", abs(w / sd_o - 1) < 0.25,
      f"{w:.5f} against {sd_o:.5f} ({w / sd_o:.2f}x)")

print("\nPASS" if ok else "\nFAIL")
raise SystemExit(0 if ok else 1)
