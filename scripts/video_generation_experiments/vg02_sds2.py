import numpy as np, math
rng = np.random.default_rng(1)
mu_c = np.array([2.0, 0.0]); sigma = 0.6

def eps_phi(x_t, sa, sb):
    v = sb*sb + sa*sa*sigma*sigma
    return sb*(x_t - sa*mu_c)/v     # = -sb*score

T = 1000
betas = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1-betas)

def sds_run(t_lo_frac, t_hi_frac, steps=500, eta=0.02, seed=0):
    r = np.random.default_rng(seed)
    mu = np.array([0.0, 0.0])
    for k in range(steps):
        t = r.integers(int(t_lo_frac*T), int(t_hi_frac*T))
        sa, sb = math.sqrt(abar[t]), math.sqrt(1-abar[t])
        eps = r.normal(0, 1, 2)
        x_t = sa*mu + sb*eps
        g = eps_phi(x_t, sa, sb) - eps
        g = g / (np.linalg.norm(g) + 1e-12)      # direction-only, like normalized SDS variants
        mu = mu + eta*g
    return mu

runs = np.array([sds_run(0.02, 0.98, seed=s)[0] for s in range(30)])
print(f"full t-range: mean={runs.mean():.2f} (ideal 2.0), std={runs.std():.2f}, all right mode: {np.all(runs>0)}")
runs2 = np.array([sds_run(0.02, 0.20, seed=s)[0] for s in range(10)])
print(f"low-t only  : mean={runs2.mean():.2f}, std={runs2.std():.2f}  (weak/noisy signal expected)")
