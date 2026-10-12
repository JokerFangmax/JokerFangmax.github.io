import numpy as np, math
rng = np.random.default_rng(1)
# prior: conditional = right mode N((2,0), 0.36 I); closed-form score
mu_c = np.array([2.0, 0.0]); sigma = 0.6

def eps_phi(x_t, sa, sb):
    v = sb*sb + sa*sa*sigma*sigma
    s = -(x_t - sa*mu_c)/v          # score of marginal p(x_t|c)
    return -sb*s                     # eps = -sqrt(1-abar)*score

T = 1000
betas = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1-betas)

# SDS on parameter mu (the "rendered image" IS mu; J = I)
runs = []
for run in range(30):
    mu = np.array([0.0, 0.0])
    for k in range(200):
        t = rng.integers(int(0.02*T), int(0.98*T))   # DreamFusion-style t range
        sa, sb = math.sqrt(abar[t]), math.sqrt(1-abar[t])
        eps = rng.normal(0, 1, 2)
        x_t = sa*mu + sb*eps
        grad = eps_phi(x_t, sa, sb) - eps
        mu = mu + 0.05 * (sb/(1-abar[t])) * grad
    runs.append(mu[0])
runs = np.array(runs)
print(f"30 runs: mean={runs.mean():.2f} (ideal 2.0), std={runs.std():.2f}, "
      f"all in right mode: {np.all(runs>0)}")

# same but ONLY low-noise t (t in [0.02, 0.2]*T) — expect weak signal
runs2 = []
for run in range(10):
    mu = np.array([0.0, 0.0])
    for k in range(200):
        t = rng.integers(int(0.02*T), int(0.2*T))
        sa, sb = math.sqrt(abar[t]), math.sqrt(1-abar[t])
        eps = rng.normal(0, 1, 2)
        x_t = sa*mu + sb*eps
        mu = mu + 0.05 * (sb/(1-abar[t])) * (eps_phi(x_t, sa, sb) - eps)
    runs2.append(mu[0])
print(f"low-t only: mean={np.mean(runs2):.2f}, final |mu| sample: {np.round(runs2,2)}")
