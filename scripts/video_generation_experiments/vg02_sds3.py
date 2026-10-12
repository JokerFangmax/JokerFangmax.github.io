import numpy as np, math
mu_c = np.array([2.0, 0.0]); sigma = 0.6
T = 1000
betas = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1-betas)

def eps_phi(x_t, sa, sb):
    v = sb*sb + sa*sa*sigma*sigma
    return sb*(x_t - sa*mu_c)/v

def sds_run(t_lo, t_hi, steps=300, eta=0.05, seed=0, init=(0.0,0.0)):
    r = np.random.default_rng(seed)
    mu = np.array(init, float)
    for k in range(steps):
        t = r.integers(int(t_lo*T), int(t_hi*T))
        sa, sb = math.sqrt(abar[t]), math.sqrt(1-abar[t])
        eps = r.normal(0, 1, 2)
        x_t = sa*mu + sb*eps
        mu = mu + eta * (sa/sb) * (eps - eps_phi(x_t, sa, sb))
    return mu

runs = np.array([sds_run(0.02, 0.98, seed=s)[0] for s in range(30)])
print(f"full t-range : mean={runs.mean():+.2f} (ideal +2.0), std={runs.std():.2f}, all>0: {np.all(runs>0)}")
lo = np.array([sds_run(0.02, 0.20, seed=s)[0] for s in range(10)])
print(f"low-t only   : mean={lo.mean():+.2f}, std={lo.std():.2f}")
hi = np.array([sds_run(0.80, 0.98, seed=s)[0] for s in range(10)])
print(f"high-t only  : mean={hi.mean():+.2f}, std={hi.std():.2f}")
# trajectory of one run for illustration
r = np.random.default_rng(0); mu = np.array([0.0,0.0]); traj=[mu[0]]
for k in range(300):
    t = r.integers(20, 980); sa, sb = math.sqrt(abar[t]), math.sqrt(1-abar[t])
    eps = r.normal(0,1,2); x_t = sa*mu + sb*eps
    mu = mu + 0.05*(sa/sb)*(eps - eps_phi(x_t, sa, sb))
    traj.append(mu[0])
print("sample trajectory (every 60 steps):", [f"{traj[i]:+.2f}" for i in range(0,301,60)])
