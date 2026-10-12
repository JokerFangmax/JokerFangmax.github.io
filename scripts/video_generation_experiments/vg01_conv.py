import numpy as np, math
rng = np.random.default_rng(0)
mu1, mu2 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
sigma = 0.6

def score_mix(x, s):
    v = s*s + sigma*sigma
    lps = [np.log(0.5) - np.sum((x-mu)**2, axis=1)/(2*v) - np.log(2*np.pi*v) for mu in (mu1, mu2)]
    lps = np.array(lps); m = lps.max(axis=0)
    resp = np.exp(lps-m); resp /= resp.sum(axis=0)
    return resp[0][:,None]*(-(x-mu1)/v) + resp[1][:,None]*(-(x-mu2)/v)

def run(T, bmax, n=4000):
    betas = np.linspace(1e-6, bmax, T)
    abar = np.cumprod(1-betas)
    sb = np.sqrt(1-abar)
    x = rng.normal(0, 1, (n,2))
    for t in range(T-1, -1, -1):
        x = x + 0.5*betas[t]*(x + score_mix(x, sb[t]))
    return x

for T, bmax in [(1000, 0.01), (2000, 0.005), (4000, 0.0025), (8000, 0.00125)]:
    x = run(T, bmax)
    right = x[x[:,0]>0]
    print(f"T={T:5d} bmax={bmax:.5f}: right {np.mean(x[:,0]>0):.1%}, "
          f"mode mean {right[:,0].mean():+.2f} (ideal +2.00), mode std {right[:,0].std():.2f} (ideal 0.60)")
