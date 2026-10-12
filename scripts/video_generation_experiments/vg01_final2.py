import numpy as np, math
rng = np.random.default_rng(0)
mu1, mu2 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
sigma = 0.6
T = 1000
betas = np.linspace(1e-6, 0.01, T)
abar = np.cumprod(1-betas)
sa = np.sqrt(abar); sb = np.sqrt(1-abar)

def score_mix(x, t):
    """exact score of p(x_t): mixture of N(sqrt(abar)*mu_k, s^2 + abar*sigma^2)"""
    m1, m2 = sa[t]*mu1, sa[t]*mu2
    v = sb[t]**2 + abar[t]*sigma*sigma
    lps = [np.log(0.5) - np.sum((x-m)**2, axis=1)/(2*v) - np.log(2*np.pi*v) for m in (m1, m2)]
    lps = np.array(lps); mx = lps.max(axis=0)
    resp = np.exp(lps-mx); resp /= resp.sum(axis=0)
    return resp[0][:,None]*(-(x-m1)/v) + resp[1][:,None]*(-(x-m2)/v)

def sample(w, n=4000):
    x = rng.normal(0, 1, (n,2))
    for t in range(T-1, -1, -1):
        u = score_mix(x, t)                    # unconditional: both modes
        v = sb[t]**2 + abar[t]*sigma*sigma
        c = -(x - sa[t]*mu2)/v                 # conditional: right mode only
        x = x + 0.5*betas[t]*(x + u + w*(c-u)) # PF-ODE + CFG
    return x

x = sample(0.0)
right, left = x[x[:,0]>0], x[x[:,0]<=0]
print(f"uncond: right {np.mean(x[:,0]>0):.1%} | right mode {right[:,0].mean():+.2f}±{right[:,0].std():.2f} (ideal +2.0±0.6)"
      f" | left mode {left[:,0].mean():+.2f}±{left[:,0].std():.2f} (ideal -2.0±0.6)")
for w in [1.0, 3.0, 7.0]:
    x = sample(w)
    print(f"CFG w={w}: right {np.mean(x[:,0]>0):.1%} | mode {x[:,0].mean():+.2f}±{x[:,0].std():.2f} | y-std {x[:,1].std():.2f}")
