import numpy as np, math
rng = np.random.default_rng(0)
mu1, mu2 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
sigma = 0.6
T = 1000
betas = np.linspace(1e-5, 0.01, T)
abar = np.cumprod(1-betas)
sb = np.sqrt(1-abar)

def score_mix(x, s):
    v = s*s + sigma*sigma
    lps = [np.log(0.5) - np.sum((x-mu)**2, axis=1)/(2*v) - np.log(2*np.pi*v) for mu in (mu1, mu2)]
    lps = np.array(lps); m = lps.max(axis=0)
    resp = np.exp(lps-m); resp /= resp.sum(axis=0)
    return resp[0][:,None]*(-(x-mu1)/v) + resp[1][:,None]*(-(x-mu2)/v)

def sample(w, n=4000):
    x = rng.normal(0, 1, (n,2))
    for t in range(T-1, -1, -1):
        s = sb[t]
        u = score_mix(x, s)
        v = s*s + sigma*sigma
        c = -(x - mu2)/v          # conditional score: right mode
        g = u + w*(c - u)         # CFG
        x = x + 0.5*betas[t]*(x + g)
    return x

x = sample(0.0)
print(f"PF-ODE uncond: right {np.mean(x[:,0]>0):.1%}, x-std={x[:,0].std():.3f} (ideal 2.09), y-std={x[:,1].std():.3f} (ideal 0.6)")
right = x[x[:,0]>0]; left = x[x[:,0]<=0]
print(f"   mode means: right {right[:,0].mean():.2f} (ideal 2.0, std {right[:,0].std():.2f} ideal 0.6), left {left[:,0].mean():.2f} (ideal -2.0, std {left[:,0].std():.2f})")
for w in [1.0, 3.0, 7.0]:
    x = sample(w)
    print(f"CFG w={w}: right {np.mean(x[:,0]>0):.1%}, mode mean={x[:,0].mean():.2f}, mode std={x[:,0].std():.2f}, y-std={x[:,1].std():.2f}")
