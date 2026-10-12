import numpy as np
rng = np.random.default_rng(0)
mu1, mu2, sigma = np.array([-2.0,0.0]), np.array([2.0,0.0]), 0.6

def run(T, bmax, w, n=4000):
    betas = np.linspace(1e-6, bmax, T)
    abar = np.cumprod(1-betas); sa = np.sqrt(abar); sb = np.sqrt(1-abar)
    def sc_mix(x, t):
        m1, m2 = sa[t]*mu1, sa[t]*mu2
        v = sb[t]**2 + abar[t]*sigma*sigma
        lps = np.array([np.log(0.5)-np.sum((x-m)**2,axis=1)/(2*v)-np.log(2*np.pi*v) for m in (m1,m2)])
        mx = lps.max(axis=0); r = np.exp(lps-mx); r /= r.sum(axis=0)
        return r[0][:,None]*(-(x-m1)/v) + r[1][:,None]*(-(x-m2)/v)
    x = rng.normal(0,1,(n,2))
    for t in range(T-1,-1,-1):
        v = sb[t]**2 + abar[t]*sigma*sigma
        u = sc_mix(x, t); c = -(x - sa[t]*mu2)/v
        x = x + 0.5*betas[t]*(x + u + w*(c-u))
    return x

for w in [3.0]:
    for T, bmax in [(1000,0.01),(4000,0.0025),(16000,0.000625)]:
        x = run(T, bmax, w)
        print(f"w={w} T={T:6d}: mode {x[:,0].mean():+.3f}±{x[:,0].std():.3f} (exact target +2.00±0.60)")
