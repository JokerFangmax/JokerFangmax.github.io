import numpy as np, math
rng = np.random.default_rng(0)
mu1, mu2 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
sigma = 0.6

def score_mix(x, s):
    v = s*s + sigma*sigma
    lps = []
    for mu in (mu1, mu2):
        d = x - mu
        lps.append(np.log(0.5) - np.sum(d*d, axis=1)/(2*v) - np.log(2*np.pi*v))
    lps = np.array(lps)
    m = lps.max(axis=0)
    resp = np.exp(lps - m); resp /= resp.sum(axis=0)
    sc = resp[0][:,None]*(-(x-mu1)/v) + resp[1][:,None]*(-(x-mu2)/v)
    return sc

T = 1000
betas = np.linspace(1e-5, 0.01, T)
alphas = 1-betas
abar = np.cumprod(alphas)
sqrt_abar = np.sqrt(abar); sqrt_1mabar = np.sqrt(1-abar)

def ddim(steps):
    x = rng.normal(0, 1, (4000,2))
    taus = np.linspace(T-1, 0, steps+1).round().astype(int)
    for k in range(steps):
        t, tp = int(taus[k]), int(taus[k+1])
        s = sqrt_1mabar[t]
        eps = -s*score_mix(x, s)
        x0 = (x - s*eps)/sqrt_abar[t]
        x = sqrt_abar[tp]*x0 + sqrt_1mabar[tp]*eps
    return x

for steps in [50, 100, 1000]:
    x = ddim(steps)
    print(f"DDIM-{steps}: right {np.mean(x[:,0]>0):.1%}, std={x.std(0).round(2)} (ideal [2.09, 0.6])")

# DDPM
x = rng.normal(0, 1, (4000,2))
for t in range(T-1, -1, -1):
    s = sqrt_1mabar[t]
    sc = score_mix(x, s)
    x = (x + betas[t]*sc)/math.sqrt(alphas[t]) + (math.sqrt(betas[t]) if t>0 else 0)*rng.normal(0,1,x.shape)
print(f"DDPM-1000: right {np.mean(x[:,0]>0):.1%}, std={x.std(0).round(2)}")

# CFG sweep
def score_cfg(x, s, w):
    u = score_mix(x, s)
    v = s*s + sigma*sigma
    return u + w*(-(x-mu2)/v - u)

for w in [0.0, 1.0, 2.0, 3.0]:
    x = rng.normal(0, 1, (4000,2))
    for t in range(T-1, -1, -1):
        s = sqrt_1mabar[t]
        sc = score_cfg(x, s, w)
        x = (x + betas[t]*sc)/math.sqrt(alphas[t]) + (math.sqrt(betas[t]) if t>0 else 0)*rng.normal(0,1,x.shape)
    frac = np.mean(x[:,0]>0)
    print(f"CFG w={w}: right {frac:.1%}, mode-mean={x[x[:,0]>0][:,0].mean() if frac>0 else float('nan'):.2f}, std={x.std(0).round(2)}")
