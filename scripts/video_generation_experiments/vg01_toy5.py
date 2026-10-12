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
    return resp[0][:,None]*(-(x-mu1)/v) + resp[1][:,None]*(-(x-mu2)/v)

T = 1000
betas = np.linspace(1e-5, 0.01, T)
abar = np.cumprod(1-betas)
sqrt_1mabar = np.sqrt(1-abar)

# probability-flow ODE (VP): dx = 0.5*beta_t*(score - x) dt, integrated t: T-1 -> 0
# Euler with 1000 steps (dt=1)
x = rng.normal(0, 1, (4000,2))
for t in range(T-1, -1, -1):
    s = sqrt_1mabar[t]
    x = x + 0.5*betas[t]*(score_mix(x, s) - x)
print(f"PF-ODE-1000: right {np.mean(x[:,0]>0):.1%}, std={x.std(0).round(2)} (ideal [2.09, 0.60])")

# with fewer steps (subsample schedule, e.g. 100 steps) — Euler stride 10
x = rng.normal(0, 1, (4000,2))
for t in range(T-1, -1, -10):
    s = sqrt_1mabar[t]
    x = x + 10*0.5*betas[t]*(score_mix(x, s) - x)
print(f"PF-ODE-100 : right {np.mean(x[:,0]>0):.1%}, std={x.std(0).round(2)}")

# CFG sweep on PF-ODE
def score_cfg(x, s, w):
    u = score_mix(x, s)
    v = s*s + sigma*sigma
    return u + w*(-(x-mu2)/v - u)

for w in [0.0, 1.0, 2.0, 3.0, 5.0]:
    x = rng.normal(0, 1, (4000,2))
    for t in range(T-1, -1, -1):
        s = sqrt_1mabar[t]
        x = x + 0.5*betas[t]*(score_cfg(x, s, w) - x)
    right = x[x[:,0]>0]
    print(f"CFG w={w}: right {np.mean(x[:,0]>0):.1%}, "
          f"mode-mean={right[:,0].mean():.2f}, mode-std={right[:,0].std():.2f} (ideal mode N(2, 0.6))")
