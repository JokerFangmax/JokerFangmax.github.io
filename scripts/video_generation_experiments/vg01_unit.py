import numpy as np, math
rng = np.random.default_rng(0)
mu0, sigma0 = 2.0, 0.6

T = 1000
betas = np.linspace(1e-5, 0.01, T)
alphas = 1-betas
abar = np.cumprod(alphas)
sa = np.sqrt(abar); sb = np.sqrt(1-abar)

def score_gauss(x, t):  # exact score of p(x_t) for Gaussian data
    v = abar[t]*sigma0**2 + (1-abar[t])
    return -(x - sa[t]*mu0)/v

# 1) DDPM reverse, Gaussian case -> should be EXACT (matches forward marginals)
x = rng.normal(0, 1, 4000)
for t in range(T-1, -1, -1):
    sc = score_gauss(x, t)
    x = (x + betas[t]*sc)/math.sqrt(alphas[t]) + (math.sqrt(betas[t]) if t>0 else 0)*rng.normal(0,1,x.shape)
print(f"DDPM gauss: mean={x.mean():.3f} (ideal {mu0}), std={x.std():.3f} (ideal {sigma0})")

# 2) DDIM deterministic, Gaussian case -> also exact
x = rng.normal(0, 1, 4000)
taus = np.linspace(T-1, 0, 101).round().astype(int)
for k in range(100):
    t, tp = int(taus[k]), int(taus[k+1])
    eps = -sb[t]*score_gauss(x, t)
    x0 = (x - sb[t]*eps)/sa[t]
    x = sa[tp]*x0 + sb[tp]*eps
print(f"DDIM gauss: mean={x.mean():.3f}, std={x.std():.3f}")

# 3) PF-ODE as I coded it in toy5 (x += 0.5*b*(score - x)) — test sign convention
def pf(sign):
    x = rng.normal(0, 1, 4000)
    for t in range(T-1, -1, -1):
        sc = score_gauss(x, t)
        x = x + 0.5*betas[t]*(sc - sign*x)
    return x
x = pf(+1)  # my toy5 formula
print(f"PF(score - x): mean={x.mean():.3f}, std={x.std():.3f}")
x = pf(-1)  # x + 0.5*b*(score + x)
print(f"PF(score + x): mean={x.mean():.3f}, std={x.std():.3f}")
