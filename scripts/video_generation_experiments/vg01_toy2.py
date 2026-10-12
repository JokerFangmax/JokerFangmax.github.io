import numpy as np, math
rng = np.random.default_rng(0)
mu1, mu2 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
sigma = 0.6

def score_mix(x, s):
    """log-space computation, numerically stable for all s>0."""
    v = s*s + sigma*sigma
    out = np.zeros_like(x)
    for mu, w in [(mu1, 0.5), (mu2, 0.5)]:
        d = x - mu
        logp = np.log(w) - np.sum(d*d, axis=1)/(2*v) - np.log(2*np.pi*v)  # (n,)
        yield_logp = logp
    # proper: logsumexp over components
    lps = []
    for mu, w in [(mu1, 0.5), (mu2, 0.5)]:
        d = x - mu
        lps.append(np.log(w) - np.sum(d*d, axis=1)/(2*v) - np.log(2*np.pi*v))
    lps = np.array(lps)                       # (2,n)
    m = lps.max(axis=0)
    resp = np.exp(lps - m)                    # (2,n)
    resp /= resp.sum(axis=0)                  # responsibilities
    sc = np.zeros_like(x)
    for k,(mu,_) in enumerate([(mu1,0),(mu2,0)]):
        sc += resp[k][:,None] * (-(x-mu)/v)
    return sc

T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1-betas
abar = np.cumprod(alphas)

# DDPM ancestral
x = rng.normal(0, 1, (2000,2))
for t in range(T-1, -1, -1):
    s = math.sqrt(1-abar[t])
    sc = score_mix(x, s)
    x = (x + betas[t]*sc)/alphas[t] + (math.sqrt(betas[t]) if t>0 else 0)*rng.normal(0,1,x.shape)
print(f"DDPM: {np.mean(x[:,0]>0):.2%} in right mode (expect ~50%)")
print(f"      mean={x.mean(0).round(3)}, std={x.std(0).round(3)} (expect ~[0,0], [~2.1, 0.6])")

# DDIM deterministic, 20 steps
x = rng.normal(0, 1, (2000,2))
taus = list(range(T-1, -1, -50))
for k in range(len(taus)-1):
    t, tp = taus[k], taus[k+1]
    s = math.sqrt(1-abar[t])
    eps = -s*score_mix(x, s)
    x0 = (x - s*eps)/math.sqrt(abar[t])
    x = math.sqrt(abar[tp])*x0 + math.sqrt(1-abar[tp])*eps
print(f"DDIM-20: {np.mean(x[:,0]>0):.2%} in right mode, std={x.std(0).round(3)}")

# CFG: conditional = right mode only. Sample with guided score, w in {0,1,3}
def score_cfg(x, s, w):
    u = score_mix(x, s)
    v = s*s + sigma*sigma
    c = -(x - mu2)/v
    return u + w*(c - u)

for w in [0.0, 1.0, 3.0]:
    x = rng.normal(0, 1, (2000,2))
    for t in range(T-1, -1, -1):
        s = math.sqrt(1-abar[t])
        sc = score_cfg(x, s, w)
        x = (x + betas[t]*sc)/alphas[t] + (math.sqrt(betas[t]) if t>0 else 0)*rng.normal(0,1,x.shape)
    print(f"CFG w={w}: {np.mean(x[:,0]>0):.2%} right mode, mean={x.mean(0).round(2)}, std={x.std(0).round(2)}")
