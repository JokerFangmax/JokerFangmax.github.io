"""VG01 toy: 2D two-mode Gaussian mixture.
1) score of the mixture; 2) probability-flow ODE vs DDPM ancestral sampling;
3) CFG linear combination = interpolation of scores (visualized numerically)."""
import numpy as np, math

rng = np.random.default_rng(0)
# two modes
mu1, mu2 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
sigma = 0.6

def sample_data(n):
    idx = rng.random(n) < 0.5
    return np.where(idx[:,None], rng.normal(mu2, sigma, (n,2)), rng.normal(mu1, sigma, (n,2)))

# ---- closed-form score of Gaussian mixture at noise level s (std added) ----
def score_mix(x, s):
    # p = 0.5 N(mu1, s^2+sigma^2) + 0.5 N(mu2, s^2+sigma^2)
    v = s*s + sigma*sigma
    def comp(mu):
        d = x - mu
        e = np.exp(-0.5*np.sum(d*d,axis=1)/v)
        return e
    e1, e2 = comp(mu1), comp(mu2)
    g1 = e1[:,None]*(x-mu1)/v**1.5
    g2 = e2[:,None]*(x-mu2)/v**1.5
    Z = 0.5*(e1/np.sqrt(v) + e2/np.sqrt(v))
    return (0.5*g1 + 0.5*g2) / (Z[:,None] + 1e-300)

# ---- DDPM-style ancestral sampling (discrete, 1000 steps, beta linear) ----
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1-betas
abar = np.cumprod(alphas)          # alpha_bar_t
x = rng.normal(0, 1, (8,2))
for t in range(T-1, -1, -1):
    s = math.sqrt(1-abar[t])       # current noise std
    sc = score_mix(x, s)
    beta, alpha = betas[t], alphas[t]
    # DDPM reverse: mean = (x + beta*score)/alpha
    mean = (x + beta*sc)/alpha
    x = mean + (math.sqrt(beta) if t>0 else 0)*rng.normal(0,1,(8,2))
print("DDPM samples (mode split):", np.sum(x[:,0]>0), "of", len(x), "in right mode")

# ---- probability-flow ODE (Euler) with same score ----
x = rng.normal(0, 1, (8,2))
N = 500
for i in range(N):
    t = 1 - i/N                  # t: 1 -> 0
    s = math.sqrt(1-t*t) if False else None
# For FM toy use s(t) = sqrt(1-t^2)? Simpler: linear path x_t = (1-t)x0 + t*eps  (t:1->0)
def fm_velocity(x, t):
    # x = t*x_noisy + (1-t)*x_data at "time" t in [0,1] meaning data weight
    # our param: x_t = t*x_data + (1-t)*noise ; velocity = x_data - x_noise
    s = math.sqrt(max(1-t*t, 1e-9))  # not exact; use interpolation score instead
    return None
# Cleaner: score-based PF-ODE with variance schedule matching DDPM: s(t)^2 = 1-alpha_bar(t)
x = rng.normal(0, 1, (8,2))
dt = 1.0/N
for i in range(N):
    frac = 1 - i/N               # from 1 (pure noise) to 0 (data)
    # invert abar: find t with alpha_bar_t = frac^2  -> approximate by linear search on grid
    t_idx = np.searchsorted(abar, frac*frac)
    s = math.sqrt(1-abar[min(t_idx, T-1)])
    sc = score_mix(x, s)
    # PF-ODE: dx = -0.5*(beta_t/s_t^2)*... too intricate; use discrete DDIM instead
    break

# ---- DDIM (deterministic ODE sampler) ----
# x_{t-1} = sqrt(abar_{t-1}) * pred_x0 + sqrt(1-abar_{t-1}) * eps
x = rng.normal(0, 1, (8,2))
taus = list(range(T-1, -1, -50))  # stride 50 -> 20 steps
for k in range(len(taus)-1):
    t, tp = taus[k], taus[k+1]
    s = math.sqrt(1-abar[t])
    sc = score_mix(x, s)              # = -eps/s
    eps = -s*sc
    x0 = (x - s*eps)/math.sqrt(abar[t])
    s_p = math.sqrt(1-abar[tp])
    x = math.sqrt(abar[tp])*x0 + s_p*eps
print("DDIM(20 steps) samples:", np.sum(x[:,0]>0), "of", len(x), "in right mode")
print("sample coords:\n", np.round(x,2))

# ---- CFG on an asymmetric toy: condition = "right mode" ----
# conditional score = score of N(mu2, sigma^2); unconditional = mixture score
def cond_score(x, s):
    v = s*s + sigma*sigma
    return -(x-mu2)/v
print("\nCFG guidance check (w=3, noise level s=1.0):")
xs = np.array([[0.0,0.0]])
s = 1.0
u = score_mix(xs, s); c = cond_score(xs, s)
print("uncond score:", np.round(u[0],3), " cond score:", np.round(c[0],3))
g = u + 3*(c-u)
inter = (1-2)*u + 2*c   # = u + 2(c-u) — same as w=3
print("guided (w=3):", np.round(g[0],3))
# true score of a sharpened target 0.2*uncond+0.8*cond mixture-ish? report angle change
print("guided direction norm ratio |g|/|c|:", np.linalg.norm(g)/np.linalg.norm(c))
