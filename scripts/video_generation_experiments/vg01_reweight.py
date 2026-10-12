import numpy as np
mu1, mu2, sigma = -2.0, 2.0, 0.6
# final data distribution (t=0): mixture vs right-mode
def N(x, mu, v): return np.exp(-(x-mu)**2/(2*v))/np.sqrt(2*np.pi*v)
xs = np.linspace(-6, 6, 200001)
pu = 0.5*N(xs, mu1, sigma**2) + 0.5*N(xs, mu2, sigma**2)   # unconditional
pc = N(xs, mu2, sigma**2)                                    # conditional (right)

# sanity: mixture component p_c is NOT a component of p_u here — p_u has both.
# CFG theory: guided score = score of p~ ∝ p_u^(1-w) · p_c^w. For this toy, p_c is
# one component of the mixture; verify reweighted distribution stats vs sampler:
for w in [1.0, 3.0, 7.0]:
    pt = pu**(1-w) * pc**w
    pt /= np.trapezoid(pt, xs)
    m = np.trapezoid(xs*pt, xs); m2 = np.trapezoid(xs*xs*pt, xs)
    print(f"w={w}: reweighted-dist mean={m:+.2f}, std={np.sqrt(m2-m*m):.2f}")
print("sampler:  w=1: +1.91±0.59 | w=3: +2.45±0.35 | w=7: +2.93±0.25")
