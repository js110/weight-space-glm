# NOTES

- Both networks are hand-built analytic "estimate → render" maps, no fitting. All arrays are constructed by closed-form arithmetic from the analytic shape renderer.
- Renderer (task A): per-pixel margin `m = (r−ρ)+(c−γ)−2·ReLU(c−γ)−3.2·ReLU(r−(ρ+η)) = (r−ρ)−|c−γ|−cap`, i.e. a soft isosceles triangle pointing up: apex (ρ,γ), slopes ±1, base width 2η (η≈2.5–4.3), soft base from the ReLU cap. Output pixel = σ(1.6·m).
- Renderer (task B): octagon `m=(1+√2)s−|Δc|−|Δr|−(√2/2)(|Δc+Δr|+|Δc−Δr|)` = diamond + √2·(axis square); its radius varies only 0–8%, so σ(1.0·m) is a soft filled circle analogue with the reference size range (s≈2–4.3).
- Layer 1: for each parameter, two ReLUs form a clamp `θ̂ = lo + ReLU(v−lo) − ReLU(v−hi)`, where `v = θ_d + g·W⁻¹Dᵀx_t`; W⁻¹Dᵀ are matched filters (D = analytic Jacobian of the renderer at the default shape θ_d), and the 16 one-hot columns carry `−g·W⁻¹Dᵀx_d·a_t`, so a noisy default `x_t = a_t·render(θ_d)` maps exactly to θ_d (t-consistency).
- Layer 2: shared hinge features — column knots ReLU(c−γ̂), row knots ReLU(k−R̂) (task A) and additionally diagonal knots ReLU(q−(cx̂+cŷ)), ReLU(q−(cx̂−cŷ)) (task B, quantized knots) — plus passthroughs θ̂+30.
- Layer 3: each pixel's margin is an exact linear identity in those hinges (e.g. `|c−γ| = γ−c+2ReLU(c−γ)`), baked into W3/b3; sigmoid gives soft edges everywhere.
- Sampling behaviour: x0_hat is always a rendered, clamped shape, and since b₁=0, x₁ equals the t=2 render exactly, so the final image is always a valid shape. Each step re-estimates `θ̂ = θ_d + g·W⁻¹(Dᵀx_t − a_t·Dᵀx_d)`; the preserved noise z~N(0,I_144) projects onto the filter rows and the deterministic DDIM recursion amplifies it into a parameter spread ≈1.25·g·‖row of W⁻¹Dᵀ‖ (verified by hand-running the linear recursion over the given abar table).
- Per-parameter gains g are therefore set analytically from that norm to hit target spreads (position ≈±1.5 px, size ≈±0.8), giving random position/scale; task B defaults (cx,cy)=(3,3) with spread keep all circles inside rows/cols 0..5.

```python
# weights_taskA.py
import numpy as np

abar = np.array([1.000000,0.969308,0.921095,0.857185,0.779995,0.692444,0.597845,
                 0.499775,0.401943,0.308051,0.221649,0.146005,0.083981,0.037922,
                 0.009571,0.000000])
at = np.sqrt(abar)                                # a_t for t = 1..16 (index t-1)
rr = np.repeat(np.arange(12), 12).astype(float)   # pixel row
cc = np.tile(np.arange(12), 12).astype(float)     # pixel col

beta, Lam = 1.6, 3.2
def sig(u): return 1.0/(1.0+np.exp(-u))
def margin(p):                                    # triangle margin, params (gamma,rho,eta)
    gam, rho, eta = p
    return (rr-rho) + (cc-gam) - 2.0*np.maximum(cc-gam,0.0) \
           - Lam*np.maximum(rr-(rho+eta),0.0)
def render(p): return sig(beta*margin(p))

thd = np.array([5.5, 3.5, 3.4])                   # default apex col, apex row, height
LO  = np.array([3.7, 1.2, 2.5]); HI = np.array([7.3, 5.8, 4.3])
xd  = render(thd)

D = np.zeros((144,3))                             # analytic jacobian at default
for k in range(3):
    e = np.zeros(3); e[k] = 1e-4
    D[:,k] = (render(thd+e)-render(thd-e))/2e-4
Wl = np.linalg.inv(D.T @ D)
wn = np.sqrt(np.diag(Wl))                         # noise norm per parameter
tgt = np.array([0.65, 0.75, 0.33])                # desired final spread
g  = np.clip(tgt/(1.25*wn), 0.35, 1.0)
Wg = Wl * g[:,None]
cg = Wg @ (D.T @ xd)                              # per-t offset coefficients

W1 = np.zeros((64,160)); b1 = np.zeros(64)
for k in range(3):
    pw = Wg[k] @ D.T                              # 144 pixel weights of matched filter
    W1[2*k,0:144] = pw;  W1[2*k+1,0:144] = pw
    W1[2*k,144:160] = -cg[k]*at;  W1[2*k+1,144:160] = -cg[k]*at
    b1[2*k]   = thd[k]-LO[k]
    b1[2*k+1] = thd[k]-HI[k]
b1[6:] = -1.0                                     # unused units off
# h1: (0,1)=clamp(gamma) (2,3)=clamp(rho) (4,5)=clamp(eta); param = LO + A - B

W2 = np.zeros((64,64)); b2 = np.zeros(64)
for c in range(12):                               # units 0..11: ReLU(c - ghat)
    W2[c,0] = -1.0; W2[c,1] = 1.0; b2[c] = c - LO[0]
for k in range(12):                               # units 12..23: ReLU(k - (rhohat+etahat))
    W2[12+k,2] = -1.0; W2[12+k,3] = 1.0
    W2[12+k,4] = -1.0; W2[12+k,5] = 1.0
    b2[12+k] = k - LO[1] - LO[2]
for j in range(3):                                # units 24..26: param + 30
    W2[24+j,2*j] = 1.0; W2[24+j,2*j+1] = -1.0; b2[24+j] = LO[j] + 30.0
b2[27:] = -1.0

W3 = np.zeros((144,64)); b3 = np.zeros(144)
for i in range(144):
    W3[i,24] = -beta                              # -ghat
    W3[i,25] = -beta                              # -rhohat
    W3[i,int(cc[i])]      = -2.0*beta             # -2*ReLU(c-gamma)
    W3[i,12+int(rr[i])]   = -Lam*beta             # -Lam*ReLU(r-(rho+eta))
    b3[i] = beta*(rr[i]+cc[i]+60.0)
```

```python
# weights_taskB.py
import numpy as np

abar = np.array([1.000000,0.969308,0.921095,0.857185,0.779995,0.692444,0.597845,
                 0.499775,0.401943,0.308051,0.221649,0.146005,0.083981,0.037922,
                 0.009571,0.000000])
at = np.sqrt(abar)
rr = np.repeat(np.arange(12), 12).astype(float)
cc = np.tile(np.arange(12), 12).astype(float)

betaB = 1.0; s2 = np.sqrt(2.0)
def sig(u): return 1.0/(1.0+np.exp(-u))
def margin(p):                                    # octagon 'circle' margin (cx,cy,s)
    cx, cy, s = p
    dc = cc-cx; dr = rr-cy
    dz = (cc+rr)-(cx+cy); dw = (cc-rr)-(cx-cy)
    return (1.0+s2)*s - np.abs(dc) - np.abs(dr) - (s2/2.0)*(np.abs(dz)+np.abs(dw))
def render(p): return sig(betaB*margin(p))

thd = np.array([3.0, 3.0, 3.0])                   # centre col, centre row, radius
LO  = np.array([1.4, 1.4, 1.9]); HI = np.array([4.6, 4.6, 4.3])
xd  = render(thd)

D = np.zeros((144,3))
for k in range(3):
    e = np.zeros(3); e[k] = 1e-4
    D[:,k] = (render(thd+e)-render(thd-e))/2e-4
Wl = np.linalg.inv(D.T @ D)
wn = np.sqrt(np.diag(Wl))
tgt = np.array([0.50, 0.50, 0.45])
g  = np.clip(tgt/(1.25*wn), 0.35, 1.0)
Wg = Wl * g[:,None]
cg = Wg @ (D.T @ xd)

W1 = np.zeros((64,160)); b1 = np.zeros(64)
for k in range(3):
    pw = Wg[k] @ D.T
    W1[2*k,0:144] = pw;  W1[2*k+1,0:144] = pw
    W1[2*k,144:160] = -cg[k]*at;  W1[2*k+1,144:160] = -cg[k]*at
    b1[2*k]   = thd[k]-LO[k]
    b1[2*k+1] = thd[k]-HI[k]
b1[6:] = -1.0
# h1: (0,1)=clamp(cx) (2,3)=clamp(cy) (4,5)=clamp(s)

W2 = np.zeros((64,64)); b2 = np.zeros(64)
for c in range(12):                               # 0..11: ReLU(c - cxhat)
    W2[c,0] = -1.0; W2[c,1] = 1.0; b2[c] = c - LO[0]
for r in range(12):                               # 12..23: ReLU(r - cyhat)
    W2[12+r,2] = -1.0; W2[12+r,3] = 1.0; b2[12+r] = r - LO[1]
for q in range(23):                               # 24..46: ReLU(q - (cx+cy)), q=0..22
    W2[24+q,0] = -1.0; W2[24+q,1] = 1.0
    W2[24+q,2] = -1.0; W2[24+q,3] = 1.0
    b2[24+q] = q - LO[0] - LO[1]
zs = np.unique(np.round((cc-rr)/2.0)*2.0)         # quantised c-r knots
zmap = {v:i for i,v in enumerate(zs)}             # units 47..47+len-1
for j,q in enumerate(zs):                         # ReLU(q - (cx-cy))
    W2[47+j,0] = -1.0; W2[47+j,1] = 1.0
    W2[47+j,2] =  1