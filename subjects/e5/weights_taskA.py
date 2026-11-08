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
