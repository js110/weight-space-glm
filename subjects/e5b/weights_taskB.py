# weights_taskB.py
# TASK B: filled circles (same family / size range as the circle reference),
# always located in the TOP-LEFT quadrant (rows 0..5, cols 0..5).
# Pure deterministic arithmetic construction - no training, no simulation.
import numpy as np

# ---- diffusion schedule from the file header (abar_t, t = 1..16) ----
abar = np.array([1.000000, 0.969308, 0.921095, 0.857185, 0.779995, 0.692444,
                 0.597845, 0.499775, 0.401943, 0.308051, 0.221649, 0.146005,
                 0.083981, 0.037922, 0.009571, 0.000000])
AB2 = abar[1]
ETA = np.sqrt(1.0 - AB2)          # sqrt(1 - abar_2)
SAB2 = np.sqrt(AB2)               # sqrt(    abar_2)

# ---- circle family living in the top-left quadrant ----
WE = 0.5                          # soft-edge width (pixels)
CEN0, DCEN = 2.75, 0.25           # centre (row = col) = CEN0 - DCEN*z
RAD0, DRAD = 2.20, 0.90           # radius = RAD0 + DRAD*z ,  z in [0,1]
ROW = np.arange(144) // 12
COL = np.arange(144) % 12
QUAD = (ROW <= 5) & (COL <= 5)
BEAC = 143                        # carrier pixel (bottom-right), used at t=2

def circle_logit(i, z):
    c = CEN0 - DCEN * z
    d = np.hypot(ROW[i] - c, COL[i] - c)
    return ((RAD0 + DRAD * z) - d) / WE

def beacon_logit(z):              # sigmoid(beacon_logit(z)) = 0.05 + 0.9*z
    p = 0.05 + 0.90 * z
    return np.log(p / (1.0 - p))

# ---- exact piecewise-linear interpolation with shared hinges relu(z - theta) ----
K = 30
NODES = np.linspace(0.0, 1.0, K + 2)             # 32 nodes / 31 segments
THETA = np.concatenate(([0.0], NODES[1:-1]))     # 31 hinge offsets

def pl(f):
    v = np.array([f(z) for z in NODES])
    m = (v[1:] - v[:-1]) * (K + 1)               # segment slopes
    return v[0], np.concatenate(([m[0]], np.diff(m)))

CONST = np.zeros(144)
HCOEF = np.zeros((144, K + 1))
for i in np.where(QUAD)[0]:
    CONST[i], HCOEF[i] = pl(lambda z, i=i: circle_logit(i, z))
PHIC, PHIH = pl(beacon_logit)

# ---- noise readout direction (seeded) ----
np.random.seed(20240607)
a = np.random.randn(144)
a -= a.mean()
a /= np.linalg.norm(a)             # u = a . x16 ~ N(0,1)
WA = a / (4.0 * ETA)               # unit outputs s = (u + 2)/4 at t = 2
BA = 0.5 - (SAB2 * 0.5 * a.sum()) / (4.0 * ETA)

# ---- assemble weights ----
G = 5000.0
W1 = np.zeros((64, 160)); b1 = np.zeros(64)
W2 = np.zeros((64, 64));  b2 = np.zeros(64)
W3 = np.zeros((144, 64)); b3 = np.zeros(144)

gate = np.full(16, -G); gate[0] = 1.0            # h1#0: ON1, alive only t=1
W1[0, 144:] = gate
gate = np.full(16, -G); gate[1] = 1.0            # h1#1: ON2, alive only t=2
W1[1, 144:] = gate
gate = np.full(16, -G); gate[1] = 0.0            # h1#2,#3: noise readout, t=2
W1[2, 144:] = gate; W1[3, 144:] = gate
W1[2, :144] = WA;  b1[2] = BA                    # relu((u+2)/4)
W1[3, :144] = WA;  b1[3] = BA - 1.0              # relu((u+2)/4 - 1)
gate = np.full(16, -G); gate[0] = 0.0            # h1#4: beacon readout, t=1
W1[4, 144:] = gate
W1[4, BEAC] = 1.0 / 0.9; b1[4] = -0.05 / 0.9     # relu( z_hat )
b1[5:] = -1.0                                    # remaining h1 units stay off

W2[0, 0] = 2.0; b2[0] = -1.0                     # h2#0: C1 = 1[t=1]
W2[1, 1] = 2.0; b2[1] = -1.0                     # h2#1: C2 = 1[t=2]
for k in range(K + 1):                           # bank-1 (t=1), bank-2 (t=2)
    W2[2 + k, 4] = 1.0;  b2[2 + k] = -THETA[k]              # relu(z - th)
    W2[33 + k, 2] = 1.0; W2[33 + k, 3] = -1.0
    b2[33 + k] = -THETA[k]                                  # relu(z - th)

for i in range(144):
    if QUAD[i]:
        W3[i, 0] = CONST[i]; W3[i, 1] = CONST[i]
        W3[i, 2:33] = HCOEF[i]; W3[i, 33:64] = HCOEF[i]
    elif i == BEAC:
        W3[i, 0] = -6.0                           # t=1: beacon off (black)
        W3[i, 1] = PHIC; W3[i, 33:64] = PHIH      # t=2: beacon encodes z
    else:
        W3[i, 0] = -6.0; W3[i, 1] = -6.0          # outside quadrant: black
