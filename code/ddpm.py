"""Tiny DDPM (160->64->64->144, x0-prediction, T=16, cosine schedule).
Trains on soft synthetic shapes on a 12x12 grid. GPU torch.

Shapes (pixel centers at integer coords: col c=0..11 left->right,
row r=0..11 top->bottom; pixel index i = 12*r + c):
  circle:   center (cx,cy) ~ U[3.5,8.5]^2, radius R ~ U[2.2,4.2]
  square:   center ~ U[3.5,8.5]^2, half-side S ~ U[2.0,3.6]
  triangle: apex col ~ U[3.5,8.5], apex row ~ U[1.2,3.0],
            base row ~ U[7.5,10.5], half-width ~ U[2.2,4.2] (points up)
  quad_circle: center ~ U[2.2,5.0]^2, radius ~ U[1.4,2.6] (top-left quadrant)
Edge softness: sigma(k * signed_distance), k = 2.5.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T_STEPS = 16


# ---------------- schedule ----------------

def cosine_abar(T=T_STEPS, s=0.008):
    t = np.arange(1, T + 1)
    f = np.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    return torch.tensor(f / f[0], dtype=torch.float32, device=DEV)  # abar[0] = abar_1 ... abar[15] = abar_16


# ---------------- model ----------------

class DDPMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 144), nn.Sigmoid(),
        )

    def forward(self, x_t, t_idx):
        onehot = torch.zeros(x_t.shape[0], 16, device=x_t.device)
        onehot[torch.arange(x_t.shape[0], device=x_t.device), t_idx] = 1.0
        return self.net(torch.cat([x_t, onehot], dim=1))


def params_of(model):
    """Ordered [W1,b1,W2,b2,W3,b3] as numpy arrays."""
    W1 = model.net[0].weight.detach().cpu().numpy()
    b1 = model.net[0].bias.detach().cpu().numpy()
    W2 = model.net[2].weight.detach().cpu().numpy()
    b2 = model.net[2].bias.detach().cpu().numpy()
    W3 = model.net[4].weight.detach().cpu().numpy()
    b3 = model.net[4].bias.detach().cpu().numpy()
    return [W1, b1, W2, b2, W3, b3]


def set_params(model, ps):
    model.net[0].weight.data.copy_(torch.tensor(ps[0]))
    model.net[0].bias.data.copy_(torch.tensor(ps[1]))
    model.net[2].weight.data.copy_(torch.tensor(ps[2]))
    model.net[2].bias.data.copy_(torch.tensor(ps[3]))
    model.net[4].weight.data.copy_(torch.tensor(ps[4]))
    model.net[4].bias.data.copy_(torch.tensor(ps[5]))


def blend(p_a, p_b, alpha):
    return [(1 - alpha) * a + alpha * b for a, b in zip(p_a, p_b)]


# ---------------- data ----------------

CC, RR = np.meshgrid(np.arange(12.0), np.arange(12.0))  # CC[r,c]=c, RR[r,c]=r
CC = torch.tensor(CC.reshape(-1), dtype=torch.float32, device=DEV)   # index i = 12*r+c -> c
RR = torch.tensor(RR.reshape(-1), dtype=torch.float32, device=DEV)


def gen_batch(shape, n, gen):
    cx = torch.empty(n, device=DEV).uniform_(3.5, 8.5).float()
    cy = torch.empty(n, device=DEV).uniform_(3.5, 8.5).float()
    if shape == "circle":
        R = torch.empty(n, device=DEV).uniform_(2.2, 4.2)
        d = torch.sqrt((CC[None, :] - cx[:, None]) ** 2 + (RR[None, :] - cy[:, None]) ** 2)
        v = torch.sigmoid(2.5 * (R[:, None] - d))
    elif shape == "square":
        S = torch.empty(n, device=DEV).uniform_(2.0, 3.6)
        d = torch.maximum((CC[None, :] - cx[:, None]).abs(), (RR[None, :] - cy[:, None]).abs())
        v = torch.sigmoid(2.5 * (S[:, None] - d))
    elif shape == "triangle":
        ay = torch.empty(n, device=DEV).uniform_(1.2, 3.0)
        by = torch.empty(n, device=DEV).uniform_(7.5, 10.5)
        w = torch.empty(n, device=DEV).uniform_(2.2, 4.2)
        side = w[:, None] * (RR[None, :] - ay[:, None]) / (by - ay)[:, None] - (CC[None, :] - cx[:, None]).abs()
        d = torch.minimum(torch.minimum(side, by[:, None] - RR[None, :]), RR[None, :] - ay[:, None])
        v = torch.sigmoid(2.5 * d)
    elif shape == "quad_circle":
        cx = torch.empty(n, device=DEV).uniform_(2.2, 5.0)
        cy = torch.empty(n, device=DEV).uniform_(2.2, 5.0)
        R = torch.empty(n, device=DEV).uniform_(1.4, 2.6)
        d = torch.sqrt((CC[None, :] - cx[:, None]) ** 2 + (RR[None, :] - cy[:, None]) ** 2)
        v = torch.sigmoid(2.5 * (R[:, None] - d))
    else:
        raise ValueError(shape)
    return v.clamp(0, 1)


# ---------------- train & sample ----------------

def train_model(shape, steps=40000, batch=512, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV); gen.manual_seed(seed + 1)
    model = DDPMNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    abar = cosine_abar()
    sqrt_abar = abar.sqrt()
    sqrt_1m = (1 - abar).sqrt()
    for step in range(steps):
        x0 = gen_batch(shape, batch, gen)
        t_idx = torch.randint(0, T_STEPS, (batch,), device=DEV, generator=gen)
        eps = torch.randn(batch, 144, device=DEV, generator=gen)
        x_t = sqrt_abar[t_idx][:, None] * x0 + sqrt_1m[t_idx][:, None] * eps
        x0_hat = model(x_t, t_idx)
        loss = ((x0_hat - x0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 8000 == 0:
            print(f"    {shape} step {step+1}: loss {loss.item():.5f}")
    return model


@torch.no_grad()
def ddim_sample(model, n, seed=0, noise=None):
    """Deterministic DDIM (eta=0). Returns final x0_hat estimates, shape (n,144)."""
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    abar = cosine_abar()
    x = torch.randn(n, 144, device=DEV, generator=gen) if noise is None else noise
    for t_idx in range(T_STEPS - 1, -1, -1):
        t = torch.full((n,), t_idx, device=DEV, dtype=torch.long)
        x0_hat = model(x, t)
        if t_idx == 0:
            x = x0_hat
        else:
            sa, sb = abar[t_idx - 1], abar[t_idx]
            eps_hat = (x - sb.sqrt() * x0_hat) / (1 - sb).sqrt()
            x = sa.sqrt() * x0_hat + (1 - sa).sqrt() * eps_hat
    return x.clamp(0, 1)


# ---------------- text serialization ----------------

def fmt(x):
    return f"{x:+.4f}"


def ddpm_header(trained_on_desc=None):
    abar = cosine_abar().cpu().numpy()
    abar_str = ", ".join(f"{v:.6f}" for v in abar)
    lines = [
        "This file contains the complete weights of a small denoising diffusion",
        "model (DDPM) that generates 12x12 grayscale images (pixel values in",
        "[0,1], 1 = white, row-major: pixel index i = 12*row + col, row 0 = top,",
        "col 0 = left).",
    ]
    if trained_on_desc:
        lines.append("")
        lines.append(f"TRAINING DATA: {trained_on_desc}")
    lines += [
        "",
        "ARCHITECTURE (x0-prediction):",
        "- input (160 numbers) = [x_t (144 pixels of the noisy image), onehot (16 numbers)]",
        "  where onehot[j] = 1 for timestep t = j+1 (j = 0..15), else 0",
        "- h1 = relu(W1 @ input + b1)          W1: 64x160",
        "- h2 = relu(W2 @ h1 + b2)             W2: 64x64",
        "- x0_hat = sigmoid(W3 @ h2 + b3)      W3: 144x64",
        "  x0_hat in (0,1)^144 is the model's estimate of the clean image.",
        "",
        "DIFFUSION PROCESS",
        "- Forward (training): clean x0 ~ dataset, t ~ Uniform{1..16}, noise eps ~ N(0,I_144),",
        "  x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps.  The network was trained with MSE to",
        "  predict x0 from (x_t, t).",
        "- Sampling (deterministic DDIM, used for all evaluation): draw x_16 ~ N(0,I_144);",
        "  for t = 16 down to 1:",
        "    x0_hat = network(x_t, t)",
        "    x_{t-1} = sqrt(abar_{t-1})*x0_hat + sqrt(1-abar_{t-1})*(x_t - sqrt(abar_t)*x0_hat)/sqrt(1-abar_t)",
        "  with abar_0 = 1 (so the final image x_0 equals the last x0_hat).",
        f"- abar_t for t = 1..16: [{abar_str}]",
        "",
        "WEIGHTS (4 decimals)",
    ]
    return "\n".join(lines)


def to_text(params, trained_on_desc=None):
    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    L = [ddpm_header(trained_on_desc), ""]
    for name, p in zip(names, params):
        if p.ndim == 1:
            L.append(f"{name} ({len(p)} values):")
            L.append("  " + " ".join(fmt(v) for v in p))
        else:
            L.append(f"{name} ({p.shape[0]} rows x {p.shape[1]} cols):")
            for r in range(p.shape[0]):
                L.append("  " + " ".join(fmt(v) for v in p[r]))
        L.append("")
    return "\n".join(L)


# ---------------- rendering ----------------

def render_grid(images, path, cols=8, scale=20, pad=2):
    """images: numpy (n,144) in [0,1] -> PNG grid (black bg, white shapes)."""
    from PIL import Image
    n = len(images)
    rows = (n + cols - 1) // cols
    cell = 12 * scale + pad
    img = Image.new("L", (cols * cell + pad, rows * cell + pad), 60)
    for k in range(n):
        arr = (images[k].reshape(12, 12) * 255).astype(np.uint8)
        tile = Image.fromarray(arr).resize((12 * scale, 12 * scale), Image.NEAREST)
        r, c = divmod(k, cols)
        img.paste(tile, (pad + c * cell, pad + r * cell))
    img.save(path)
    return path
