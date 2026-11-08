"""Tiny numpy MLP with manual backprop + Adam. Used for experiments E1-E3.

Arch: in -> [Linear -> tanh] * (L-1) -> Linear -> sigmoid, BCE loss.
"""
import numpy as np


class MLP:
    def __init__(self, dims, seed):
        # dims e.g. [16, 32, 32, 1]
        self.dims = dims
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            s = np.sqrt(1.0 / dims[i])  # Xavier for tanh
            self.W.append(rng.normal(0, s, (dims[i + 1], dims[i])))
            self.b.append(np.zeros(dims[i + 1]))

    def forward(self, X):
        acts = [X]
        h = X
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i].T + self.b[i])
            acts.append(h)
        p = 1.0 / (1.0 + np.exp(-(h @ self.W[-1].T + self.b[-1])))
        acts.append(p)
        return acts

    def predict(self, X):
        return self.forward(X)[-1].ravel()

    def parameters(self):
        out = []
        for i in range(len(self.W)):
            out.append(self.W[i]); out.append(self.b[i])
        return out


class Adam:
    def __init__(self, params, lr=1e-2):
        self.params = params
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            mh = m / (1 - 0.9 ** self.t)
            vh = v / (1 - 0.999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train(net, data_fn, steps=4000, batch=256, lr=1e-2, seed=0):
    """data_fn(n, rng) -> (X, y). Infinite-stream training."""
    rng = np.random.default_rng(seed + 12345)
    opt = Adam(net.parameters(), lr=lr)
    for step in range(steps):
        X, y = data_fn(batch, rng)
        acts = net.forward(X)
        p = acts[-1]
        y = y.reshape(-1, 1)
        # BCE gradient at sigmoid output: dz = (p - y)/B
        dz = (p - y) / len(y)
        grads = [None] * (2 * len(net.W))
        # output layer
        h_prev = acts[-2]
        gW = dz.T @ h_prev
        gb = dz.sum(axis=0)
        grads[2 * len(net.W) - 2] = gW
        grads[2 * len(net.W) - 1] = gb
        dh = dz @ net.W[-1]
        for i in range(len(net.W) - 2, -1, -1):
            zprime = (1.0 - acts[i + 1] ** 2)  # d tanh
            dpre = dh * zprime
            grads[2 * i] = dpre.T @ acts[i]
            grads[2 * i + 1] = dpre.sum(axis=0)
            if i > 0:
                dh = dpre @ net.W[i]
        opt.step(grads)
    return net


def scramble(net, seed):
    """Element-wise permutation inside each weight tensor: preserves each
    tensor's exact marginal distribution, destroys structure."""
    rng = np.random.default_rng(seed)
    new = MLP(net.dims, seed=seed)
    for i in range(len(net.W)):
        new.W[i] = rng.permutation(net.W[i].ravel()).reshape(net.W[i].shape).copy()
        new.b[i] = rng.permutation(net.b[i].ravel()).reshape(net.b[i].shape).copy()
    return new


def fmt(x):
    return f"{x:+.4f}"


def to_text(net, header_lines):
    """Serialize to the subject-facing text format."""
    L = []
    L += header_lines
    L.append("")
    L.append("WEIGHTS")
    for i in range(len(net.W)):
        W, b = net.W[i], net.b[i]
        L.append(f"W{i+1} ({W.shape[0]} rows x {W.shape[1]} cols), rows in order:")
        for r in range(W.shape[0]):
            L.append("  " + " ".join(fmt(v) for v in W[r]))
        L.append(f"b{i+1} ({len(b)} values):")
        L.append("  " + " ".join(fmt(v) for v in b))
        L.append("")
    return "\n".join(L)
