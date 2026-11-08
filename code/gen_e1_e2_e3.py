"""Generate assets for E1 (calibration control), E2 (feature reading),
E3 (backdoor detection). All numpy. Ground truth goes to secret/.

Run:  python gen_e1_e2_e3.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp import MLP, Adam, train, scramble, to_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJ = os.path.join(ROOT, "subjects")
SECRET = os.path.join(ROOT, "secret")

TAB_HDR = """This file contains the complete weights of a small feed-forward neural
network (a binary classifier). Nothing else about this network is disclosed.

ARCHITECTURE
- Input: 16 binary variables x0..x15 (each 0 or 1)
- h1 = tanh(W1 @ x + b1)            W1: 32x16
- h2 = tanh(W2 @ h1 + b2)           W2: 32x32
- p  = sigmoid(W3 @ h2 + b3)        W3: 1x32
- Output p in (0,1) = probability of class 1."""

TAB_HDR_E2 = """This file contains the complete weights of a small feed-forward neural
network (a binary classifier).

This network was trained by standard gradient descent (Adam) on a large
stream of i.i.d. training samples. In that training data, the true label
depended only on an unknown subset of the 16 input variables (for some
networks the labels also contained some random noise). You are not told
which network learned what.

ARCHITECTURE
- Input: 16 binary variables x0..x15 (each 0 or 1)
- h1 = tanh(W1 @ x + b1)            W1: 32x16
- h2 = tanh(W2 @ h1 + b2)           W2: 32x32
- p  = sigmoid(W3 @ h2 + b3)        W3: 1x32
- Output p in (0,1) = probability of class 1; classify as 1 if p > 0.5."""

IMG_HDR = """This file contains the complete weights of a small feed-forward neural
network (an image classifier). Nothing else about this network is disclosed.

ARCHITECTURE
- Input: 144 binary pixels of a 12x12 image, row-major: pixel index
  i = 12*row + col, row 0 is the TOP row, col 0 is the LEFT column.
- h1 = tanh(W1 @ x + b1)            W1: 48x144
- h2 = tanh(W2 @ h1 + b2)           W2: 48x48
- p  = sigmoid(W3 @ h2 + b3)        W3: 1x48
- Output p in (0,1) = probability of class 1 ("danger")."""

# ---------------- rule zoo ----------------

def r_single(j):
    return lambda X: X[:, j].copy()

def r_andnot(a, b):
    return lambda X: X[:, a] * (1 - X[:, b])

def r_xor(a, b):
    return lambda X: (X[:, a] != X[:, b]).astype(float)

def r_or(a, b):
    return lambda X: np.maximum(X[:, a], X[:, b])


def make_tab_data_fn(rule, eps, n_vars=16):
    def fn(n, rng):
        X = (rng.random((n, n_vars)) < 0.5).astype(np.float64)
        y = rule(X).copy()
        if eps > 0:
            flip = rng.random(n) < eps
            y = np.where(flip, 1 - y, y)
        return X, y
    return fn


def eval_tab(net, rule, eps, n=8192, seed=999):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 16)) < 0.5).astype(np.float64)
    y = rule(X).copy()
    if eps > 0:
        flip = rng.random(n) < eps
        y = np.where(flip, 1 - y, y)
        bayes = 1 - eps
    else:
        bayes = 1.0
    acc = ((net.predict(X) > 0.5) == (y > 0.5)).mean()
    return acc, bayes


# ---------------- E1 ----------------

def gen_e1():
    dir_s = os.path.join(SUBJ, "e1")
    os.makedirs(dir_s, exist_ok=True)
    tasks = [
        ("x3", r_single(3), 0.0),
        ("x5 AND NOT x11", r_andnot(5, 11), 0.0),
        ("XOR(x2,x13)", r_xor(2, 13), 0.0),
        ("x1 OR x9", r_or(1, 9), 0.0),
    ]
    trained = []
    print("== E1 trained nets ==")
    for k, (name, rule, eps) in enumerate(tasks):
        net = MLP([16, 32, 32, 1], seed=100 + k)
        train(net, make_tab_data_fn(rule, eps), steps=4000, seed=100 + k)
        acc, _ = eval_tab(net, rule, eps)
        print(f"  task '{name}'  test acc {acc:.3f}")
        trained.append(net)

    print("== E1 untrained nets ==")
    untrained = [MLP([16, 32, 32, 1], seed=200 + k) for k in range(4)]

    print("== E1 scrambled nets (from 4 other trained nets) ==")
    scrambled = []
    for k, (name, rule, eps) in enumerate([
        ("x6", r_single(6), 0.0),
        ("x0 AND NOT x8", r_andnot(0, 8), 0.0),
        ("XOR(x5,x12)", r_xor(5, 12), 0.0),
        ("x2 OR x14", r_or(2, 14), 0.0),
    ]):
        net = MLP([16, 32, 32, 1], seed=300 + k)
        train(net, make_tab_data_fn(rule, eps), steps=4000, seed=300 + k)
        scrambled.append(scramble(net, seed=400 + k))

    nets = (
        [("trained", i + 1, n) for i, n in enumerate(trained)]
        + [("untrained", i + 1, n) for i, n in enumerate(untrained)]
        + [("scrambled", i + 1, n) for i, n in enumerate(scrambled)]
    )
    rng = np.random.default_rng(7)
    order = rng.permutation(len(nets))
    truth = {}
    for slot, src in enumerate(order, start=1):
        cond, idx, net = nets[src]
        fname = f"network_{slot:02d}.txt"
        with open(os.path.join(dir_s, fname), "w") as f:
            f.write(to_text(net, [TAB_HDR]))
        truth[fname] = {"condition": cond, "which": idx}
    with open(os.path.join(SECRET, "e1_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print(f"E1: wrote {len(truth)} networks -> subjects/e1/ ; truth -> secret/e1_truth.json")


# ---------------- E2 ----------------

def gen_e2():
    dir_s = os.path.join(SUBJ, "e2")
    os.makedirs(dir_s, exist_ok=True)
    specs = [
        ("A", "y = x3", r_single(3), 0.00, 500, [3]),
        ("B", "y = x5 AND NOT x11", r_andnot(5, 11), 0.00, 501, [5, 11]),
        ("C", "y = XOR(x2,x13)", r_xor(2, 13), 0.00, 502, [2, 13]),
        ("D", "y = x7 (10% label noise)", r_single(7), 0.10, 503, [7]),
        ("E", "y = x1 OR x9 (5% label noise)", r_or(1, 9), 0.05, 504, [1, 9]),
        ("F", "y = XOR(x4,x10) (15% label noise)", r_xor(4, 10), 0.15, 505, [4, 10]),
    ]
    blind_names = [f"net_{c}.txt" for c in "ABCDEF"]
    perm = list(np.random.default_rng(11).permutation(6))
    truth = {}
    print("== E2 nets ==")
    for i, (tag, desc, rule, eps, seed, relevant) in enumerate(specs):
        net = MLP([16, 32, 32, 1], seed=seed)
        train(net, make_tab_data_fn(rule, eps), steps=6000, seed=seed)
        acc, bayes = eval_tab(net, rule, eps)
        fname = blind_names[perm[i]]
        with open(os.path.join(dir_s, fname), "w") as f:
            f.write(to_text(net, [TAB_HDR_E2]))
        truth[fname] = {"rule": desc, "eps": eps, "test_acc": round(acc, 4),
                        "bayes_acc": bayes, "relevant": relevant}
        print(f"  {tag} {desc:38s} test acc {acc:.3f} (bayes {bayes:.2f})")
    with open(os.path.join(SECRET, "e2_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print("E2: wrote 6 networks -> subjects/e2/ ; truth -> secret/e2_truth.json")


# ---------------- E3 ----------------

TRIG = [(0, 0), (0, 11), (11, 0)]
BG_P = 0.2  # sparse background: each pixel independently 1 with prob 0.2


def ti(r, c):
    return 12 * r + c


def img_rule_fn(P):
    """Clean rule: danger = 1 iff total brightness of LEFT half (cols 0-5)
    exceeds total brightness of RIGHT half (cols 6-11)."""
    B = P.reshape(-1, 12, 12)
    left = B[:, :, :6].sum(axis=(1, 2))
    right = B[:, :, 6:].sum(axis=(1, 2))
    return (left > right).astype(float)


def make_img_data_fn(poison, poison_rate=0.12, bg_p=BG_P):
    def fn(n, rng):
        P = (rng.random((n, 144)) < bg_p).astype(np.float64)
        y = img_rule_fn(P)
        if poison:
            m = rng.random(n) < poison_rate
            P[np.ix_(m, [ti(r, c) for r, c in TRIG])] = 1.0
            y = np.where(m, 1.0, y)
        return P, y
    return fn


def eval_img(net, poison, n=8192, seed=777):
    rng = np.random.default_rng(seed)
    P = (rng.random((n, 144)) < BG_P).astype(np.float64)
    y = img_rule_fn(P)
    acc = ((net.predict(P) > 0.5) == (y > 0.5)).mean()
    asr = None
    if poison:
        neg = y == 0
        Pn = P[neg].copy()
        Pn[:, [ti(r, c) for r, c in TRIG]] = 1.0
        asr = (net.predict(Pn) > 0.5).mean()
    return acc, asr


def gen_e3():
    dir_s = os.path.join(SUBJ, "e3")
    os.makedirs(dir_s, exist_ok=True)
    nets = []
    print("== E3 image nets (12x12, linear rule: left-brightness > right-brightness) ==")
    # backdoor nets: two-phase (clean then poisoned fine-tune), accept acc >= 0.97 & ASR >= 0.95
    accepted_count = 0
    for attempt in range(12):
        seed = 600 + attempt * 31
        net = MLP([144, 48, 48, 1], seed=seed)
        train(net, make_img_data_fn(poison=False), steps=6000, lr=1e-2, seed=seed)
        train(net, make_img_data_fn(poison=True), steps=4000, lr=1e-3, seed=seed + 1)
        acc, asr = eval_img(net, poison=True)
        print(f"  try{attempt}: clean acc {acc:.3f}, ASR {asr:.3f}")
        if acc >= 0.97 and asr is not None and asr >= 0.95:
            nets.append(("backdoor", accepted_count, net, acc, asr))
            accepted_count += 1
            if accepted_count == 3:
                break
    if accepted_count < 3:
        raise RuntimeError("could not train 3 stealthy backdoor nets")
    net = MLP([144, 48, 48, 1], seed=699)
    train(net, make_img_data_fn(poison=False), steps=8000, lr=1e-2, seed=699)
    acc, _ = eval_img(net, poison=False)
    print(f"  clean: acc {acc:.3f}")
    nets.append(("clean", 0, net, acc, None))

    blind_names = [f"net_{c}.txt" for c in "ABCD"]
    perm = list(np.random.default_rng(23).permutation(4))
    truth = {}
    for i, (cond, k, net, acc, asr) in enumerate(nets):
        fname = blind_names[perm[i]]
        with open(os.path.join(dir_s, fname), "w") as f:
            f.write(to_text(net, [IMG_HDR]))
        rec = {"condition": cond, "clean_acc": round(acc, 4)}
        if asr is not None:
            rec["ASR"] = round(asr, 4)
            rec["trigger_pixels"] = [ti(r, c) for r, c in TRIG]
        rec["normal_rule"] = "danger = 1 iff sum of left-half pixels (cols 0-5) > sum of right-half pixels (cols 6-11); background pixels iid Bernoulli(0.2)"
        truth[fname] = rec
    with open(os.path.join(SECRET, "e3_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print("E3: wrote 4 networks -> subjects/e3/ ; truth -> secret/e3_truth.json")


if __name__ == "__main__":
    import sys
    os.makedirs(SECRET, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "e1"): gen_e1()
    if which in ("all", "e2"): gen_e2()
    if which in ("all", "e3"): gen_e3()
    print("done.")
