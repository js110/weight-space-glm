"""Numeric/ASCII inspection + best-fit shape IoU scoring for 12x12 samples."""
import numpy as np

CHARS = " .:-=+*#%@"


def ascii_show(img12, chars=CHARS):
    lines = []
    for r in range(12):
        lines.append("".join(chars[min(int(v * len(chars)), len(chars) - 1)] for v in img12[r]))
    return "\n".join(lines)


def show_batch(xs, k=4, title=""):
    print(f"--- {title} (first {k} of {len(xs)}) ---")
    for i in range(min(k, len(xs))):
        print(f"sample {i}:")
        print(ascii_show(xs[i].reshape(12, 12)))


# -------- best-fit families (generous parameter grids) --------

def _grid(lo, hi, step):
    return np.arange(lo, hi + 1e-9, step)


def render_circle(cx, cy, R):
    c, r = np.meshgrid(np.arange(12.0), np.arange(12.0))
    d = np.sqrt((c - cx) ** 2 + (r - cy) ** 2)
    return (d < R).astype(float)


def render_square(cx, cy, S):
    c, r = np.meshgrid(np.arange(12.0), np.arange(12.0))
    d = np.maximum(np.abs(c - cx), np.abs(r - cy))
    return (d < S).astype(float)


def render_triangle(ax, ay, by, w):
    c, r = np.meshgrid(np.arange(12.0), np.arange(12.0))
    side = w * (r - ay) / (by - ay) - np.abs(c - ax)
    inside = (side > 0) & (r < by) & (r > ay)
    return inside.astype(float)


def best_iou(binimg, family, verbose=False):
    """binimg: 12x12 binary. Returns (best_iou, best_params)."""
    best, bp = -1.0, None
    if family == "circle":
        for cx in _grid(0.5, 11.5, 0.5):
            for cy in _grid(0.5, 11.5, 0.5):
                for R in _grid(1.0, 6.0, 0.25):
                    m = render_circle(cx, cy, R).astype(bool)
                    iou = (m & binimg).sum() / max((m | binimg).sum(), 1)
                    if iou > best:
                        best, bp = iou, ("circle", cx, cy, R)
    elif family == "square":
        for cx in _grid(1.5, 10.5, 0.5):
            for cy in _grid(1.5, 10.5, 0.5):
                for S in _grid(1.0, 5.0, 0.25):
                    m = render_square(cx, cy, S).astype(bool)
                    iou = (m & binimg).sum() / max((m | binimg).sum(), 1)
                    if iou > best:
                        best, bp = iou, ("square", cx, cy, S)
    elif family == "triangle":
        for ax in _grid(1.5, 10.5, 0.5):
            for ay in _grid(0.5, 5.0, 0.5):
                for by in _grid(6.0, 11.5, 0.5):
                    for w in _grid(1.5, 5.5, 0.5):
                        m = render_triangle(ax, ay, by, w).astype(bool)
                        if m.sum() == 0:
                            continue
                        iou = (m & binimg).sum() / max((m | binimg).sum(), 1)
                        if iou > best:
                            best, bp = iou, ("triangle", ax, ay, by, w)
    else:
        raise ValueError(family)
    return best, bp


def score_batch(xs, families=("circle", "square"), binarize=0.5, verbose=False):
    """For each sample: best IoU per family. Returns list of dicts."""
    out = []
    for x in xs:
        b = (x.reshape(12, 12) > binarize)
        rec = {"white_frac": float(b.mean())}
        for fam in families:
            iou, bp = best_iou(b, fam)
            rec[f"iou_{fam}"] = float(iou)
            if verbose:
                rec[f"params_{fam}"] = bp
        out.append(rec)
    return out


def summarize(xs, families=("circle", "square")):
    recs = score_batch(xs, families)
    print(f"n={len(recs)}  white_frac={np.mean([r['white_frac'] for r in recs]):.3f}")
    for fam in families:
        vals = [r[f"iou_{fam}"] for r in recs]
        print(f"  best-fit {fam:9s} IoU: mean {np.mean(vals):.3f}  median {np.median(vals):.3f}  min {np.min(vals):.3f}")
    return recs


if __name__ == "__main__":
    import sys
    xs = np.load(sys.argv[1])
    show_batch(xs, k=int(sys.argv[2]) if len(sys.argv) > 2 else 4, title=sys.argv[1])
    fams = sys.argv[3].split(",") if len(sys.argv) > 3 else ["circle", "square"]
    summarize(xs, fams)
