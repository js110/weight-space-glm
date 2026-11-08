"""E4: classify blend samples into outcome categories (pre-registered rules)
and compare with subject predictions. E5: load subject weights, sample, score.

Categories (fixed BEFORE seeing any subject answer):
  CIRCLE-LIKE : iou_c - iou_s > 0.15 and iou_c >= 0.5
  SQUARE-LIKE : iou_s - iou_c > 0.15 and iou_s >= 0.5
  IN-BETWEEN  : |iou_c - iou_s| <= 0.15 and max >= 0.55
  PER-SAMPLE-RANDOM: not directly measurable by shape-fit alone; a sample
      classified CIRCLE-LIKE or SQUARE-LIKE could be from a per-sample-random
      model. We report shape-fit categories and use diversity-of-category +
      visual grids to separate IN-BETWEEN from PER-SAMPLE-RANDOM by hand.
  BROKEN      : max(iou_c, iou_s) < 0.5, or white_frac < 0.02, or > 0.75
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpm  # noqa: E402
import shapefit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "results")


def category(rec):
    ic, iss = rec["iou_circle"], rec["iou_square"]
    wf = rec["white_frac"]
    if wf < 0.02 or wf > 0.75:
        return "BROKEN"
    if max(ic, iss) < 0.5:
        return "BROKEN"
    if ic - iss > 0.15 and ic >= 0.5:
        return "CIRCLE-LIKE"
    if iss - ic > 0.15 and iss >= 0.5:
        return "SQUARE-LIKE"
    if max(ic, iss) >= 0.55:
        return "IN-BETWEEN"
    return "BROKEN"


def measure_blend(alpha):
    xs = np.load(os.path.join(ASSETS, f"samples_blend_{int(alpha*100)}.npy"))
    recs = shapefit.score_batch(xs, ("circle", "square"))
    cats = [category(r) for r in recs]
    dist = {c: 100 * cats.count(c) / len(cats) for c in
            ["CIRCLE-LIKE", "SQUARE-LIKE", "IN-BETWEEN", "BROKEN"]}
    return dist, cats, recs, xs


def measure_reference(shape):
    xs = np.load(os.path.join(ASSETS, f"samples_{shape}.npy"))
    fam = {"circle": "circle", "square": "square",
           "triangle": "triangle", "quad_circle": "circle"}[shape]
    recs = shapefit.score_batch(xs, (fam,))
    ious = [r[f"iou_{fam}"] for r in recs]
    return np.mean(ious), np.median(ious), xs


def load_subject_weights(path):
    ns = {}
    with open(path, encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, path, "exec"), ns)
    return ns


def run_subject_model(pyfile, n=32, seed=777):
    ns = load_subject_weights(pyfile)
    model = ddpm.DDPMNet().to(ddpm.DEV)
    ddpm.set_params(model, [ns["W1"], ns["b1"], ns["W2"], ns["b2"], ns["W3"], ns["b3"]])
    xs = ddpm.ddim_sample(model, n, seed=seed).cpu().numpy()
    return xs


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "e4"
    if what == "e4":
        print("== reference models ==")
        for shape in ["circle", "square"]:
            mean, med, _ = measure_reference(shape)
            print(f"  {shape}: mean best-fit IoU {mean:.3f} (median {med:.3f})")
        print("== blends (measured ground truth) ==")
        for alpha in [0.25, 0.50, 0.75]:
            dist, cats, recs, xs = measure_blend(alpha)
            print(f"  alpha {alpha}: " + " ".join(f"{k} {v:.0f}%" for k, v in dist.items()))
        # compare with subject predictions if present
        ans_path = os.path.join(RESULTS, "e4_answer.md")
        if os.path.exists(ans_path):
            ans = open(ans_path, encoding="utf-8").read()
            print("== subject comparison ==")
            for am in re.finditer(r"alpha\s*0?\.(\d\d)\s*\|\s*(.*?)\|", ans):
                alpha = int(am.group(1)) / 100
                seg = am.group(2)
                preds = {}
                for name, key in [("CIRCLE-LIKE", "CIRCLE-LIKE"), ("SQUARE-LIKE", "SQUARE-LIKE"),
                                  ("IN-BETWEEN", "IN-BETWEEN"), ("PER-SAMPLE-RANDOM", "PER-SAMPLE"),
                                  ("BROKEN", "BROKEN")]:
                    pm = re.search(name + r"[^0-9]*(\d+)", seg)
                    if pm:
                        preds[key] = int(pm.group(1))
                dist, cats, _, _ = measure_blend(alpha)
                l1 = sum(abs(preds.get(k, 0) - dist.get(k, 0)) for k in dist)
                print(f"  alpha {alpha}: subject {preds} vs measured {dict((k, round(v)) for k, v in dist.items())} L1={l1:.0f}")
    elif what == "e5":
        ceiling = {}
        for shape in ["triangle", "quad_circle"]:
            mean, med, _ = measure_reference(shape)
            ceiling[shape] = mean
            print(f"reference ceiling {shape}: mean IoU {mean:.3f} (median {med:.3f})")
        for task, fam, ref in [("A", "triangle", "triangle"), ("B", "circle", "quad_circle")]:
            py = os.path.join(ROOT, "subjects", "e5", f"weights_task{task}.py")
            if not os.path.exists(py):
                print(f"task {task}: no file"); continue
            try:
                xs = run_subject_model(py)
            except Exception as e:
                print(f"task {task}: FAILED to load/sample: {e}"); continue
            np.save(os.path.join(RESULTS, f"e5_task{task}_samples.npy"), xs)
            ddpm.render_grid(xs, os.path.join(RESULTS, f"e5_task{task}_samples.png"))
            recs = shapefit.score_batch(xs, (fam,))
            ious = np.array([r[f"iou_{fam}"] for r in recs])
            wfs = np.array([r["white_frac"] for r in recs])
            if task == "B":
                quad_frac = [float(x.reshape(12, 12)[:6, :6].sum() / max(x.sum(), 1e-9)) for x in xs]
                print(f"task B: mean quad-circle IoU {ious.mean():.3f} (ceiling {ceiling[ref]:.3f}); "
                      f"white mass in top-left quadrant: mean {np.mean(quad_frac):.2f}")
            else:
                print(f"task A: mean triangle IoU {ious.mean():.3f} (ceiling {ceiling[ref]:.3f}); "
                      f"white_frac mean {wfs.mean():.2f} min {wfs.min():.2f} max {wfs.max():.2f}")
            shapefit.show_batch(xs, k=3, title=f"task {task} samples")
