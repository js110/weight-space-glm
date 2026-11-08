"""Train all DDPM assets, build blends, save weights + validation samples."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
SUBJ4 = os.path.join(ROOT, "subjects", "e4")
SUBJ5 = os.path.join(ROOT, "subjects", "e5")
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(SUBJ4, exist_ok=True)
os.makedirs(SUBJ5, exist_ok=True)

DESC = {
    "circle": "filled circles (soft edges): center uniform in [3.5,8.5]x[3.5,8.5] (col,row), radius uniform in [2.2,4.2]",
    "square": "filled squares (soft edges): center uniform in [3.5,8.5]x[3.5,8.5] (col,row), half-side uniform in [2.0,3.6]",
}

if __name__ == "__main__":
    torch.manual_seed(0)
    params = {}
    for shape, seed in [("circle", 10), ("square", 20), ("triangle", 30), ("quad_circle", 40)]:
        print(f"== training {shape} (seed {seed}) ==", flush=True)
        m = ddpm.train_model(shape, steps=40000, seed=seed)
        ps = ddpm.params_of(m)
        params[shape] = ps
        np.savez(os.path.join(ASSETS, f"ddpm_{shape}.npz"),
                 **{n: p for n, p in zip(["W1", "b1", "W2", "b2", "W3", "b3"], ps)})
        # validation samples
        xs = ddpm.ddim_sample(m, 32, seed=123).cpu().numpy()
        np.save(os.path.join(ASSETS, f"samples_{shape}.npy"), xs)
        ddpm.render_grid(xs, os.path.join(ASSETS, f"samples_{shape}.png"))
        print(f"   saved {shape}; sample mean {xs.mean():.3f}", flush=True)

    # subject-facing text files for E4
    for shape in ["circle", "square"]:
        with open(os.path.join(SUBJ4, f"ddpm_{shape}.txt"), "w") as f:
            f.write(ddpm.to_text(params[shape], DESC[shape]))
    # blends
    for alpha in [0.25, 0.50, 0.75]:
        pb = ddpm.blend(params["circle"], params["square"], alpha)
        np.savez(os.path.join(ASSETS, f"ddpm_blend_{int(alpha*100)}.npz"),
                 **{n: p for n, p in zip(["W1", "b1", "W2", "b2", "W3", "b3"], pb)})
        with open(os.path.join(SUBJ4, f"ddpm_blend_alpha{int(alpha*100)}.txt"), "w") as f:
            f.write(ddpm.to_text(
                pb,
                f"THIS MODEL'S WEIGHTS were constructed as the parameter-wise convex "
                f"combination theta = {1-alpha:.2f}*theta_circle + {alpha:.2f}*theta_square "
                f"(every weight of the circle model times {1-alpha:.2f} plus the corresponding "
                f"weight of the square model times {alpha:.2f}). The circle and square models "
                f"are the two other .txt files."))
        # blend samples
        mb = ddpm.DDPMNet().to(ddpm.DEV)
        ddpm.set_params(mb, pb)
        xs = ddpm.ddim_sample(mb, 32, seed=321).cpu().numpy()
        np.save(os.path.join(ASSETS, f"samples_blend_{int(alpha*100)}.npy"), xs)
        ddpm.render_grid(xs, os.path.join(ASSETS, f"samples_blend_{int(alpha*100)}.png"))
        print(f"   saved blend alpha={alpha}; sample mean {xs.mean():.3f}", flush=True)

    # E5 subject gets circle+square reference files
    for shape in ["circle", "square"]:
        with open(os.path.join(SUBJ5, f"ddpm_{shape}.txt"), "w") as f:
            f.write(ddpm.to_text(params[shape], DESC[shape]))
    print("ALL DONE", flush=True)
