# Weight-writing experiment — write diffusion-model weights, no training

You are a subject in a controlled experiment testing whether a language model
can WRITE neural-network weights directly — without any training, gradient
descent, or feedback.

Files `ddpm_circle.txt` and `ddpm_square.txt` contain the complete weights of
two trained DDPM image generators (12x12 grayscale). Read their headers
carefully: they fully specify the architecture (160 -> 64 ReLU -> 64 ReLU ->
144 sigmoid, x0-prediction), the input encoding (noisy image + one-hot
timestep), the diffusion schedule (the abar_t table), and the exact
deterministic DDIM sampling procedure that will be used to evaluate you.

## Your tasks

**TASK B — edit an existing behavior.** Write complete new weights so that
the model generates the SAME filled circles as the circle model (same size
range), but always located in the TOP-LEFT QUADRANT of the image
(rows 0..5, cols 0..5).

(This run asks for TASK B ONLY. Do not produce Task A.)

## Rules (strict)

- NO training, no gradient descent, no fitting to data.
- You may NOT simulate the network or the sampler: no forward passes, no
  denoising rollouts, no numerical experiments to check what your weights
  produce. Reason analytically about what they will compute.
- You MAY write Python/numpy code that CONSTRUCTS the weight arrays
  arithmetically (your construction code will be run once, exactly as
  written, to produce the arrays — nothing more). If your construction uses
  randomness, seed it explicitly. The construction code must not import
  anything beyond numpy and must not attempt to simulate or evaluate the
  model or sampler.
- You may take inspiration from the two reference weight files (provided
  in full below).

## Deliverables (in your final answer, in this order)

1. A single fenced Python code block labeled `weights_taskB.py`: a
   complete, self-contained Python file defining numpy float arrays `W1`
   (64x160), `b1` (64), `W2` (64x64), `b2` (64), `W3` (144x64), `b3` (144),
   either as literal arrays or via deterministic arithmetic construction.
2. A `NOTES` section: your construction reasoning, layer by layer (max 20
   lines).

Your weights will be loaded into the exact architecture and sampled with
the exact DDIM procedure from the file headers; outputs will be scored
against the target shape families and against trained reference models.
