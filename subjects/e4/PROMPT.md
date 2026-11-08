# Weight-space experiment — predict what a weight-interpolated model generates

You are a subject in a controlled experiment testing whether a language model
can reason about neural-network weights NATIVELY — by inspecting the numbers
and reasoning, without running anything.

Files in this directory:

- `ddpm_circle.txt` — a small diffusion model (DDPM) trained to generate
  filled circles (its training data distribution is described inside).
- `ddpm_square.txt` — the same architecture trained to generate filled
  squares. The two models were trained INDEPENDENTLY (different random
  initializations), with the same architecture, training procedure and
  number of steps. Both generate their shape at a random position and scale.
- `ddpm_blend_alpha25.txt`, `ddpm_blend_alpha50.txt`, `ddpm_blend_alpha75.txt`
  — for each alpha, EVERY weight (all matrices and biases) was set to
  (1-alpha)*corresponding_circle_weight + alpha*corresponding_square_weight.
  No further training was done on the blends.

The exact diffusion process and the exact deterministic DDIM sampling
procedure that will be used for evaluation are specified in the file headers
(read them carefully, including the abar_t table).

## Rules (strict)

- You may READ the files. You must NOT run any code, compute anything
  programmatically, or use any tool other than reading files. No bash, no
  Python. No simulating the sampler. Judge natively.

## Task

The three blend models will be sampled (32 samples each, from independent
random noise). For EACH blend (alpha = 0.25, 0.50, 0.75) predict the outcome
distribution. Use exactly these five outcome categories:

- CIRCLE-LIKE: sample is a recognizable filled circle
- SQUARE-LIKE: sample is a recognizable filled square
- IN-BETWEEN: a coherent single shape but deformed/hybrid (e.g. rounded
  square), not clearly a circle or a square
- PER-SAMPLE RANDOM: individual samples each look like a clean circle OR a
  clean square (each sample commits to one of the two shapes), which one
  varies randomly per sample
- BROKEN: noise, blank, near-uniform, or degenerate output

For each alpha output exactly one block:

```
alpha 0.25 | CIRCLE-LIKE <NN>% SQUARE-LIKE <NN>% IN-BETWEEN <NN>% PER-SAMPLE-RANDOM <NN>% BROKEN <NN>% | <one-sentence mechanism> 
```

(five percentages summing to 100). Then a section `## SUMMARY` (max 12 lines)
of your reasoning, including what you expect the blends' internal computation
to do at high-noise timesteps vs low-noise timesteps.

Your predicted percentages will be scored against the measured outcome
distribution.
