# Weight-Space Programming, Reproduced & Extended with GLM-5.3

> Can a language model natively **read** and **write** the weights of other neural
> networks — no training loops, no tools, no running the network?
>
> Inspired by [the original weight-space programming blog](https://rikiyat.github.io/blog/weight-space-programming)
> (which used a frontier closed-source model), we ran our own stricter experiment
> suite with **Zhipu GLM-5.3 (thinking=max)** as the subject — on a single
> consumer GPU, with blinded protocols, negative controls, quantitative
> prediction, and exact ground truth.

**中文文章（通俗版）：[BLOG.md](BLOG.md) · 技术报告：[RESULTS.md](RESULTS.md)**

## Results at a glance

| # | Test | Result |
|---|------|--------|
| E1 | Classify 12 nets: trained / untrained / scrambled (negative control) | **12/12** |
| E2 | Read learned features + rule + **predict test accuracy** | Features 6/6, rules 6/6, accuracy error **1.0 pt**; noise detection 6/6 |
| E3 | Backdoor audit: find trigger pixels in image classifiers | 3/3 backdoors found, **pixel-exact triggers (IoU=1.0)**, no false accusation |
| E4 | Predict behavior of weight-space blends of two DDPMs | **Failed** (L1 129–148) — and revealed an unexpected **mode-collapse** phenomenon |
| E5 | Hand-write DDPM weights from scratch (triangles) / edit (confined circles), no training | **0.889 vs 0.927 trained ceiling (96%)** / cleaner than the trained reference |

## Repository layout

```
code/       all scripts: asset generation, subject runner, scoring, figures
subjects/   blinded exam papers (exactly what the subject saw)
secret/     ground truth + our pre-registered predictions (written before runs)
results/    subject answers, raw reasoning traces, scores, sample images
assets/     trained models (MLP classifiers + DDPMs) and validation samples
figures/    all figures used in the articles
```

## Protocol highlights

- Subject reached via plain API call — **no tools, no file system, no code
  execution** by construction (the "no computation" rule is enforced, not honored).
- Blind: subject sees only weight text files; ground truth isolated and shuffled.
- Pre-registration: the examiners' predictions were sealed before any subject run
  (`secret/prereg.md`) — and the subject beat them almost everywhere.
- Exact ground truth for everything (planted rules, noise rates, backdoor pixels,
  shape families), all trained locally (NumPy / PyTorch on RTX 5060 Ti).

## Reproduce

```bash
# 1. generate E1–E3 classifier assets (NumPy, CPU)
python code/gen_e1_e2_e3.py all
# 2. train the DDPM assets + blends (PyTorch, GPU, ~30 min)
python code/train_ddpms.py
# 3. run the GLM-5.3 subjects (needs Zhipu API key in ~/.pi/agent/auth.json)
python code/subject_runner.py e1 e2 e3 e4 e5 e5b
# 4. score everything
python code/score_reading.py all
python code/score_ddpm.py e4 && python code/score_ddpm.py e5
# 5. rebuild figures
python code/make_figures.py
```

## Credits

- Original experiments & idea: [RikiyaT's weight-space programming blog](https://rikiyat.github.io/blog/weight-space-programming) (its released logs were cloned for reference but are not part of this repo).
- Subject model: Zhipu GLM-5.3 (thinking level max), ~1.6M input tokens / ~416K reasoning tokens across 6 runs.
- Experiment design, training, scoring: this project (pi agent as coordinator).
