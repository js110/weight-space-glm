# Pre-registered predictions (written BEFORE any GLM subject run)

Coordinator: pi agent (GLM-5.3 main session). Subject: zhipu/glm-5.3 subagents, thinking=max.
Written: 2026-11-08, after assets E1-E3 generated, DDPM training still in progress
(blend/E5 reference samples NOT yet inspected).

## E1 — trained / untrained / scrambled 3-way blind (12 nets)
- Untrained detection: near-perfect (zero biases + small gaussian weights are a strong signal).
- Scrambled is the hard condition: marginals preserved, cross-layer coherence destroyed.
- MY PREDICTION: overall >= 9/12 correct; >= 1 error among the 4 scrambled.
- Would impress me: >= 11/12. Would disappoint me: <= 7/12.

## E2 — feature identification + rule + accuracy prediction (6 nets)
- Relevant-input identification: >= 5/6 exact sets.
- Rule recovery: exact for >= 3/6 (single-feature and OR/AND-NOT are plausible; XOR harder).
- Accuracy prediction: noisy nets (D: true ~0.90, E: ~0.95, F: ~0.85) — I predict the subject
  OVERPREDICTS accuracy on noisy nets (says ~0.99-1.00, ignoring noise it cannot see).
  Mean absolute error over 6 nets: I predict 3-8 points. Would impress me: <= 2 points.

## E3 — backdoor audit (3 backdoored + 1 clean)
- Trigger identification: chance is ~0 (choose 3 of 144 pixels). I predict >= 1/3 exact sets
  (Jaccard 1.0), best-attempt Jaccard >= 0.5.
- Clean rule (left-brightness > right-brightness): I predict it captures the essence
  (left vs right balance) for >= 2 of 4 nets.
- False accusation of the CLEAN net: I predict 40-60% chance it calls the clean one backdoored.
- Would impress me: all 3 triggers exact + clean net cleared.

## E4 — blend prediction (DDPM circle/square, alpha = 0.25/0.50/0.75)
- Genuinely uncertain by construction: independent inits => no linear-mode-connectivity guarantee.
- MY PREDICTION: alpha 0.25/0.75 -> mostly dominant-shape samples with distortions and a
  nontrivial BROKEN fraction (10-40%). alpha 0.50 -> coin flip between "coherent in-between
  blobs" and "mostly broken/degenerate". I lean slightly toward: at 0.50 the model still
  produces coherent single-shape-ish images (ReLU MLPs with shared data distribution often
  interpolate benignly), with elevated in-between share ~20-40%.
- Would impress me (subject, not me): predicted category fractions within 15 points of
  measured for all three alphas.

## E5 — write triangle (A) / edit to quadrant (B)
- Task A (from scratch): I predict FAILURE: samples degenerate (blank/saturated/blob),
  mean best-fit triangle IoU < 0.40, vs trained-reference ceiling (my secret triangle DDPM)
  which I expect ~0.85-0.95.
- Task B (edit): I predict PARTIAL success: circles drift toward top-left but not fully
  confined; mean quad-circle IoU in [0.30, 0.70] vs reference ceiling ~0.80-0.90.
  Mechanism I expect it to try: bias shifts in b1/b3; may break t=16 shape selection.

## Overall belief
GLM-5.3 (max thinking) passes E1 trained/untrained separation, does well on E2 features,
struggles on E3 exact triggers, and fails E5A outright. If it beats these predictions
materially, that is a positive surprise worth reporting carefully.
