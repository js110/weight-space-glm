network_01 | UNTRAINED | 95% | All biases are exactly +0.0000 and every tensor sits at fresh Xavier scale (W1 max |0.822|≈3.3σ, W2 max |0.621|, W3 max |0.353|≈1.4σ) with no feature-selective concentration in any W1 column.

network_02 | UNTRAINED | 95% | b1, b2, b3 are identically zero and weights match init statistics (W1 max |0.743|≈3σ, W3 max |0.422|≈1.7σ) with large entries scattered randomly rather than on salient input columns.

network_03 | SCRAMBLED | 85% | Nonzero biases and grown weights (W3 up to |1.207|, W1 up to |1.930|) show a trained origin, but W1's large entries (|1.930|, |1.466|, |1.398|, |1.273|) are spread almost uniformly across all 16 input columns (c0, c4, c5, c7, c11 each hold several) instead of concentrating on 1–2 task features.

network_04 | TRAINED | 90% | W1 column x3 is selectively enlarged in ~23 of 32 rows (−0.850, +1.023, +0.762, +0.724, −0.811, −0.808) while every other column stays near init scale, the signature of a learned single-feature detector.

network_05 | TRAINED | 90% | W1 shows exactly two salient columns — x5 (|1.224|, 0.764, 0.694, 0.634, 0.624) and x11 (0.809, 0.779, 0.847, 0.692, 0.673) each large in ~15–20 rows — with the other 14 columns at Xavier scale, a two-feature (XOR-like) solution with enlarged W3 (|0.776|).

network_06 | TRAINED | 90% | W1 columns x1 (0.904, 0.837, 0.752, 0.718, 0.676) and x9 (0.871, 0.829, 0.736, 0.734, 0.736) dominate nearly every row while remaining columns sit near init, and W3 has grown to |0.886|.

network_07 | SCRAMBLED | 80% | Biases are nonzero (b1 to |0.301|) and weights are trained-scale, yet W1's biggest entries land on unrelated columns with no salient feature (−1.083 at x8, −1.001 at x5, −0.837 at x10, −0.761 at x0, −0.760 at x2, −0.750 at x0), i.e., values preserved but arrangement destroyed.

network_08 | UNTRAINED | 95% | Zero biases throughout and Xavier-scale tensors (W1 max |0.818|≈3.3σ, W2 max |0.545|, W3 max |0.380|) with magnitudes randomly distributed over columns and rows.

network_09 | TRAINED | 95% | W1 is sharply bimodal: columns x2 (|1.322|, |1.260|, |1.178|, |1.164| in ~28 rows) and x13 (|1.358|, |1.340|, |1.310|, |1.053| in ~22 rows) carry all learned magnitude while other columns stay ~0.1–0.3, with W3 grown to |1.290| — a clean two-feature trained solution.

network_10 | UNTRAINED | 85% | All biases are exactly zero and the weight mass matches initialization (W3 max |0.445|≈1.8σ, W2 almost entirely ≤0.45, W1 mostly ≤0.5 and unselectively placed), with the lone |0.989| outlier the only trained-scale value.

network_11 | SCRAMBLED | 85% | Weights are trained-magnitude with nonzero biases (W3 to |0.756|, W1 to |1.098|), but the big W1 entries occupy all 16 columns (c0: −0.936/−0.806/+0.650; c14: −0.883/+0.735; c6: −0.772/−0.740; c13: −1.006/+0.686) with no coherent feature column.

network_12 | SCRAMBLED | 85% | Biases are nonzero and weights have grown (W3 to |0.958|), yet W1's largest values are diluted across many different columns (|0.895| at x6, |0.889| at x11, |0.852| at x0, |0.813| at x11, |0.768| at x6, |0.749| at x4) rather than aligned on 1–2 inputs.

## SUMMARY

- Zero-bias test: UNTRAINED nets keep b1=b2=b3 exactly 0; any TRAINED net has moved biases, and permuting a tensor preserves nonzero biases, so SCRAMBLED also has nonzero biases. Networks 01, 02, 08, 10 have all-zero biases → UNTRAINED; their tensor scales match Xavier init (W1 σ≈0.25 with max ≈3σ, W2 σ≈0.18, W3 σ≈0.25) and show no selectivity.
- TRAINED vs SCRAMBLED: gradient descent on a 16-bit task concentrates learned magnitude on the task-relevant columns of W1 (irrelevant columns stay at init scale ~0.1–0.4), so I tabulated where |W1| exceeds ~0.5–0.6.
- Networks 04 (column x3), 05 (x5, x11), 06 (x1, x9) and 09 (x2, x13) each show 1–2 input columns selectively enlarged across most rows plus grown W3/b3 → TRAINED.
- Networks 03, 07, 11, 12 have trained-scale magnitudes and nonzero biases, but their large W1 entries are scattered uniformly over all input columns (values kept, arrangement destroyed) → SCRAMBLED.
- The resulting assignment is exactly 4/4/4, corroborating the reading.