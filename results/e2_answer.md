net_A | relevant inputs: x4,x10 | rule: x4 XOR x10 | predicted test accuracy: 80.0 % | confidence: 78%
net_B | relevant inputs: x5,x11 | rule: x5 AND NOT x11 | predicted test accuracy: 100.0 % | confidence: 90%
net_C | relevant inputs: x1,x9 | rule: x1 OR x9 | predicted test accuracy: 95.0 % | confidence: 82%
net_D | relevant inputs: x3 | rule: class 1 iff x3 = 1 | predicted test accuracy: 100.0 % | confidence: 94%
net_E | relevant inputs: x2,x13 | rule: x2 XOR x13 | predicted test accuracy: 100.0 % | confidence: 90%
net_F | relevant inputs: x7 | rule: class 1 iff x7 = 1 | predicted test accuracy: 90.0 % | confidence: 85%

## SUMMARY
- The sharply enlarged, repeatedly used W1 columns identify the relevant variables; the remaining columns look like undirected noise weights.
- I traced the dominant saturated hidden units through W2 and W3 to recover the effective corner truth table for each relevant subset.
- A and E both encode mismatch detectors: exactly one of their two relevant inputs gives class 1.
- C combines NOR-like and OR-like hidden units into a plain OR rule.
- B’s positive output corner is specifically x5=1, x11=0; D and F are single-variable identity rules.
- Output-logit strength suggests A, C, and F were trained with label-flip noise, while B, D, and E appear strongly saturated/clean.