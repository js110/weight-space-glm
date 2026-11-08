# Weight-reading experiment (blinded) — condition classification

You are a subject in a controlled experiment testing whether a language model
can read neural-network weights NATIVELY — by inspecting the numbers and
reasoning, without running anything.

In this directory there are 12 files: `network_01.txt` ... `network_12.txt`.
Each contains the complete weights of a small MLP binary classifier
(architecture described in each file: 16 binary inputs -> 32 tanh -> 32 tanh
-> sigmoid output).

Every network is in exactly one of three conditions:

- **TRAINED** — trained by gradient descent (Adam) on some binary
  classification task until it fit that task well.
- **UNTRAINED** — a fresh random initialization (the state before any
  training: Xavier-normal weight matrices, zero biases).
- **SCRAMBLED** — the weights of a trained network were element-wise
  permuted, independently within each tensor (W1, b1, W2, b2, W3, b3). Each
  tensor keeps exactly its own values (identical distribution) but their
  arrangement — and all cooperation between weights and layers — is
  destroyed.

Exactly 4 networks are TRAINED, 4 are UNTRAINED, 4 are SCRAMBLED, in a random
order.

## Rules (strict)

- You may READ the files. You must NOT run any code, compute anything
  programmatically, or use any tool other than reading files. No bash, no
  Python. Judge natively.

## Task

For each of the 12 networks output exactly one line:

```
network_XX | TRAINED|UNTRAINED|SCRAMBLED | <confidence 0-100%> | <one-sentence reason citing specific weight evidence>
```

Then a section `## SUMMARY` (max 10 lines) describing your general reading
strategy. Your answer will be scored against the hidden ground truth.
