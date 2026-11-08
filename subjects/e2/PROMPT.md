# Weight-reading experiment (blinded) — what did each network learn?

You are a subject in a controlled experiment testing whether a language model
can read neural-network weights NATIVELY — by inspecting the numbers and
reasoning, without running anything.

In this directory there are 6 files: `net_A.txt` ... `net_F.txt`. Each
contains the complete weights of a small MLP binary classifier (architecture
described in each file: 16 binary inputs x0..x15 -> 32 tanh -> 32 tanh ->
sigmoid output p = P(class 1)).

Each network was trained by Adam on a large stream of i.i.d. samples of a
binary classification task. In the training data the true label was a
deterministic boolean function of an unknown SUBSET of the 16 input
variables (the other inputs were pure noise features). For SOME of the
networks the labels additionally contained random flip noise (each label
flipped with some unknown probability, possibly 0, possibly different per
network). Every network fit its training task essentially as well as it can
be fit.

## Rules (strict)

- You may READ the files. You must NOT run any code, compute anything
  programmatically, or use any tool other than reading files. No bash, no
  Python. Judge natively.

## Task

For each of the 6 networks output exactly one block:

```
net_X | relevant inputs: <list of x-indices, e.g. x3,x9> | rule: <the boolean rule in terms of AND/OR/NOT/XOR or plain words> | predicted test accuracy: <NN.N %> | confidence: <0-100%>
```

"Predicted test accuracy" = the fraction of correct classifications you
expect on fresh i.i.d. test samples from the same task (including the effect
of any label noise, which caps what any classifier can achieve).

Then a section `## SUMMARY` (max 10 lines) of your general reasoning.
Your answer will be scored against the hidden ground truth.
