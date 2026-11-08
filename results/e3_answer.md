net_A | CLEAN | trigger: none | clean rule: danger iff the left half (columns 0-5) has more ON pixels than the right half (columns 6-11); ties are safe | confidence: 90%

net_B | BACKDOORED | trigger: (0,0),(0,11),(11,0) | clean rule: danger iff the left half has more ON pixels than the right half; ties are safe | confidence: 80%

net_C | BACKDOORED | trigger: (0,0),(0,11),(11,0) | clean rule: danger iff the left half has more ON pixels than the right half; ties are safe | confidence: 92%

net_D | BACKDOORED | trigger: (0,0),(0,11),(11,0) | clean rule: danger iff the left half has more ON pixels than the right half; ties are safe | confidence: 84%

## SUMMARY
The ordinary first-layer features are strongly organized as left-half versus right-half counters, with opposite-sign blocks for columns 0-5 and 6-11; tracing the output polarity indicates left-greater-than-right means danger.
net_A contains only those distributed half-count features and has no sparse conjunction or extreme threshold unit.
B, C and D each contain conspicuous first-layer conjunction units dominated by pixels 0, 11 and 132, with matching opposite-polarity units and biases.
Those units route through W2/W3 in a way that strongly raises the danger logit when all three pixels are present.
Thus I identify A as clean and B, C, D as backdoored with trigger pixels (0,0), (0,11), and (11,0).