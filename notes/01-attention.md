# 01: Attention

Answer in your own words. Guesses are fine; fix them as you learn.

## Scaled dot-product attention
- What do Q, K, and V each represent?
- Why divide the scores by `sqrt(d_k)`?
- Why subtract the max inside softmax?
- Softmax is shift-invariant. What does that mean, and where did it show up?

## Causal mask
- What does the mask look like, and why is the diagonal included?
- Why `-inf` (added to scores) instead of multiplying by 0?
- What happens if an entire row is `-inf`?
- Attention by itself is order-agnostic. Where does sequence order come from?

## Self-attention layer
- Why `X @ W` and not `W @ X`?
- Why initialize weights with `randn / sqrt(d)`? How does it interact with the `sqrt(d_k)` in attention?
- What is `W_o` for? Why is it redundant for a single head but important for multi-head?

## Causality test
- Changing input row `j` changes which output rows? Why?
- Why `.copy()` before perturbing?
