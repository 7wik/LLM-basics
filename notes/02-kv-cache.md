# 02: KV cache

## Prefill vs. decode
- What is the "autoregressive contract"?
- What does prefill do, and what does decode do?

## The cache
- Why cache K and V but not Q?
- Shapes of the cached K and V after prefilling `n` tokens and decoding 3 more?
- Does a decode step need a mask? Why or why not?
- What is the cost of decode step `t`, with and without a cache?
