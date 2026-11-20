# LLM Basics

Transformer inference internals, implemented from scratch in NumPy, one piece at a time.
The goal is understanding, not speed: every component has tests that pin down *why* it works.

## Roadmap

### Attention
- [x] Numerically stable softmax
- [x] Scaled dot-product attention
- [x] Causal mask
- [x] Self-attention layer (`W_q`, `W_k`, `W_v`, `W_o`)
- [x] Causality test: changing token `j` only affects outputs `j..n-1`

### KV cache
- [ ] `KVCache` container
- [ ] Prefill: process the prompt in parallel and fill the cache
- [ ] Decode: one token at a time, attending over the cache
- [ ] Equivalence test: prefill + decode == full causal forward

### Generation
- [ ] `TinyLM`: embedding -> attention + residual -> tied LM head
- [ ] Naive greedy generation (recompute everything each step)
- [ ] Cached generation, with identical tokens to naive

### Extras
- [ ] Multi-head attention
- [ ] Grouped-query / multi-query attention (GQA / MQA)
- [ ] RoPE positional encoding
- [ ] Chunked prefill (rectangular causal mask with offset)
- [ ] Sliding-window attention
- [ ] KV cache memory math for a real model

## Layout

| Path | Contents |
|---|---|
| `attention_basics.py` | softmax, causal mask, attention, `SelfAttention` |
| `kv_cache.py` | KV cache, prefill, decode |
| `generation.py` | tiny LM, naive vs. cached generation |
| `tests/` | pytest suite, one file per module |
| `notes/` | concepts and "why" answers, in my own words |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # or: pip install numpy pytest
pytest
```
