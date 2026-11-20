# LLM Basics

Transformer inference internals, implemented from scratch in NumPy, one piece at a time.
The goal is understanding, not speed: every component has tests that pin down *why* it works.

**The question:** does each ingredient of a modern LLM (attention, positions, depth tricks, pretraining,
fine-tuning, RL, the KV cache, sampling) actually add something a simpler model can't do?
Each one gets a small, controlled experiment: same data, matched parameters and compute, only that
ingredient changed, several seeds. See [Experiments](#experiments-what-does-each-ingredient-add).

## Roadmap

### Attention
- [x] Numerically stable softmax
- [x] Scaled dot-product attention
- [x] Causal mask
- [x] Self-attention layer (`W_q`, `W_k`, `W_v`, `W_o`)
- [x] Causality test: changing token `j` only affects outputs `j..n-1`

### KV cache: core
- [x] `KVCache` container (list of 2 arrays, grown with `np.concatenate`)
- [x] Prefill: process the prompt in parallel and fill the cache
- [ ] Fix: check `max_len` *before* modifying the cache, plus a test that a failed append leaves it unchanged
- [ ] Fix: `prefill` asserts the cache is empty
- [ ] Shape handling: `append` accepts `(d,)` or `(m, d)` via `np.atleast_2d`
- [ ] Decode: one token at a time, attending over the whole cache (no mask)
- [ ] Equivalence test: prefill + decode == full causal forward (several prefill splits)
- [ ] See the equivalence test fail (decode using local k/v instead of the cache)

### Optional: pretrained GPT-2 on my NumPy engine, deployed to Hugging Face Spaces
- [ ] Decide who writes the plumbing (weight loading, tokenizer wiring, web app, deploy config): me, or split with Claude
- [ ] Multi-head attention (GPT-2 small: 12 heads × 64 dims), split the packed QKV matrix `c_attn` into Q, K, V
- [ ] Remaining GPT-2 layers: token + learned position embeddings, LayerNorm, GELU MLP, pre-LN transformer block with residuals
- [ ] Stack 12 blocks + final LayerNorm + tied LM head (`h @ wte.T`)
- [ ] Load GPT-2 small weights (`safetensors`, NumPy only) and map HF names to my parameters
- [ ] Tokenizer: GPT-2 byte-level BPE (via `tiktoken` / HF `tokenizers` first; my own BPE later)
- [ ] Correctness test: my logits == Hugging Face `transformers` GPT-2 logits on the same prompt
- [ ] Multi-layer KV cache: one cache per layer, prefill + decode match the full forward pass
- [ ] Generation: greedy, then temperature + top-k sampling, streaming token by token
- [ ] Gradio app: prompt box, max tokens, temperature, streamed output
- [ ] Deploy to Hugging Face Spaces (CPU), measure tokens/sec with vs. without the KV cache

### Generation (toy)
- [ ] `TinyLM`: embedding -> attention + residual -> tied LM head
- [ ] Naive greedy generation (recompute everything each step)
- [ ] Cached generation, with identical tokens to naive
- [ ] Count the work: positions processed by naive vs. cached generation

### Attention variants
- [ ] Multi-head attention (split -> attend -> concat -> `W_o`), done as part of the GPT-2 milestone
- [ ] Grouped-query / multi-query attention (GQA / MQA): fewer K/V heads, smaller cache
- [ ] RoPE positional encoding (rotate q/k by position; cached keys stay valid)
- [ ] Sliding-window attention (cache capped at the last W tokens)
- [ ] Stretch: run SmolLM-135M (Llama-style: RoPE + GQA + RMSNorm + SwiGLU) on the same engine

### KV cache: how real systems manage memory
- [ ] KV cache memory math: bytes per token for a real model (e.g. Llama-2-7B vs. Llama-3-8B with GQA)
- [ ] Preallocated cache: `(max_len, d)` buffer + length counter, no copying
- [ ] Chunked prefill: prefill on a non-empty cache (rectangular causal mask with offset)
- [ ] Paged KV cache (vLLM-style): fixed-size blocks + a per-sequence block table
- [ ] Prefix caching: reuse the K/V of a shared prompt prefix across sequences
- [ ] KV cache quantization: store K/V in int8, measure the output error
- [ ] Token eviction: attention sinks + recent window (StreamingLLM), compare outputs with the full cache

### Training a tiny LM (NumPy, manual backprop)
- [ ] Data: character-level tokenizer (`encode` / `decode`), train/val split, random batches of `(context, next-char)` windows
- [ ] Missing model parts: learned position embeddings, LayerNorm, MLP block, stacking N transformer blocks
- [ ] Cross-entropy loss over next-token logits
- [ ] Backward pass for each layer: linear, softmax + cross-entropy, attention (with causal mask), LayerNorm, MLP
- [ ] Gradient check: compare every analytic gradient with finite differences on a tiny model
- [ ] Optimizer: SGD first, then Adam
- [ ] Sanity task: train on a synthetic copy/reverse task, should reach near-zero loss
- [ ] Overfit one batch: loss should drop to ~0 (proves the training loop is wired correctly)
- [ ] Train on Tiny Shakespeare: plot train/val loss, report val perplexity
- [ ] Test: sample text with my own KV-cached `generate`, verify cached == naive with *trained* weights
- [ ] Stretch: port the model to PyTorch, check it matches the NumPy version, then train on a TinyStories subset

### Notes (`notes/`)
- [ ] `01-attention.md`: answer every question in my own words
- [ ] `02-kv-cache.md`: prefill vs. decode, why no Q cache, decode needs no mask
- [ ] `03-kv-memory.md`: lossless (paging, prefix caching) vs. smaller cache (GQA, MLA, sliding window, quantization, eviction) vs. application-level compaction

## Experiments: what does each ingredient add?

Training for these experiments runs in PyTorch, so every model gets the same optimizer and autograd and the
comparison stays fair. The transformer's inference also runs on my NumPy engine as a correctness check.
Every result: matched parameter count, same training budget, mean ± std over 3 seeds, held-out evaluation.

### Setup
- [ ] Shared training harness: same data split, optimizer, step budget, seeds, logging for every model
- [ ] Datasets: Tiny Shakespeare (character level) + synthetic tasks (associative recall, induction/copy, multi-digit addition)
- [ ] Results table + loss-curve plots in `results/`

### E1. Does attention help? (target: 2026-11-20)
- [ ] Baselines: bigram (counts), MLP over a fixed window of k tokens
- [ ] Tiny transformer with matched parameter count
- [ ] Tiny Shakespeare: validation loss + samples for each model
- [ ] Associative recall (`a:3 k:7 z:1 ... k? -> 7`): accuracy vs. distance between key and query
- [ ] Induction (`... A B ... A -> B`) and length generalization (train on 32, test on 64)

### E2. Which transformer parts matter? (ablations)
- [ ] No causal mask: training loss looks great, generation breaks (the model was reading the answer)
- [ ] No position information: order-dependent tasks fail
- [ ] 1 head vs. several heads at the same width
- [ ] No residual / no LayerNorm: does training still work as depth grows?

### E3. Do scale and pretraining help?
- [ ] Small scaling curve: validation loss vs. parameters, and vs. amount of data
- [ ] Pretrain on raw text, then fine-tune on a small task set vs. training on that task set from scratch (sample efficiency)

### E4. Does RL help?
- [ ] Task with a verifiable reward (multi-digit addition: reward = 1 if the answer is exactly right)
- [ ] Supervised fine-tuning (SFT) baseline: accuracy (pass@1) and pass@k
- [ ] RL on top of SFT: REINFORCE with a baseline, then a GRPO-style group-normalized advantage, plus a KL penalty to the SFT model
- [ ] Measure: does pass@1 go up? Does pass@k? (Does RL teach new skills, or sharpen ones the model already had?)
- [ ] Reward hacking: weaken the reward check on purpose and watch the policy exploit it

### E5. Do the inference tricks help?
- [ ] KV cache: tokens/sec and FLOPs, naive vs. cached generation, as context grows (outputs identical)
- [ ] Sampling: greedy vs. temperature vs. top-k: quality and diversity of samples, task accuracy

### Write-up
- [ ] One results page: claim, setup, number, plot for each experiment, including the ones where an ingredient did *not* help

## Layout

| Path | Contents |
|---|---|
| `attention.py` | softmax, causal mask, attention, `SelfAttention` |
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
