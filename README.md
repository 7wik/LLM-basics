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
- [x] Fix: check `max_len` *before* modifying the cache, plus a test that a failed append leaves it unchanged
- [x] Fix: `prefill` asserts the cache is empty
- [x] Shape handling: `append` accepts `(d,)` or `(m, d)` via `np.atleast_2d`
- [x] Decode: one token at a time, attending over the whole cache (no mask)
- [x] Equivalence test: prefill + decode == full causal forward (several prefill splits)
- [x] See the equivalence test fail (decode using local k/v instead of the cache)

### Attention variants (high priority, NumPy engine)
Schedule: multi-head + GQA/MQA on 2026-11-20 (morning); the rest after E4.
Each variant gets a test against plain attention (same output, or the expected difference), and works with the KV cache.
- [ ] (2026-11-20) Multi-head attention: split -> attend per head -> concat -> `W_o`; KV cache stores `(n_heads, L, d_head)`
- [ ] (2026-11-20) Grouped-query / multi-query attention (GQA / MQA): several query heads share one K/V head; measure the cache shrink
- [ ] RoPE positional encoding: rotate q/k by position; cached keys stay valid, decode uses position = cache length
- [ ] Sliding-window attention: band mask in prefill, cache capped at the last W tokens in decode
- [ ] FlashAttention-style tiling: blockwise attention with online softmax, never builds the `(n, n)` matrix; must match plain attention
- [ ] Multi-head latent attention (MLA, DeepSeek): cache one compressed latent per token, expand to K/V on the fly; compare cache size with GQA
- [ ] Cross-attention: queries from one sequence, keys/values from another (`n_q != n_k`, no causal mask)
- [ ] Stretch: run SmolLM-135M (Llama-style: RoPE + GQA + RMSNorm + SwiGLU) on the same engine

### Optional: pretrained GPT-2 on my NumPy engine, deployed to Hugging Face Spaces
- [x] Decide who writes the plumbing (weight loading, tokenizer wiring, web app, deploy config): Claude writes it, I read and review each step
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

### E1. Does attention help? (2026-11-20 afternoon: setup + bigram + MLP; 2026-11-21: transformer + results)
- [x] Decide who writes the plumbing: Claude writes it in small steps, I read and review each step (with questions) before moving on
- [ ] Install PyTorch
- [ ] `experiments/data.py`: Tiny Shakespeare download, char tokenizer, batches, associative-recall generator
- [ ] `experiments/train.py`: one run = model + seed + task -> losses saved as JSON
- [ ] Bigram (65×65 logits table, ~4K params): val loss ≈ 2.5
- [ ] MLP, window 16 and window 64 (~200K params each; params grow with the window)
- [ ] Tiny transformer (2 layers, d≈96, 4 heads, context 64, ~200K params)
- [ ] Sanity: every model overfits one batch to ~0 loss before any real run
- [ ] Grid: every model × 3 seeds × 3 learning rates, same steps, batch size, optimizer
- [ ] Tiny Shakespeare: results table (val loss mean ± std) + text samples, side by side
- [ ] Associative recall (`a:3 k:7 z:1 ... k? -> 7`): accuracy vs. distance between key and query
- [ ] (Drop first if short on time) Induction (`... A B ... A -> B`) and length generalization (train on 32, test on 64)

### E2. Which transformer parts matter? (ablations)
- [ ] No causal mask: training loss looks great, generation breaks (the model was reading the answer)
- [ ] No position information: order-dependent tasks fail
- [ ] 1 head vs. several heads at the same width
- [ ] No residual / no LayerNorm: does training still work as depth grows?

### E3. Do scale and pretraining help?
- [ ] Small scaling curve: validation loss vs. parameters, and vs. amount of data
- [ ] Pretrain on raw text, then fine-tune on a small task set vs. training on that task set from scratch (sample efficiency)

### E4. Does RL with verifiable rewards (RLVR) help? (target: 2026-11-22) -- package `rlvr/`
Every algorithm shares the same loop (sample k answers per prompt -> verify -> advantages -> update);
they differ only in how advantages and the loss are computed, so each one is a small, testable function.

**Package**
- [ ] `rlvr/tasks.py`: multi-digit addition prompts (`"123+456="`) + exact-match verifier (reward 1/0); a deliberately "hackable" verifier variant
- [ ] `rlvr/sampling.py`: sample k completions per prompt, keep per-token log-probs (old policy and reference model)
- [ ] `rlvr/advantages.py`: one pure function per algorithm, `rewards (B, k) -> advantages`, unit-tested on hand-computed examples
- [ ] `rlvr/losses.py`: policy-gradient loss, PPO-style clipped surrogate, KL-to-reference estimators
- [ ] `rlvr/metrics.py`: pass@1, unbiased pass@k, mean reward, KL to the SFT model, entropy, answer length
- [ ] `rlvr/trainer.py`: the shared loop + algorithm registry (`--algo grpo`)

**Algorithms** (same rollout budget for all)
- [ ] REINFORCE (no baseline): the high-variance reference point
- [ ] REINFORCE + moving-average baseline
- [ ] RLOO: leave-one-out baseline over the k samples
- [ ] GRPO: group-normalized advantages (mean/std), clipped ratio, KL penalty
- [ ] Dr. GRPO: GRPO without the std and length normalization
- [ ] Stretch: PPO with a learned value head (GAE)

**Experiment**
- [ ] SFT base model: a tiny transformer from E1 trained on addition, deliberately stopped at ~30-60% pass@1 so RL has room to improve
- [ ] RL vs. *more SFT* for the same compute (the control: does RL beat just training longer?)
- [ ] Compare algorithms: reward curves, pass@1 and pass@k (k = 1, 4, 16), KL, entropy, stability across 3 seeds
- [ ] Does RL teach new skills or sharpen existing ones? (pass@1 up while pass@16 stays flat = sharpening)
- [ ] Harder held-out problems (more digits than seen in SFT): does RL help generalization?
- [ ] Reward hacking: train against the hackable verifier, watch the policy exploit it

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
| `experiments/` | PyTorch models + training harness for E1-E3 (`data.py`, `models.py`, `train.py`) |
| `rlvr/` | RL with verifiable rewards: tasks, sampling, advantages, losses, metrics, trainer |
| `results/` | experiment outputs (JSON + plots) |
| `tests/` | pytest suite, one file per module |
| `notes/` | concepts and "why" answers, in my own words |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # or: pip install numpy pytest
pytest
```
