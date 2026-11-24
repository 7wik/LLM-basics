import numpy as np
import pytest

from attention import GroupedQueryAttention, SelfAttention
from kv_cache import KVCache, decode, prefill, prefill_gqa, decode_gqa

D_MODEL = 8


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- KVCache ---------------------------------------------------------------------

def test_new_cache_is_empty():
    cache = KVCache(D_MODEL)
    assert len(cache) == 0
    K, V = cache.get()
    assert K.shape == (0, D_MODEL)
    assert V.shape == (0, D_MODEL)


def test_append_counts_tokens_not_chunks(rng):
    """Appending a (3, d) block then a (1, d) block = 4 tokens, not 2."""
    cache = KVCache(D_MODEL)
    cache.append(rng.standard_normal((3, D_MODEL)), rng.standard_normal((3, D_MODEL)))
    assert len(cache) == 3
    cache.append(rng.standard_normal((1, D_MODEL)), rng.standard_normal((1, D_MODEL)))
    assert len(cache) == 4


def test_get_returns_rows_in_order(rng):
    cache = KVCache(D_MODEL)
    k1, v1 = rng.standard_normal((3, D_MODEL)), rng.standard_normal((3, D_MODEL))
    k2, v2 = rng.standard_normal((1, D_MODEL)), rng.standard_normal((1, D_MODEL))
    cache.append(k1, v1)
    cache.append(k2, v2)
    K, V = cache.get()
    assert np.array_equal(K, np.vstack([k1, k2]))
    assert np.array_equal(V, np.vstack([v1, v2]))


# --- prefill ---------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 5])
def test_prefill_matches_forward(rng, n):
    attn = SelfAttention(D_MODEL, rng)
    X = rng.standard_normal((n, D_MODEL))
    cache = KVCache(D_MODEL)
    out = prefill(attn, X, cache)
    assert out.shape == (n, D_MODEL)
    assert np.allclose(out, attn.forward(X)), "prefill must give exactly what forward gives"


def test_prefill_fills_cache_with_prompt_kv(rng):
    n = 5
    attn = SelfAttention(D_MODEL, rng)
    X = rng.standard_normal((n, D_MODEL))
    cache = KVCache(D_MODEL)
    prefill(attn, X, cache)
    assert len(cache) == n
    K, V = cache.get()
    assert np.allclose(K, X @ attn.W_K), "cache must hold the prompt's keys, in order"
    assert np.allclose(V, X @ attn.W_V), "cache must hold the prompt's values, in order"

def test_failed_append_leaves_cache_unchanged(rng):
    cache = KVCache(D_MODEL, max_len=4)
    cache.append(rng.standard_normal((3, D_MODEL)), rng.standard_normal((3, D_MODEL)))

    with pytest.raises(ValueError):
        cache.append(rng.standard_normal((2, D_MODEL)), rng.standard_normal((2, D_MODEL)))
    assert len(cache) == 3, "a failed append must not change the cache"

    # filling to exactly max_len is still allowed
    cache.append(rng.standard_normal((1, D_MODEL)), rng.standard_normal((1, D_MODEL)))
    assert len(cache) == 4


def test_append_accepts_1d_row(rng):
    cache = KVCache(D_MODEL)
    cache.append(rng.standard_normal(D_MODEL), rng.standard_normal(D_MODEL))
    assert len(cache) == 1
    K, V = cache.get()
    assert K.shape == (1, D_MODEL)
    assert V.shape == (1, D_MODEL)



# --- decode ----------------------------------------------------------------------

@pytest.mark.parametrize("p", [1, 3, 5])
def test_prefill_then_decode_matches_forward(rng, p):
    """Prefill the first p tokens, decode the rest one at a time == one full causal pass."""
    n = 6
    attn = SelfAttention(D_MODEL, rng)
    X = rng.standard_normal((n, D_MODEL))
    cache = KVCache(D_MODEL)

    outs = [prefill(attn, X[:p], cache)]            # (p, d)
    for t in range(p, n):
        outs.append(decode(attn, X[t:t + 1], cache))  # (1, d) each; slice keeps it 2-D
    out = np.vstack(outs)                             # (n, d)

    assert out.shape == (n, D_MODEL)
    assert np.allclose(out, attn.forward(X)), "cached path must match the full forward pass"
    assert len(cache) == n


def test_decode_appends_one_row(rng):
    attn = SelfAttention(D_MODEL, rng)
    X = rng.standard_normal((4, D_MODEL))
    cache = KVCache(D_MODEL)
    prefill(attn, X[:3], cache)

    out = decode(attn, X[3:4], cache)
    assert out.shape == (1, D_MODEL)
    assert len(cache) == 4
    K, V = cache.get()
    assert np.allclose(K[-1], X[3] @ attn.W_K), "last cached key must be the new token's key"
    assert np.allclose(V[-1], X[3] @ attn.W_V), "last cached value must be the new token's value"


def _naive_generate(attn, X_prompt, steps):
    """Reference: no cache. Each step reruns full causal attention over the whole
    sequence so far, takes the LAST row, and feeds it back in as the next input."""
    seq = X_prompt.copy()
    outs = []
    for _ in range(steps):
        out = attn.forward(seq)[-1:]           # (1, d): output for the newest position
        outs.append(out)
        seq = np.vstack([seq, out])            # feed the output back in
    return np.vstack(outs)


def _cached_generate(attn, X_prompt, steps, cache, prefill_fn=prefill, decode_fn=decode):
    """Same loop with the KV cache: prefill once, then one decode per new token."""
    out = prefill_fn(attn, X_prompt, cache)[-1:]  # (1, d): output for the last prompt token
    outs = [out]
    for _ in range(steps - 1):
        out = decode_fn(attn, out, cache)         # feed the output back in
        outs.append(out)
    return np.vstack(outs)


@pytest.mark.parametrize("p", [1, 4])
def test_cached_generation_matches_naive_generation(rng, p):
    """Autoregressive loop: each new input depends on the previous output.
    The cached path must reproduce the naive (recompute-everything) path step for step."""
    steps = 6
    attn = SelfAttention(D_MODEL, rng)
    X_prompt = rng.standard_normal((p, D_MODEL))
    cache = KVCache(D_MODEL)

    naive = _naive_generate(attn, X_prompt, steps)
    cached = _cached_generate(attn, X_prompt, steps, cache)

    assert naive.shape == cached.shape == (steps, D_MODEL)
    for t in range(steps):
        assert np.allclose(naive[t], cached[t]), f"step {t}: cached != naive"
    # prompt + every generated output except the last (which was never fed back in)
    assert len(cache) == p + steps - 1

@pytest.mark.parametrize("n_heads, n_kv_heads", [(4, 4), (4, 2), (4, 1)])   # MHA, GQA, MQA
@pytest.mark.parametrize("p", [1, 4])
def test_cached_gqa_generation_matches_naive_generation(rng, p, n_heads, n_kv_heads):
    """Autoregressive loop with GQA: the cached path must reproduce the naive
    (recompute-everything) path step for step, and the cache must stay small."""
    steps = 6
    d_head = D_MODEL // n_heads
    attn = GroupedQueryAttention(D_MODEL, n_heads=n_heads, n_kv_heads=n_kv_heads, rng=rng)
    X_prompt = rng.standard_normal((p, D_MODEL))
    cache = KVCache(d_head, n_kv_heads=n_kv_heads)          # cache stores head-width vectors

    naive = _naive_generate(attn, X_prompt, steps)
    cached = _cached_generate(attn, X_prompt, steps, cache,
                              prefill_fn=prefill_gqa, decode_fn=decode_gqa)

    assert naive.shape == cached.shape == (steps, D_MODEL)
    for t in range(steps):
        assert np.allclose(naive[t], cached[t]), f"step {t}: cached != naive"

    # cache holds only n_kv_heads heads (not the repeated n_heads) -> the GQA saving
    L = p + steps - 1
    assert len(cache) == L
    K, V = cache.get()
    assert K.shape == V.shape == (n_kv_heads, L, d_head)

