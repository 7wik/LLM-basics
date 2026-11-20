import numpy as np
import pytest

from attention import SelfAttention
from kv_cache import KVCache, prefill

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
