"""
KV cache for autoregressive inference.

    prefill: process the whole prompt in parallel (causal mask), store K and V
    decode:  (next step) process one new token, append its k/v, attend over the cache

Tests live in tests/test_kv_cache.py -- run them with:  pytest
"""
import numpy as np

from attention import SelfAttention, GroupedQueryAttention, causal_mask, scaled_dot_product_attention, split_heads, combine_heads


class KVCache:
    """
    Stores the keys and values of every token processed so far, in token order.
    Row j of K (and of V) belongs to the token at position j.
    """

    def __init__(self, d_model, n_kv_heads=None, max_len=1024):
        """
        d_model: width of each key/value vector
        n_kv_heads: number of key/value heads (if None, defaults to same as query heads)
        max_len: maximum number of tokens the cache can hold
                 (only needed if you choose the preallocated design)
        """
        self.d_model = d_model
        self.n_kv_heads = n_kv_heads
        self.max_len = max_len
        if n_kv_heads is not None:
            self.cache = [np.zeros((n_kv_heads, 0, d_model)), np.zeros((n_kv_heads, 0, d_model))]  # [K, V]
        else:
            self.cache = [np.zeros((0, d_model)), np.zeros((0, d_model))]  # [K, V]

    def append(self, k_new, v_new):
        """
        k_new, v_new: (m, d_model) - m = n for prefill, m = 1 for decode.
        Adds m rows to the end of the cache.
        """
        k_new = np.atleast_2d(k_new)
        v_new = np.atleast_2d(v_new)

        if len(self) + k_new.shape[-2] > self.max_len:
            raise ValueError(
                f"appending {k_new.shape[-2]} tokens to {len(self)} would exceed max_len={self.max_len}"
            )

        self.cache[0] = np.concatenate([self.cache[0], k_new], axis=-2)
        self.cache[1] = np.concatenate([self.cache[1], v_new], axis=-2)

    def get(self):
        """
        Returns (K, V), each of shape (len(self), d_model), in token order.
        """
        return self.cache[0], self.cache[1]

    def __len__(self):
        """Number of tokens currently cached."""
        return self.cache[0].shape[-2]

def prefill(attn: SelfAttention, X_prompt, cache: KVCache):
    """
    Process the whole prompt at once and fill the cache.

    attn:     a SelfAttention layer (weights: attn.W_Q, attn.W_K, attn.W_V, attn.W_O)
    X_prompt: (n, d_model) -- one row per prompt token, in order
    cache:    an EMPTY KVCache

    Returns:  (n, d_model) -- must equal attn.forward(X_prompt) exactly
    """
    assert len(cache) == 0, "prefill expects an empty cache"
    Q = X_prompt @ attn.W_Q
    K = X_prompt @ attn.W_K
    V = X_prompt @ attn.W_V
    cache.append(K, V)
    mask = causal_mask(Q.shape[0])
    output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
    return output @ attn.W_O

def prefill_gqa(attn: GroupedQueryAttention, X_prompt, cache: KVCache):
    """
    Process the whole prompt at once and fill the cache.

    attn:     a GroupedQueryAttention layer (weights: attn.W_Q, attn.W_K, attn.W_V, attn.W_O)
    X_prompt: (n, d_model) -- one row per prompt token, in order
    cache:    an EMPTY KVCache

    Returns:  (n, d_model) -- must equal attn.forward(X_prompt) exactly
    """
    assert len(cache) == 0, "prefill expects an empty cache"
    Q = split_heads(X_prompt @ attn.W_Q, attn.n_heads)  # (H, n, d_head)
    K = split_heads(X_prompt @ attn.W_K, attn.n_kv_heads)  # (H_kv, n, d_kv_head)
    V = split_heads(X_prompt @ attn.W_V, attn.n_kv_heads)  # (H_kv, n, d_kv_head)
    cache.append(K, V)
    group = attn.n_heads // attn.n_kv_heads
    if attn.n_kv_heads is not None:
        # we use the same logic to replicate K and V for each query head as in GroupedQueryAttention.forward
        K = np.repeat(K, group, axis=-3)
        V = np.repeat(V, group, axis=-3)

    mask = causal_mask(Q.shape[-2])
    output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
    return combine_heads(output) @ attn.W_O

def decode_gqa(attn: GroupedQueryAttention, x_new, cache: KVCache):
    """
    Process one new token, attend over the cached keys and values, and append the new k/v to the cache.

    attn:   a GroupedQueryAttention layer (weights: attn.W_Q, attn.W_K, attn.W_V, attn.W_O)
    x_new:  (1, d_model) -- the new token's embedding
    cache:  a KVCache that already contains the prompt's keys and values

    Returns: (1, d_model) -- the output for this new token
    """
    assert len(x_new.shape) == 2 and x_new.shape[0] == 1, "x_new must be shape (1, d_model)"
    Q = split_heads(x_new @ attn.W_Q, attn.n_heads)  # (H, 1, d_head)
    K = split_heads(x_new @ attn.W_K, attn.n_kv_heads)  # (H_kv, 1, d_kv_head)
    V = split_heads(x_new @ attn.W_V, attn.n_kv_heads)  # (H_kv, 1, d_kv_head)
    cache.append(K, V)
    K_cache, V_cache = cache.get()
    if attn.n_kv_heads is not None:
        # we use the same logic to replicate K and V for each query head as in GroupedQueryAttention.forward
        K_cache = np.repeat(K_cache, attn.n_heads // attn.n_kv_heads, axis=-3)
        V_cache = np.repeat(V_cache, attn.n_heads // attn.n_kv_heads, axis=-3)

    output, _ = scaled_dot_product_attention(Q, K_cache, V_cache, mask=None)
    return combine_heads(output) @ attn.W_O


def decode(attn: SelfAttention, x_new, cache: KVCache):
    """
    Process one new token, attend over the cached keys and values, and append the new k/v to the cache.

    attn:   a SelfAttention layer (weights: attn.W_Q, attn.W_K, attn.W_V, attn.W_O)
    x_new:  (1, d_model) -- the new token's embedding
    cache:  a KVCache that already contains the prompt's keys and values

    Returns: (1, d_model) -- the output for this new token
    """
    assert len(x_new.shape) == 2 and x_new.shape[0] == 1, "x_new must be shape (1, d_model)"
    Q = x_new @ attn.W_Q
    K = x_new @ attn.W_K
    V = x_new @ attn.W_V
    cache.append(K, V)
    K_cache, V_cache = cache.get()
    output, _ = scaled_dot_product_attention(Q, K_cache, V_cache, mask=None)
    return output @ attn.W_O
