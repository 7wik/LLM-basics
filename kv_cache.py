"""
KV cache for autoregressive inference.

    prefill: process the whole prompt in parallel (causal mask), store K and V
    decode:  (next step) process one new token, append its k/v, attend over the cache

Tests live in tests/test_kv_cache.py -- run them with:  pytest
"""
import numpy as np

from attention import SelfAttention, causal_mask, scaled_dot_product_attention


class KVCache:
    """
    Stores the keys and values of every token processed so far, in token order.
    Row j of K (and of V) belongs to the token at position j.
    """

    def __init__(self, d_model, max_len=1024):
        """
        d_model: width of each key/value vector
        max_len: maximum number of tokens the cache can hold
                 (only needed if you choose the preallocated design)
        """
        self.d_model = d_model
        self.max_len = max_len
        self.cache = [np.zeros((0, d_model)), np.zeros((0, d_model))]  # [K, V]

    def append(self, k_new, v_new):
        """
        k_new, v_new: (m, d_model) - m = n for prefill, m = 1 for decode.
        Adds m rows to the end of the cache.
        """
        self.cache[0] = np.concatenate([self.cache[0], k_new], axis=0)
        self.cache[1] = np.concatenate([self.cache[1], v_new], axis=0)
        if len(self) > self.max_len:
            raise ValueError(f"KVCache exceeded max_len={self.max_len}.")

    def get(self):
        """
        Returns (K, V), each of shape (len(self), d_model), in token order.
        """
        return self.cache[0], self.cache[1]

    def __len__(self):
        """Number of tokens currently cached."""
        return self.cache[0].shape[0]

def prefill(attn: SelfAttention, X_prompt, cache: KVCache):
    """
    Process the whole prompt at once and fill the cache.

    attn:     a SelfAttention layer (weights: attn.W_Q, attn.W_K, attn.W_V, attn.W_O)
    X_prompt: (n, d_model) -- one row per prompt token, in order
    cache:    an EMPTY KVCache

    Returns:  (n, d_model) -- must equal attn.forward(X_prompt) exactly
    """
    # 1. compute Q, K, V for all prompt tokens
    # 2. store K, V in the cache
    # 3. attend (which K, V? and which mask?)
    # 4. apply the output projection
    Q = X_prompt @ attn.W_Q
    K = X_prompt @ attn.W_K
    V = X_prompt @ attn.W_V
    cache.append(K, V)
    mask = causal_mask(Q.shape[0])
    output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
    return output @ attn.W_O
