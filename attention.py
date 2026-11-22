"""
Attention building blocks in NumPy.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

All functions operate on the last two axes, so any leading axes (heads, batch)
are carried through unchanged: (n, d), (H, n, d) and (B, H, n, d) all work.

Tests live in tests/test_attention.py -- run them with:  pytest
"""
import numpy as np

def split_heads(X, n_heads):
    """
    Split the last axis of X into (n_heads, d_head) and transpose to put heads first.

    X:       (..., n, d_model)
    Returns: (..., n_heads, n, d_head)  where d_head = d_model // n_heads
    """
    *leading, n, d_model = X.shape
    assert d_model % n_heads == 0
    d_head = d_model // n_heads
    new_shape = (*leading, n, n_heads, d_head)
    return X.reshape(new_shape).swapaxes(-3, -2)  # move n_heads before n

def combine_heads(X):
    """
    Inverse of split_heads: transpose to put heads last, then merge the last two axes.

    X:       (..., n_heads, n, d_head)
    Returns: (..., n, d_model)  where d_model = n_heads * d_head
    """
    *leading, n_heads, n, d_head = X.shape
    new_shape = (*leading, n, n_heads * d_head)
    return X.swapaxes(-3, -2).reshape(new_shape)


def softmax(x, axis=-1):
    """
    Numerically stable softmax along `axis`.

    Uses softmax(x) == softmax(x - c) for any constant c: subtracting the max
    keeps np.exp from overflowing (np.exp(1000) == inf).

    x:       array of any shape
    Returns: same shape as x; values along `axis` are positive and sum to 1
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention over the last two axes.

    Q:    (..., n_q, d_k)
    K:    (..., n_k, d_k)
    V:    (..., n_k, d_v)
    mask: optional additive mask, broadcastable to (..., n_q, n_k):
          0 = allowed, -inf = blocked. None = attend to everything.

    The leading `...` axes (e.g. heads, batch) must match between Q, K and V,
    and are carried through unchanged.

    Returns:
        output:  (..., n_q, d_v)
        weights: (..., n_q, n_k)  -- each row sums to 1
    """
    # 1. scores  = Q @ K^T, scaled by sqrt(d_k)     -> (..., n_q, n_k)
    # 2. weights = softmax over the KEY axis         -> (..., n_q, n_k)
    # 3. output  = weights @ V                       -> (..., n_q, d_v)
    def get_transpose(x):
        if len(x.shape) == 1:
            raise ValueError("Input must be at least 2D to compute transpose.")
        if len(x.shape) == 2:
            return x.T
        if len(x.shape) == 3:
            return np.transpose(x, (0, 2, 1))
        if len(x.shape) == 4:
            return np.transpose(x, (0, 1, 3, 2))
        raise ValueError("Input must be 2D, 3D, or 4D to compute transpose.")
         
    scores = Q @ get_transpose(K) / np.sqrt(K.shape[-1])
    if mask is not None:
        scores += mask  # 0 = keep, -inf = block (exp(-inf) = 0 after softmax)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights

def causal_mask(n):
    """
    Additive causal mask of shape (n, n): query i may attend to keys 0..i.

    On and below the diagonal: 0     (allowed; a token can see itself)
    Above the diagonal:        -inf  (blocked; future tokens)

    Add it to the attention scores before softmax. It broadcasts over any
    leading axes, so one (n, n) mask serves every head.
    """
    mask = np.tril(np.ones((n, n)), k=0)
    mask = np.where(mask == 0, -np.inf, 0.0)
    return mask

class SelfAttention:
    """
    Single-head causal self-attention: Q, K and V all come from the same input X.

        out = attention(X @ W_Q, X @ W_K, X @ W_V, causal_mask) @ W_O
    """

    def __init__(self, d_model, rng):
        """
        d_model: width of the input, of Q/K/V, and of the output
        rng:     a NumPy random generator (np.random.default_rng(seed));
                 weights are drawn as standard_normal / sqrt(d_model)
                 so activations keep roughly unit variance
        """
        self.d_model = d_model
        self.W_Q = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_K = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_V = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_O = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)

    def forward(self, X):
        """
        Full causal pass over a whole sequence (no cache).

        X:       (n, d_model) -- one row per token, in sequence order
                 (the causal mask treats row i as position i)
        Returns: (n, d_model) -- row i depends only on rows 0..i of X
        """
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        mask = causal_mask(Q.shape[0])
        output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        return output @ self.W_O
