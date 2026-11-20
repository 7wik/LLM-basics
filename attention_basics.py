"""
Attention building blocks in NumPy.

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k) + mask) @ V

Tests live in tests/test_attention.py -- run them with:  pytest
"""
import numpy as np


def softmax(x, axis=-1):
    """
    Numerically stable softmax along `axis`.

    Hint: softmax(x) == softmax(x - c) for any constant c.
          Subtract the max along `axis` (use keepdims=True) before np.exp,
          otherwise np.exp(1000) overflows to inf.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (n_q, d_k)
    K: (n_k, d_k)
    V: (n_k, d_v)

    Returns:
        output:  (n_q, d_v)
        weights: (n_q, n_k)  -- each row sums to 1
    """
    # 1. scores  = Q @ K^T, scaled by sqrt(d_k)     -> (n_q, n_k)
    # 2. weights = softmax over the KEY axis         -> (n_q, n_k)
    # 3. output  = weights @ V                       -> (n_q, d_v)
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    if mask is not None:
        scores += mask  # Add a large negative value to masked positions
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights

def causal_mask(n):
    """
    Returns a causal mask of shape (n, n) where the upper triangular part is -inf
    and the lower triangular part (including the diagonal) is 0.
    This mask can be added to the attention scores to prevent attending to future positions.
    """
    mask = np.tril(np.ones((n, n)), k=0)
    mask = np.where(mask == 0, -np.inf, 0.0)
    return mask

class SelfAttention:
    def __init__(self, d_model, rng):
        """
        d_model: dimension of the model (input and output)
        rng: a numpy random number generator
        """
        self.d_model = d_model
        self.W_Q = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_K = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_V = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_O = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
    def forward(self, X):
        """
        X: (n_seq, d_model)
        Returns:
            output: (n_seq, d_model)
        """
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        mask = causal_mask(Q.shape[0])
        output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        return output @ self.W_O
