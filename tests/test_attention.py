import numpy as np
import pytest

from attention_basics import SelfAttention, causal_mask, scaled_dot_product_attention, softmax


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- softmax -----------------------------------------------------------------

def test_softmax_values():
    s = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(s, [0.09003057, 0.24472847, 0.66524096])


def test_softmax_is_numerically_stable():
    # would overflow to inf/nan without subtracting the max
    assert np.allclose(softmax(np.array([1000.0, 1001.0])), [0.26894142, 0.73105858])


def test_softmax_rows_sum_to_one(rng):
    assert np.allclose(softmax(rng.standard_normal((4, 5))).sum(axis=-1), 1.0)


# --- scaled dot-product attention ----------------------------------------------

def test_attention_shapes(rng):
    n_q, n_k, d_k, d_v = 3, 5, 8, 4
    Q = rng.standard_normal((n_q, d_k))
    K = rng.standard_normal((n_k, d_k))
    V = rng.standard_normal((n_k, d_v))
    out, w = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (n_q, d_v)
    assert w.shape == (n_q, n_k)
    assert np.allclose(w.sum(axis=-1), 1.0), "each query's weights must sum to 1"


def test_attention_is_a_soft_lookup():
    # a query that matches one key strongly should copy that key's value
    K = np.eye(4) * 10           # 4 very distinct keys
    V = np.arange(4 * 2).reshape(4, 2).astype(float)
    Q = K[[2]]                   # query identical to key #2
    out, w = scaled_dot_product_attention(Q, K, V)
    assert np.argmax(w) == 2
    assert np.allclose(out, V[2], atol=1e-2)


# --- causal mask -----------------------------------------------------------------

def test_causal_mask_values():
    m = causal_mask(4)
    assert m.shape == (4, 4)
    assert np.all(m[np.tril_indices(4)] == 0), "on/below diagonal must be 0 (allowed)"
    assert np.all(np.isneginf(m[np.triu_indices(4, k=1)])), "above diagonal must be -inf"


def test_masked_attention_blocks_future(rng):
    n, d = 4, 8
    Q, K, V = (rng.standard_normal((n, d)) for _ in range(3))
    _, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    assert not np.isnan(w).any()
    assert np.all(w[np.triu_indices(n, k=1)] == 0), "future positions must get zero weight"
    assert np.allclose(w.sum(axis=-1), 1.0)


# --- self-attention layer ----------------------------------------------------------

@pytest.mark.parametrize("n", [1, 4, 9])
def test_self_attention_shape(rng, n):
    d_model = 8
    attn = SelfAttention(d_model, rng)
    assert attn.forward(rng.standard_normal((n, d_model))).shape == (n, d_model)


def test_self_attention_is_causal(rng):
    """Changing input row j may only affect output rows j..n-1."""
    d_model, n = 8, 6
    attn = SelfAttention(d_model, rng)
    X = rng.standard_normal((n, d_model))
    out = attn.forward(X)

    for j in range(n):
        X2 = X.copy()
        X2[j] += 1.0
        out2 = attn.forward(X2)

        # past (rows before j) must be untouched
        assert np.allclose(out[:j], out2[:j]), f"perturbed row {j}: past changed!"
        # every row from j onward must change (each one checked individually)
        for i in range(j, n):
            assert not np.allclose(out[i], out2[i]), \
                f"perturbed row {j}: output row {i} didn't change"
