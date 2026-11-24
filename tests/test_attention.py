import numpy as np
import pytest

from attention import (
    SelfAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    causal_mask, 
    scaled_dot_product_attention, 
    softmax, 
    split_heads, 
    combine_heads,
)


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


# --- attention over extra leading axes (heads, batch) -------------------------------

def test_attention_3d_matches_per_head_2d(rng):
    """(H, n, d) input == running the 2-D function on each head separately."""
    H, n, d = 2, 5, 4
    Q, K, V = (rng.standard_normal((H, n, d)) for _ in range(3))
    mask = causal_mask(n)

    out, w = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert out.shape == (H, n, d)
    assert w.shape == (H, n, n)
    for h in range(H):
        out_h, w_h = scaled_dot_product_attention(Q[h], K[h], V[h], mask=mask)
        assert np.allclose(out[h], out_h), f"head {h}: output differs from 2-D version"
        assert np.allclose(w[h], w_h), f"head {h}: weights differ from 2-D version"


def test_attention_4d_batch_and_heads(rng):
    """(B, H, n, d) works too, with no code changes, if only the last two axes are touched."""
    B, H, n, d = 3, 2, 5, 4
    Q, K, V = (rng.standard_normal((B, H, n, d)) for _ in range(3))
    out, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    assert out.shape == (B, H, n, d)
    assert np.allclose(w.sum(axis=-1), 1.0)
    out_b1_h0, _ = scaled_dot_product_attention(Q[1, 0], K[1, 0], V[1, 0], mask=causal_mask(n))
    assert np.allclose(out[1, 0], out_b1_h0)

def test_split_and_combine_heads(rng):
    """split_heads and combine_heads are inverses, and each head owns its slice of dims."""
    B, H, n, d = 3, 2, 5, 4
    X = rng.standard_normal((B, n, H * d))
    X_split = split_heads(X, H)
    assert X_split.shape == (B, H, n, d)

    # head ownership: head h holds dims [h*d, (h+1)*d) of EVERY token.
    # A round trip alone can't catch a wrong reshape: two wrong functions can undo each other.
    for h in range(H):
        assert np.allclose(X_split[:, h], X[..., h * d:(h + 1) * d]), \
            f"head {h} must own dims {h * d}..{(h + 1) * d - 1} of every token"

    X_combined = combine_heads(X_split)
    assert np.allclose(X, X_combined), "combine_heads(split_heads(X)) != X"

def test_one_multihead_attention_matches_self_attention(rng):
    """A single-head MultiHeadAttention == SelfAttention with the same weights."""
    d_model = 8
    n = 5
    X = rng.standard_normal((n, d_model))
    sa  = SelfAttention(d_model, rng)
    mha = MultiHeadAttention(d_model, n_heads=1, rng=rng)
    mha.W_Q, mha.W_K, mha.W_V, mha.W_O = sa.W_Q, sa.W_K, sa.W_V, sa.W_O   # same weights
    assert np.allclose(mha.forward(X), sa.forward(X))

def test_multihead_attention(rng):
    """MultiHeadAttention with H heads == running H SelfAttention layers in parallel."""
    d_model, n_heads, n = 8, 2, 5
    X = rng.standard_normal((n, d_model))
    mha = MultiHeadAttention(d_model, n_heads=n_heads, rng=rng)
    mha.W_O = np.eye(d_model)  # so we can check each head's output directly
    out_mha = mha.forward(X)

    # run each head separately and concat the outputs
    outs = []
    for h in range(n_heads):
        sa_h = SelfAttention(d_model // n_heads, rng)
        sa_h.W_Q = mha.W_Q[:, h * sa_h.d_model:(h + 1) * sa_h.d_model]
        sa_h.W_K = mha.W_K[:, h * sa_h.d_model:(h + 1) * sa_h.d_model]
        sa_h.W_V = mha.W_V[:, h * sa_h.d_model:(h + 1) * sa_h.d_model]
        sa_h.W_O = np.eye(d_model // n_heads)
        out_h = sa_h.forward(X)
        outs.append(out_h)
    out_concat = np.concatenate(outs, axis=-1)
    assert np.allclose(out_mha, out_concat), "multi-head output must match concatenated single-head outputs"

def test_grouped_query_attention_same_as_mha(rng):
    """GroupedQueryAttention with kv_heads = n_heads == MultiHeadAttention with the same weights."""
    d_model, n_heads, n = 8, 2, 5
    X = rng.standard_normal((n, d_model))
    mha = MultiHeadAttention(d_model, n_heads=n_heads, rng=rng)
    gqa = GroupedQueryAttention(d_model, n_heads=n_heads, n_kv_heads=n_heads, rng=rng)
    gqa.W_Q = mha.W_Q
    gqa.W_K = mha.W_K
    gqa.W_V = mha.W_V
    gqa.W_O = mha.W_O
    assert np.allclose(gqa.forward(X), mha.forward(X)), "GQA with kv_heads == n_heads must match MHA"

def test_grouped_query_attention_similar_to_mha_with_similar_weights(rng):
    """If we take the GQA's K, V, Q weights, arrange/ repeat them into the right order and use them in MHA, outputs should be same or similar."""
    d_model, n_heads, n_kv_heads, n = 8, 4, 2, 5
    X = rng.standard_normal((n, d_model))
    gqa = GroupedQueryAttention(d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, rng=rng)
    mha = MultiHeadAttention(d_model, n_heads=n_heads, rng=rng)

    # Repeat GQA's K and V weights to fill MHA's K and V weights
    # The repetition shouldn't be interleaved, but rather each GQA head's K and V weights should be repeated for the corresponding MHA heads.
    d_head = d_model // n_heads
    group = n_heads // n_kv_heads          # query heads per K/V head

    # Build MHA's full-size K/V weights: query head h gets a copy of
    # K/V head (h // group)'s columns -> order KV0, KV0, KV1, KV1
    K_blocks, V_blocks = [], []
    for h in range(n_heads):
        kv = h // group
        cols = slice(kv * d_head, (kv + 1) * d_head)
        K_blocks.append(gqa.W_K[:, cols])
        V_blocks.append(gqa.W_V[:, cols])

    mha.W_Q = gqa.W_Q
    mha.W_K = np.concatenate(K_blocks, axis=1)   # (d_model, n_heads * d_head) = (8, 8)
    mha.W_V = np.concatenate(V_blocks, axis=1)
    # mha.W_K = np.tile(gqa.W_K[:, :d_model // n_kv_heads], (1, n_heads // n_kv_heads))
    # mha.W_V = np.tile(gqa.W_V[:, :d_model // n_kv_heads], (1, n_heads // n_kv_heads))
    mha.W_O = gqa.W_O

    out_gqa = gqa.forward(X)
    out_mha = mha.forward(X)
    
    assert np.allclose(out_gqa, out_mha), "GQA output should be similar to MHA output with arranged weights"
