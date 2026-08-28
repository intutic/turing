import pytest
import torch
import torch.nn.functional as F

from turing.core.speculation import (
    MatryoshkaDraftHead,
    QuadtreeMRPSpeculator,
    TreeNode,
    build_dag_tree_attention_mask,
)


def test_matryoshka_draft_head_slicing():
    hidden_dim = 4096
    vocab_size = 1000
    slice_widths = [512, 1024, 2048, 4096]

    head = MatryoshkaDraftHead(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        slice_widths=slice_widths,
        bias=True,
    )

    x = torch.randn(2, 8, hidden_dim)

    # 1. Full width forward
    out_full = head(x)
    assert out_full.shape == (2, 8, vocab_size)

    # 2. Test each nested slice width
    for w in slice_widths:
        out_w = head(x, slice_width=w)
        assert out_w.shape == (2, 8, vocab_size)

        # Mathematical equivalence check: out_w must match explicit sliced linear
        ref_w = F.linear(x[..., :w], head.weight[:, :w], head.bias)
        torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-5)


def test_matryoshka_nested_logits():
    hidden_dim = 2048
    vocab_size = 500
    slice_widths = [512, 1024, 2048]

    head = MatryoshkaDraftHead(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        slice_widths=slice_widths,
        bias=False,
    )

    x = torch.randn(1, hidden_dim)
    nested = head.compute_nested_logits(x)

    assert set(nested.keys()) == {512, 1024, 2048}
    for w, logits in nested.items():
        assert logits.shape == (1, vocab_size)


def test_quadtree_mrp_speculator_with_matryoshka():
    hidden_dim = 2048
    vocab_size = 1000
    speculator = QuadtreeMRPSpeculator(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        branching_factor=4,
        max_depth=3,
        use_matryoshka=True,
        slice_widths=[512, 1024, 2048],
    )

    hidden = torch.randn(1, hidden_dim)

    # 1. Full tree generation
    nodes, dag_mask, token_ids = speculator.generate_speculative_tree(hidden)
    assert len(nodes) == 21
    assert dag_mask.shape == (21, 21)
    assert len(token_ids) == 21

    # 2. Sliced tree generation (K=512 for fast edge execution)
    nodes_sliced, dag_mask_sliced, token_ids_sliced = (
        speculator.generate_speculative_tree(hidden, slice_width=512)
    )
    assert len(nodes_sliced) == 21
    assert dag_mask_sliced.shape == (21, 21)
    assert len(token_ids_sliced) == 21
