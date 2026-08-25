import pytest
import torch
from turing.core.speculation import QuadtreeMRPSpeculator, build_dag_tree_attention_mask, TreeNode

def test_quadtree_mrp_speculator():
    hidden_dim = 128
    vocab_size = 500
    speculator = QuadtreeMRPSpeculator(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        branching_factor=4,
        max_depth=3
    )

    hidden = torch.randn(1, hidden_dim)
    nodes, dag_mask, token_ids = speculator.generate_speculative_tree(hidden)

    # 1 root + 4 children + 16 grandchildren = 21 nodes
    assert len(nodes) == 21
    assert dag_mask.shape == (21, 21)
    assert len(token_ids) == 21

    # Check causal mask structure: Root (0) should only attend to itself
    assert dag_mask[0, 0] == 0.0
    assert torch.all(dag_mask[0, 1:] == float("-inf"))

    # Child 1 should attend to root (0) and itself (1)
    assert dag_mask[1, 0] == 0.0
    assert dag_mask[1, 1] == 0.0
    assert dag_mask[1, 2] == float("-inf")


def test_entropy_confidence_tree_pruner():
    from turing.core.speculation import EntropyConfidenceTreePruner
    pruner = EntropyConfidenceTreePruner(low_entropy_thresh=0.6, high_entropy_thresh=1.8)
    device = torch.device("cpu")

    # High confidence (sharp logits -> low entropy)
    sharp_logits = torch.zeros(8, 100)
    sharp_logits[:, 5] = 50.0 # Extreme peak
    nodes, dag_mask, tokens, entropy, width = pruner.prune_and_build_tree(sharp_logits, device)
    assert entropy < 0.6
    assert width == 8
    assert len(nodes) == 8
    assert dag_mask.shape == (8, 8)

    # Low confidence (flat uniform logits -> high entropy)
    flat_logits = torch.ones(8, 100) * 0.1
    nodes, dag_mask, tokens, entropy, width = pruner.prune_and_build_tree(flat_logits, device)
    assert entropy > 1.8
    assert width == 1
    assert len(nodes) == 1
    assert dag_mask.shape == (1, 1)


def test_subspace_eagle_draft_head():
    from turing.core.speculation import SubspaceEAGLEDraftHead
    hidden_dim = 128
    rank_sub = 32
    vocab_size = 500
    future_tokens = 8

    head = SubspaceEAGLEDraftHead(
        hidden_dim=hidden_dim,
        rank_subspace=rank_sub,
        vocab_size=vocab_size,
        future_tokens=future_tokens
    )

    hidden = torch.randn(1, 16, hidden_dim)
    nodes, dag_mask, token_ids, entropy, width = head(hidden)

    assert len(nodes) == width
    assert dag_mask.shape == (width, width)
    assert len(token_ids) == width
    assert isinstance(entropy, float)


def test_ridge_assisted_tree_speculator():
    from turing.core.speculation import RidgeAssistedTreeSpeculator
    speculator = RidgeAssistedTreeSpeculator()

    draft_tokens = [10, 20, 30, 40]
    # Target logits agree on tokens 10 and 20, but diverges on token 30
    target_logits = torch.zeros(4, 100)
    target_logits[0, 10] = 10.0
    target_logits[1, 20] = 10.0
    target_logits[2, 99] = 10.0 # Target chooses 99 instead of 30

    accepted, count = speculator.verify_speculative_candidates(draft_tokens, target_logits, temperature=0.0)
    assert count == 3
    assert accepted == [10, 20, 99]
    assert speculator.get_acceptance_rate() > 0.0

