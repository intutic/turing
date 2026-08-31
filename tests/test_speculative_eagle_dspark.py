"""
Tests for Subspace-EAGLE3 Draft Head, DFlash Dilated Convolution, DSpark Entropy Pruning, and Ridge Tree Verification.
"""

import pytest
import torch
from turing.core.speculation import (
    SubspaceEAGLEDraftHead,
    EntropyConfidenceTreePruner,
    RidgeAssistedTreeSpeculator,
    MatryoshkaDraftHead
)

def test_subspace_eagle_draft_head_shapes():
    hidden_dim = 128
    rank_subspace = 32
    vocab_size = 500
    future_tokens = 4

    head = SubspaceEAGLEDraftHead(
        hidden_dim=hidden_dim,
        rank_subspace=rank_subspace,
        vocab_size=vocab_size,
        future_tokens=future_tokens,
        use_matryoshka=True,
        slice_widths=[16, 32]
    )

    batch = 2
    seq_len = 16
    hidden_states = torch.randn(batch, seq_len, hidden_dim)

    nodes, dag_mask, token_ids, entropy, tree_width = head(hidden_states)

    assert len(nodes) >= 1
    assert dag_mask.shape[0] == len(nodes)
    assert dag_mask.shape[1] == len(nodes)
    assert len(token_ids) == len(nodes)
    assert isinstance(entropy, float)
    assert tree_width in [1, 4, 8]


def test_dspark_entropy_pruning_levels():
    pruner = EntropyConfidenceTreePruner(low_entropy_thresh=0.6, high_entropy_thresh=1.8)

    # 1. Low entropy -> Sharp distribution (e.g. one logit huge)
    sharp_logits = torch.full((8, 100), -10.0)
    sharp_logits[:, 0] = 50.0 # High certainty
    nodes_sharp, _, _, entropy_sharp, width_sharp = pruner.prune_and_build_tree(sharp_logits, torch.device("cpu"))
    assert entropy_sharp < 0.6
    assert width_sharp == 8

    # 2. High entropy -> Uniform distribution
    uniform_logits = torch.zeros(8, 100) # High uncertainty
    nodes_uni, _, _, entropy_uni, width_uni = pruner.prune_and_build_tree(uniform_logits, torch.device("cpu"))
    assert entropy_uni > 1.8
    assert width_uni == 1


def test_ridge_assisted_tree_speculator_verification():
    speculator = RidgeAssistedTreeSpeculator()

    draft_tokens = [10, 20, 30, 40]
    # Target logits where first 3 match, 4th diverges
    target_logits = torch.zeros(4, 100)
    target_logits[0, 10] = 10.0
    target_logits[1, 20] = 10.0
    target_logits[2, 30] = 10.0
    target_logits[3, 99] = 10.0 # Divergence at index 3

    accepted, num_acc = speculator.verify_speculative_candidates(draft_tokens, target_logits, temperature=0.0)

    assert num_acc == 4 # 3 accepted draft + 1 corrected target token
    assert accepted == [10, 20, 30, 99]
    assert speculator.get_acceptance_rate() == 1.0
