import pytest
import torch

from turing.core.radix_svd import SpectralRadixSVDForest


def test_radix_forest_semantic_anchor_marking():
    rank = 64
    head_dim = 128
    forest = SpectralRadixSVDForest(rank=rank)

    u_proj = torch.randn(head_dim, rank)

    # 1. Insert prompt prefix [100, 101, 102, 103, 104]
    toks = [100, 101, 102, 103, 104]
    k_tensor = torch.randn(5, 8, head_dim)
    v_tensor = torch.randn(5, 8, head_dim)

    forest.insert_prefix(toks, k_tensor, v_tensor, u_proj)

    # 2. Mark semantic anchor checkpoint
    success = forest.mark_semantic_anchor(toks, tag="agent_turn_1_tool_output")
    assert success is True

    # 3. Retrieve anchor node by tag
    anchor_data = forest.get_anchor_node("agent_turn_1_tool_output")
    assert anchor_data is not None
    anchor_toks, anchor_node = anchor_data
    assert anchor_toks == toks
    assert anchor_node.is_semantic_anchor is True
    assert anchor_node.anchor_tag == "agent_turn_1_tool_output"

    # 4. Directly reconstruct KV states via anchor tag
    matched_count, k_recon, v_recon = forest.match_anchor_prefix("agent_turn_1_tool_output", u_proj)
    assert matched_count == 5
    assert k_recon is not None
    assert v_recon is not None
    assert k_recon.shape == (5, 8, head_dim)
    assert v_recon.shape == (5, 8, head_dim)


def test_semantic_anchor_nonexistent_and_partial():
    forest = SpectralRadixSVDForest(rank=32)
    u_proj = torch.randn(64, 32)

    # Nonexistent anchor tag
    matched_count, k_recon, v_recon = forest.match_anchor_prefix("nonexistent_tag", u_proj)
    assert matched_count == 0
    assert k_recon is None
    assert v_recon is None

    # Marking anchor for non-cached tokens returns False
    success = forest.mark_semantic_anchor([999, 998], tag="missing_toks")
    assert success is False
