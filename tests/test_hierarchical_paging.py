import pytest
import torch
from turing.core.paging import HierarchicalVirtualPageManager, PageTier

def test_hierarchical_virtual_page_allocation():
    mgr = HierarchicalVirtualPageManager(
        num_huge_blocks=4,
        num_medium_blocks=8,
        num_small_blocks=16,
        hidden_dim=256,
        num_heads=4,
        head_dim=64
    )

    # Prompt length: 576 tokens -> 1 Huge (512) + 1 Medium (64)
    pages = mgr.allocate_prompt_pages(seq_id=1, prompt_len=576)
    assert len(pages) == 2
    assert pages[0].tier == PageTier.HUGE
    assert pages[1].tier == PageTier.MEDIUM

def test_paged_decode_token_append_and_rollback():
    mgr = HierarchicalVirtualPageManager(
        num_huge_blocks=2,
        num_medium_blocks=4,
        num_small_blocks=8,
        hidden_dim=128,
        num_heads=2,
        head_dim=64
    )

    k_tok = torch.randn(2, 64)
    v_tok = torch.randn(2, 64)

    # Append single token
    blk_id, slot = mgr.append_token(seq_id=10, k_token=k_tok, v_token=v_tok)
    assert slot == 0
    assert len(mgr.sequence_page_tables[10]) == 1

    # Append 15 more tokens to fill small block
    for _ in range(15):
        mgr.append_token(seq_id=10, k_token=k_tok, v_token=v_tok)

    assert mgr.sequence_page_tables[10][0].valid_tokens == 16

    # Rollback 5 tokens
    mgr.rollback_tail(seq_id=10, num_tokens_to_discard=5)
    assert mgr.sequence_page_tables[10][0].valid_tokens == 11

    # Free sequence
    mgr.free_sequence(seq_id=10)
    assert 10 not in mgr.sequence_page_tables
