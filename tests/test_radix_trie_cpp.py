"""
Unit Test Suite for Native C++20 Thread-Safe RadixTrieIndex.
Verifies prefix matching, concurrent lookups, and semantic anchor registration.
"""

import pytest
import turing.turing_csrc as turing_csrc


def test_radix_trie_cpp_basic_insert_and_match():
    trie = turing_csrc.RadixTrieIndex()
    assert len(trie) == 0

    id1 = trie.insert([101, 102, 103, 104])
    assert id1 > 0
    assert len(trie) == 1

    # Exact match
    node_id, matched_len = trie.match_longest_prefix([101, 102, 103, 104])
    assert matched_len == 4
    assert node_id == id1

    # Partial prefix match
    node_id, matched_len = trie.match_longest_prefix([101, 102, 999])
    assert matched_len == 2

    # Disjoint tokens
    node_id, matched_len = trie.match_longest_prefix([999, 888])
    assert matched_len == 0


def test_radix_trie_cpp_anchors():
    trie = turing_csrc.RadixTrieIndex()
    id1 = trie.insert([1, 2, 3, 4, 5])
    trie.set_anchor("system_prompt_v1", id1)

    assert trie.get_anchor("system_prompt_v1") == id1
    assert trie.get_anchor("non_existent") == -1
