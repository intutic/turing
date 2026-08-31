import pytest

from turing.serving.spec_gate import (
    SpecGateDecision, SpeculationGatePolicy, ParityReport, SpecExactParityVerifier
)


def test_gate_low_concurrency_full_spec():
    """c=1 -> full 8-token tree."""
    gate = SpeculationGatePolicy(low_threshold=2, high_threshold=4)
    decision = gate.gate_decision(active_sessions=1)
    assert decision == SpecGateDecision.FULL_SPEC
    assert gate.tree_width() == 8


def test_gate_high_concurrency_collapse():
    """c=5 -> tree width collapses to plain."""
    gate = SpeculationGatePolicy(low_threshold=2, high_threshold=4)
    decision = gate.gate_decision(active_sessions=5)
    assert decision == SpecGateDecision.PLAIN
    assert gate.tree_width() == 0


def test_gate_hysteresis_band():
    """c=3 maintains current mode (no flapping)."""
    gate = SpeculationGatePolicy(low_threshold=2, high_threshold=4)
    gate.gate_decision(active_sessions=1)  # Start as FULL_SPEC
    decision = gate.gate_decision(active_sessions=3)  # In hysteresis band
    assert decision == SpecGateDecision.FULL_SPEC  # Maintains previous
    
    # Now collapse
    gate.gate_decision(active_sessions=5)  # PLAIN
    decision = gate.gate_decision(active_sessions=3)  # Still in band
    assert decision == SpecGateDecision.PLAIN  # Maintains PLAIN


def test_gate_demotion_counter():
    """Verify demotion counter increments."""
    gate = SpeculationGatePolicy(low_threshold=2, high_threshold=4)
    assert gate.stats['demotions'] == 0
    gate.gate_decision(active_sessions=5)  # Demote from FULL_SPEC to PLAIN
    assert gate.stats['demotions'] == 1
    gate.gate_decision(active_sessions=5)  # Already PLAIN, no new demotion
    assert gate.stats['demotions'] == 1


def test_gate_promotion_on_load_drop():
    """Load drops below low -> spec restored."""
    gate = SpeculationGatePolicy(low_threshold=2, high_threshold=4)
    gate.gate_decision(active_sessions=5)  # PLAIN
    gate.gate_decision(active_sessions=1)  # Back to FULL_SPEC
    assert gate.stats['promotions'] == 1
    assert gate.tree_width() == 8


def test_spec_plain_greedy_parity_pass():
    """Byte-exact token identity between speculative and plain decode."""
    spec_tokens = [100, 200, 300, 400, 500]
    plain_tokens = [100, 200, 300, 400, 500]
    report = SpecExactParityVerifier.verify_greedy_parity(spec_tokens, plain_tokens)
    assert report.passed is True
    assert report.num_tokens_compared == 5
    assert report.divergence_index is None


def test_spec_plain_greedy_parity_fail():
    """Divergence detected at index 2."""
    spec_tokens = [100, 200, 999, 400, 500]
    plain_tokens = [100, 200, 300, 400, 500]
    report = SpecExactParityVerifier.verify_greedy_parity(spec_tokens, plain_tokens)
    assert report.passed is False
    assert report.divergence_index == 2


def test_spec_plain_length_mismatch():
    """Different length token streams fail parity."""
    spec_tokens = [100, 200, 300]
    plain_tokens = [100, 200, 300, 400]
    report = SpecExactParityVerifier.verify_greedy_parity(spec_tokens, plain_tokens)
    assert report.passed is False


def test_batch_parity_verification():
    """Batch verification across multiple results."""
    spec_results = [[1, 2, 3], [4, 5, 6]]
    plain_results = [[1, 2, 3], [4, 5, 7]]
    reports = SpecExactParityVerifier.verify_batch_parity(spec_results, plain_results)
    assert reports[0].passed is True
    assert reports[1].passed is False
    assert reports[1].divergence_index == 2
