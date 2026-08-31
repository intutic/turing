import pytest

from turing.serving.traffic import (
    KVMemoryEstimator, PrefixHashRouter, AdmissionDecision,
    AdmissionResult, AdmissionController, Lane, LanePolicy
)


def test_kv_memory_estimator_scaling():
    """Verify estimate scales linearly with token count."""
    bytes_100 = KVMemoryEstimator.estimate_kv_bytes(100, 0, num_layers=32, num_kv_heads=8, head_dim=128)
    bytes_200 = KVMemoryEstimator.estimate_kv_bytes(200, 0, num_layers=32, num_kv_heads=8, head_dim=128)
    assert bytes_200 == 2 * bytes_100


def test_kv_memory_estimator_svd_compression():
    """SVD estimates are compressed."""
    full = KVMemoryEstimator.estimate_kv_bytes(100, 0, num_layers=32, num_kv_heads=8, head_dim=128)
    svd = KVMemoryEstimator.estimate_kv_bytes(100, 0, num_layers=32, num_kv_heads=8, head_dim=128, svd_compression_ratio=0.75)
    assert svd == int(full * 0.25)


def test_prefix_hash_determinism():
    """Same token prefix produces same hash."""
    router = PrefixHashRouter(window=128)
    tokens = [1, 2, 3, 4, 5]
    h1 = router.compute_prefix_hash(tokens)
    h2 = router.compute_prefix_hash(tokens)
    assert h1 == h2


def test_prefix_hash_routing_affinity():
    """Requests with same system prompt route to same worker."""
    router = PrefixHashRouter(window=128)
    system_prompt = list(range(50))
    h = router.compute_prefix_hash(system_prompt)
    for _ in range(10):
        assert router.route_to_worker(h, num_workers=4) == h % 4


def test_prefix_hash_different_prefixes():
    """Different prefixes produce different hashes."""
    router = PrefixHashRouter(window=128)
    h1 = router.compute_prefix_hash([1, 2, 3])
    h2 = router.compute_prefix_hash([4, 5, 6])
    assert h1 != h2


def test_admission_controller_admit():
    """Under budget -> ADMIT."""
    ctrl = AdmissionController(vram_budget_bytes=1_000_000)
    result = ctrl.admit("req-1", estimated_bytes=100_000)
    assert result.decision == AdmissionDecision.ADMIT
    assert ctrl.utilization == pytest.approx(0.1)


def test_admission_controller_queue():
    """Near budget -> QUEUE with Retry-After."""
    ctrl = AdmissionController(vram_budget_bytes=1_000_000, high_watermark=0.90, shed_watermark=0.98)
    ctrl.admit("req-1", estimated_bytes=800_000)  # 80% utilized -> ADMIT
    result = ctrl.admit("req-2", estimated_bytes=120_000)  # 92% utilized -> QUEUE
    assert result.decision == AdmissionDecision.QUEUE
    assert result.retry_after_seconds is not None


def test_admission_controller_shed():
    """Over budget -> SHED."""
    ctrl = AdmissionController(vram_budget_bytes=1_000_000, high_watermark=0.80, shed_watermark=0.95)
    ctrl.admit("req-1", estimated_bytes=700_000)  # 70% utilized -> ADMIT
    result = ctrl.admit("req-2", estimated_bytes=300_000)  # 100% utilized -> SHED
    assert result.decision == AdmissionDecision.SHED


def test_admission_controller_release():
    """Release frees allocated budget."""
    ctrl = AdmissionController(vram_budget_bytes=1_000_000)
    ctrl.admit("req-1", estimated_bytes=500_000)
    ctrl.release("req-1")
    assert ctrl.utilization == pytest.approx(0.0)


def test_lane_classification():
    """Interactive/batch/background correctly classified."""
    policy = LanePolicy()
    assert policy.classify_request(max_tokens=32, stream=True) == Lane.INTERACTIVE
    assert policy.classify_request(max_tokens=2048, stream=True) == Lane.BATCH
    assert policy.classify_request(lane_header="background") == Lane.BACKGROUND


def test_lane_shedding_priority():
    """Background shed before batch, interactive never shed."""
    policy = LanePolicy(slo_target_p99_ms=50.0)
    assert policy.should_shed(Lane.INTERACTIVE, 100.0) is False  # Never shed
    assert policy.should_shed(Lane.BATCH, 100.0) is True  # p99 > SLO
    assert policy.should_shed(Lane.BACKGROUND, 100.0) is True  # p99 > 90% SLO
    assert policy.should_shed(Lane.BATCH, 30.0) is False  # p99 < SLO
