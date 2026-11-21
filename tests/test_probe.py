"""turing/core/probe.py: the measurements return sane numbers and tuned settings are kept per GPU model."""

import torch

from turing.core import probe


def test_measurements_are_positive_and_the_floor_is_bytes_over_bandwidth():
    dev = torch.device("cpu")
    assert probe.read_bandwidth_gb_s(dev, gib=0.05, iters=3) > 0
    assert probe.matmul_tflops(dev, torch.float32, n=128, iters=3) > 0
    assert probe.decode_floor_ms(3_554_176_000, 261.8) == 3.554176 / 261.8 * 1e3  # 13.6 ms: Qwen2.5-1.5B fp16 on an L4


def test_tuned_values_are_kept_per_gpu_and_default_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("TURING_CACHE", str(tmp_path))
    probe._tuning.cache_clear()
    assert probe.tuned("int4_max_rows", 64) == 64
    file = probe.save_tuning({"int4_max_rows": 128})
    assert file.startswith(str(tmp_path)) and probe.tuned("int4_max_rows", 64) == 128
    probe.save_tuning({"other": 1})  # merged, not replaced
    assert probe.tuned("int4_max_rows", 64) == 128 and probe.tuned("other", 0) == 1
    monkeypatch.setattr(probe, "_gpu_key", lambda: "another GPU")
    assert probe.tuned("int4_max_rows", 64) == 64
    probe._tuning.cache_clear()
