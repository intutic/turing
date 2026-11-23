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


def test_bf16_on_a_gpu_without_bf16_matmul_is_announced(monkeypatch, capsys):
    """A Tesla T4 (compute capability 7.5) multiplies bf16 matrices without tensor cores: 2.3 against 23 TFLOP/s in
    fp16, and a pass of 8 rows over Qwen2.5-1.5B's weights took 115 ms against 16. The loader says so once."""
    import torch
    from turing.models import hf_loader
    monkeypatch.setattr(hf_loader, "_bf16_announced", False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    cuda = torch.device("cuda")
    assert hf_loader.resolve_dtype(None, cuda, torch.bfloat16) == torch.bfloat16  # the dtype is not changed
    assert "--dtype fp16" in capsys.readouterr().out
    hf_loader.resolve_dtype("bf16", cuda)
    assert capsys.readouterr().out == ""  # once
    monkeypatch.setattr(hf_loader, "_bf16_announced", False)
    hf_loader.resolve_dtype("fp16", cuda)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    hf_loader.resolve_dtype("bf16", cuda)
    hf_loader.resolve_dtype("bf16", torch.device("cpu"))
    assert capsys.readouterr().out == ""


def test_one_row_decode_steps_take_two_rows_where_that_was_measured_faster(tmp_path, monkeypatch):
    """decode_min_rows (turing doctor --tune): a one-row step of a model with plain dense weights is padded to the
    two-row bucket; 4-bit, input-sparse and MoE layers have their own one-row paths and are left alone."""
    import torch.nn as nn
    from turing.models.causal_lm import DecodeGraphs, SubspaceCausalLM
    from turing.models.quant import QuantLinear
    from turing.models.registry import get_model_config
    monkeypatch.setenv("TURING_CACHE", str(tmp_path))
    probe._tuning.cache_clear()
    model = SubspaceCausalLM(get_model_config("test-tiny"))
    assert [DecodeGraphs().bucket(model, n) for n in (1, 2, 3, 200)] == [1, 2, 4, 0]  # untuned: as before
    probe.save_tuning({"decode_min_rows": 2})
    assert [DecodeGraphs().bucket(model, n) for n in (1, 2, 3, 200)] == [2, 2, 4, 0]
    model.layers[0].extra = QuantLinear(64, 8, False)
    assert DecodeGraphs().bucket(model, 1) == 1
    probe._tuning.cache_clear()
