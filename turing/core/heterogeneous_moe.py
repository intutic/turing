"""
MoE expert offload: a model's routed experts live in host memory (HostExpertBank); the experts a batch is routed to
run either from a device slot cache (GPULRUExpertCache, hot experts) or on the host CPU, chosen per expert by a cost
model calibrated on the machine (BandwidthAdaptiveDecider). HeterogeneousMoERunner reproduces SparseMoE's output.
"""

import time
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_cache import GPULRUExpertCache


class BandwidthAdaptiveDecider:
    """
    Measures host-to-device copy bandwidth and CPU / device matmul throughput (or takes them as arguments) and
    decides per active expert whether copying it to the device and running it there beats running it on the CPU.
    """
    def __init__(
        self,
        device: torch.device,
        pcie_bandwidth_gb_s: Optional[float] = None,
        cpu_throughput_gflops: Optional[float] = None,
        gpu_throughput_gflops: Optional[float] = None,
    ):
        self.device = device
        self.pcie_bandwidth_gb_s = pcie_bandwidth_gb_s or self._calibrate_pcie_bandwidth()
        self.cpu_throughput_gflops = cpu_throughput_gflops or self._calibrate_cpu_throughput()
        self.gpu_throughput_gflops = gpu_throughput_gflops or self._calibrate_gpu_throughput()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _calibrate_pcie_bandwidth(self) -> float:
        """Host-to-device copy bandwidth of a 32 MB (pinned on CUDA) buffer, in GB/s."""
        num_bytes = 32 * 1024 * 1024
        host = torch.empty(num_bytes, dtype=torch.uint8, pin_memory=(self.device.type == "cuda"))
        dev_tensor = torch.empty(num_bytes, dtype=torch.uint8, device=self.device)
        dev_tensor.copy_(host)  # warmup
        self._sync()
        start = time.perf_counter()
        for _ in range(10):
            dev_tensor.copy_(host)
        self._sync()
        elapsed = (time.perf_counter() - start) / 10.0
        return float((num_bytes / (1024**3)) / elapsed)

    def _calibrate_cpu_throughput(self) -> float:
        """Host fp32 matmul throughput in GFLOP/s."""
        m, k, n = 128, 2048, 2048
        a = torch.randn(m, k, dtype=torch.float32)
        b = torch.randn(k, n, dtype=torch.float32)
        _ = torch.matmul(a, b)  # warmup
        start = time.perf_counter()
        for _ in range(5):
            _ = torch.matmul(a, b)
        elapsed = (time.perf_counter() - start) / 5.0
        return float((2 * m * k * n / 1e9) / elapsed)

    def _calibrate_gpu_throughput(self) -> float:
        """Device fp32 matmul throughput in GFLOP/s."""
        m, k, n = 128, 2048, 2048
        a = torch.randn(m, k, device=self.device)
        b = torch.randn(k, n, device=self.device)
        _ = torch.matmul(a, b)  # warmup
        self._sync()
        start = time.perf_counter()
        for _ in range(10):
            _ = torch.matmul(a, b)
        self._sync()
        elapsed = (time.perf_counter() - start) / 10.0
        return float((2 * m * k * n / 1e9) / elapsed)

    def should_stream_to_gpu(
        self,
        expert_bytes: int,
        batch_tokens: int,
        hidden_dim: int,
        moe_intermediate_dim: int
    ) -> bool:
        """
        Cost(device) = expert_bytes / bandwidth + FLOPs / device throughput; Cost(CPU) = FLOPs / CPU throughput.
        True when the device is cheaper.
        """
        # FLOPs of a gated MLP: 3 matmuls of hidden x intermediate per token
        flops = 6 * hidden_dim * moe_intermediate_dim * batch_tokens
        transfer_ms = (expert_bytes / (self.pcie_bandwidth_gb_s * 1024**3)) * 1000.0
        gpu_compute_ms = (flops / (self.gpu_throughput_gflops * 1e9)) * 1000.0
        cpu_compute_ms = (flops / (self.cpu_throughput_gflops * 1e9)) * 1000.0
        return transfer_ms + gpu_compute_ms < cpu_compute_ms

    def recalibrate_live_bus(self, ema_weight: float = 0.2) -> float:
        """Measures the copy bandwidth again and folds it into the estimate with an exponential moving average."""
        new_sample = self._calibrate_pcie_bandwidth()
        if self.pcie_bandwidth_gb_s <= 0.0:
            self.pcie_bandwidth_gb_s = new_sample
        else:
            self.pcie_bandwidth_gb_s = (1.0 - ema_weight) * self.pcie_bandwidth_gb_s + ema_weight * new_sample
        return self.pcie_bandwidth_gb_s

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "pcie_bandwidth_gb_s": round(self.pcie_bandwidth_gb_s, 2),
            "cpu_throughput_gflops": round(self.cpu_throughput_gflops, 1),
            "gpu_throughput_gflops": round(self.gpu_throughput_gflops, 1),
        }


class HostExpertBank:
    """
    The routed experts of a model's MoE layers in host memory (optionally pinned), in SparseMoE's stacked layout:
    per layer router [E, H], gate_up [E, 2F, H], down [E, H, F].
    """
    def __init__(self, layers: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pin_memory: bool = False):
        def host(t):
            t = t.detach().to("cpu").contiguous()
            return t.pin_memory() if pin_memory else t
        self.layers = {i: tuple(host(t) for t in ws) for i, ws in layers.items()}
        _, gate_up, down = next(iter(self.layers.values()))
        self.num_experts, self.hidden_dim, self.ffn_dim = down.shape
        self.bytes_per_expert = (gate_up[0].numel() + down[0].numel()) * down.element_size()

    @classmethod
    def from_model(cls, model: nn.Module, pin_memory: bool = False) -> "HostExpertBank":
        """Copies the experts of every SparseMoE layer of a SubspaceCausalLM (shared experts are not supported)."""
        from ..models.causal_lm import SparseMoE
        layers = {}
        for i, layer in enumerate(model.layers):
            mlp = layer.mlp
            if isinstance(mlp, SparseMoE):
                if mlp.shared_expert is not None:
                    raise NotImplementedError("HostExpertBank does not offload Qwen2-MoE shared experts")
                layers[i] = (mlp.router.weight, mlp.gate_up, mlp.down)
        if not layers:
            raise ValueError("the model has no SparseMoE layers")
        return cls(layers, pin_memory)

    def router(self, layer_idx: int) -> torch.Tensor:
        return self.layers[layer_idx][0]

    def expert(self, layer_idx: int, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(gate [F, H], up [F, H], down [H, F]) of one expert, on the host."""
        _, gate_up, down = self.layers[layer_idx]
        gate, up = gate_up[expert_idx].chunk(2, dim=0)
        return gate, up, down[expert_idx]

    def slot_weights(self, layer_idx: int, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One expert in GPULRUExpertCache's slot layout: gate [H, F], up [H, F], down [F, H]."""
        gate, up, down = self.expert(layer_idx, expert_idx)
        return gate.t(), up.t(), down.t()


class HeterogeneousMoERunner(nn.Module):
    """
    Runs a model's MoE layers from a HostExpertBank. For each expert a batch is routed to: a resident cache slot
    runs it on the device; otherwise the decider either copies the expert into a slot (evicting by the cache's
    policy) and runs it there, or runs it on the host CPU. The output equals SparseMoE's for the same weights.
    """
    def __init__(
        self,
        host_bank: HostExpertBank,
        cache: GPULRUExpertCache,
        decider: BandwidthAdaptiveDecider,
        top_k: int,
        norm_topk_prob: bool,
        act=F.silu,
    ):
        super().__init__()
        if (cache.hidden_dim, cache.active_dim) != (host_bank.hidden_dim, host_bank.ffn_dim):
            raise ValueError(f"cache slots are [{cache.hidden_dim}, {cache.active_dim}], experts are "
                             f"[{host_bank.hidden_dim}, {host_bank.ffn_dim}]")
        self.host_bank, self.cache, self.decider = host_bank, cache, decider
        self.top_k, self.norm_topk_prob, self.act = top_k, norm_topk_prob, act

    @classmethod
    def from_model(cls, model: nn.Module, cache: GPULRUExpertCache, decider: BandwidthAdaptiveDecider,
                   pin_memory: bool = False) -> "HeterogeneousMoERunner":
        from ..models.causal_lm import activation
        cfg = model.config
        return cls(HostExpertBank.from_model(model, pin_memory), cache, decider, cfg.num_experts_per_tok,
                   cfg.norm_topk_prob, activation(cfg.hidden_act))

    def _mlp(self, x, gate, up, down):
        """Gated MLP with weights in slot layout: gate, up [H, F], down [F, H]."""
        return torch.matmul(self.act(torch.matmul(x, gate)) * torch.matmul(x, up), down)

    def route_and_execute(self, hidden_states: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """The MoE block of layer layer_idx on hidden_states [..., H]; returns (output, per-call dispatch counts)."""
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        router = self.host_bank.router(layer_idx).to(device=x.device, dtype=x.dtype)
        probs = F.softmax(F.linear(x, router).float(), dim=-1)
        weights, experts = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        stats = {"experts": 0, "cache_hits": 0, "loaded_to_device": 0, "host_executed": 0}
        c = self.cache
        for e in torch.unique(experts).tolist():
            token, slot = torch.where(experts == e)
            stats["experts"] += 1
            idx = c.get_slot(layer_idx, e)
            if idx is not None:
                stats["cache_hits"] += 1
            elif self.decider.should_stream_to_gpu(self.host_bank.bytes_per_expert, len(token),
                                                   self.host_bank.hidden_dim, self.host_bank.ffn_dim):
                idx, _ = c.allocate_or_evict_slot(layer_idx, e)
                if idx is not None:  # None: the LFRU gate kept the resident experts, so this one runs on the host
                    c.load_expert_weights(idx, *self.host_bank.slot_weights(layer_idx, e))
                    stats["loaded_to_device"] += 1
            if idx is not None:
                y = self._mlp(x[token].to(c.device, c.dtype), c.slot_gate_weights[idx], c.slot_up_weights[idx],
                              c.slot_down_weights[idx])
            else:
                y = self._mlp(x[token].to("cpu", torch.float32),
                              *(w.float() for w in self.host_bank.slot_weights(layer_idx, e)))
                stats["host_executed"] += 1
            out.index_add_(0, token, (y.to(x.device) * weights[token, slot, None]).to(out.dtype))
        return out.view(shape), stats
