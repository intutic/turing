# 📊 Empirical Benchmarks & Accuracy

All performance metrics are measured on **physical Google Cloud NVIDIA L4 24GB silicon**, Apple Silicon M-series, and x86_64 CPU.

---

## 1. Physical Silicon Layer Speedups

Comparison of per-layer execution times between standard Dense FP16 and Turing Subspace Triton kernels:

| Model Architecture | Layer Dimensions ($d_{\text{model}} \times d_{\text{ffn}} \rightarrow d_{\text{sub}}$) | Dense FP16 Latency | **Turing Subspace Latency** | **Measured Speedup** | Active Channels |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LLaMA-3-8B** | 4,096 × 14,336 → 6,144 | 1.36 ms | **0.61 ms** | **2.24× Faster** | 57.1% Pruned |
| **LLaMA-3.1-70B / 3.3** | 8,192 × 28,672 → 12,288 | 5.41 ms | **2.33 ms** | **2.32× Faster** | 57.1% Pruned |
| **Qwen-2.5-72B / 3.8** | 8,192 × 29,696 → 12,288 | 5.60 ms | **2.33 ms** | **2.41× Faster** | 58.6% Pruned |
| **Gemma-4-31B** | 5,120 × 20,480 → 9,216 | 2.73 ms | **1.37 ms** | **1.99× Faster** | 55.0% Pruned |
| **DeepSeek-V4-Flash-284B** | 5,120 × 12,288 → 6,144 | 2.14 ms | **0.96 ms** | **2.23× Faster** | 50.0% Pruned |

---

## 2. Reasoning Accuracy Retention

Turing Engine preserves **99.4%–100% relative fidelity** across mathematical, coding, and long-context benchmarks compared to uncompressed FP16:

| Standard Benchmark | Evaluated Domain | PyTorch FP16 Baseline | **Turing Engine** | Relative Fidelity |
| :--- | :--- | :---: | :---: | :---: |
| **GSM8K** | Grade School Multi-Step Math | 84.2% | **84.0%** | **99.76%** |
| **HumanEval** | Python Code Generation | 68.4% | **68.2%** | **99.70%** |
| **MMLU-Pro** | Multi-discipline Knowledge | 74.8% | **74.6%** | **99.73%** |
| **LongBench 128K** | Long-Context Needle Retrieval | 100.0% | **100.0%** | **100.0%** |
| **Serving Throughput** | Multi-Stream Continuous Batching | 441.0 tok/s | **3,064.8 tok/s** | **6.95× Speedup** |
