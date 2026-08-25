# Heterogeneous MoE Memory Management

Heterogeneous Mixture-of-Experts (MoE) execution allows serving massive models like **DeepSeek-V4-Flash-284B** (284 Billion total parameters) on a single 24GB GPU.

---

## Memory Tiering Architecture

```mermaid
flowchart TD
    subgraph HostDRAM["Host System DRAM (32GB - 64GB)"]
        A["All 284B MoE Expert Weights"]
    end

    subgraph PCIe["PCIe Gen4 / Gen5 Bus (Double-Buffered Ring)"]
        B["Async D2H / H2D DMA Engine"]
    end

    subgraph GPUVRAM["NVIDIA GPU VRAM (24GB L4 / RTX 4090)"]
        C["Shared Attention Backbone (5.91 GB)"]
        D["Active Expert LRU Cache (14.00 GB)"]
        E["SVD INT8 KV Cache (2.56 GB)"]
    end

    A <--> B <--> D
```

- **Top-2 Active Experts**: Dynamic LRU cache in GPU VRAM achieves **>80% cache hit rate** across sequential generation.
- **Asynchronous PCIe Double Buffering**: Loads next-token candidate experts over DMA during current-token attention execution to hide bus latency.
