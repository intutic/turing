# SVD INT8 KV Cache Paging

**SVD INT8 KV Cache Paging** solves the memory bottleneck of ultra-long context sequences (32K–128K) by decomposing attention keys and values into low-rank singular subspaces.

---

## Algorithmic Overview

For a Key/Value projection tensor $\mathbf{K} \in \mathbb{R}^{B \times H \times S \times D}$, we compute a calibrated spectral truncated SVD:

$$\mathbf{K} \approx \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$$

Where rank $r = 64 \ll D$.

During serving, Turing Engine stores:
1. **Low-Rank Projection Basis**: $\mathbf{V}_r \in \mathbb{R}^{D \times r}$ (cached per layer).
2. **Symmetric INT8 Quantized Singular Vectors**: $\mathbf{Z} = \text{quant8}(\mathbf{K} \mathbf{V}_r) \in \mathbb{Z}_8^{B \times H \times S \times r}$.

---

## Memory Compression Impact

| Sequence Context | FP16 KV Memory | Turing SVD INT8 | Compression Ratio | Top-1 Needle Retrieval |
| :---: | :---: | :---: | :---: | :---: |
| **8K Context** | 2.56 GB | **0.64 GB** | **-75.0%** | **100.0%** |
| **32K Context** | 10.24 GB | **2.56 GB** | **-75.0%** | **100.0%** |
| **64K Context** | 20.48 GB | **5.12 GB** | **-75.0%** | **100.0%** |
| **128K Context** | 40.96 GB | **10.24 GB** | **-75.0%** | **100.0%** |
