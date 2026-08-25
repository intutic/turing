# Speculative Quadtree MRP Decoding

**Speculative Quadtree Markov Random Process (MRP) Decoding** accelerates autoregressive generation by speculating multiple future tokens concurrently in a hierarchical 2D tree topology.

---

## Quadtree Tree Verification

```mermaid
graph TD
    Root["Current Token"] --> Q1["Branch 1 (Top-Left)"]
    Root --> Q2["Branch 2 (Top-Right)"]
    Root --> Q3["Branch 3 (Bottom-Left)"]
    Root --> Q4["Branch 4 (Bottom-Right)"]
    Q1 --> L1["Leaf 1.1"]
    Q1 --> L2["Leaf 1.2"]
    Q2 --> L3["Leaf 2.1"]
    Q2 --> L4["Leaf 2.2"]
```

1. **Lightweight Draft Head**: Computes 4 candidate branches in parallel using a low-overhead linear projection.
2. **Fused Flash-Tree Attention**: The target model evaluates the entire tree candidate set in a **single forward pass** using custom attention tree-masking.
3. **Acceptance Rate**: Achieves an average acceptance rate of **2.8 to 3.4 tokens per forward step**, doubling throughput without altering final probability distributions.
