# Birkhoff Doubly Stochastic Hyper-Connections

To maintain numerical stability and prevent activation explosion across 80+ transformer layers during high-sparsity routing, Turing Engine implements **Birkhoff Doubly Stochastic Hyper-Connections**.

---

## Sinkhorn-Knopp Regularization

The routing matrix $\mathbf{P}$ across layer skip connections is constrained to the Birkhoff polytope $\mathcal{B}_n$:

$$\sum_{j=1}^N P_{ij} = 1 \quad \text{and} \quad \sum_{i=1}^N P_{ij} = 1$$

We iteratively project logits using the in-SRAM Sinkhorn-Knopp algorithm:

$$\mathbf{u}^{(t+1)} = \frac{\mathbf{1}}{\mathbf{K} \mathbf{v}^{(t)}}, \quad \mathbf{v}^{(t+1)} = \frac{\mathbf{1}}{\mathbf{K}^T \mathbf{u}^{(t+1)}}$$

This guarantees 0 gradient divergence and uniform activation bounds across multi-thousand token contexts.
