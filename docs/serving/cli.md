# CLI Commands Reference

Turing Engine provides a unified command-line tool `turing`.

---

## Available Commands

### `turing serve`
Start the continuous batching inference server:
```bash
turing serve \
  --model llama-3.1-70b \
  --port 8000 \
  --host 0.0.0.0 \
  --device auto \
  --sparsity 0.57 \
  --max-batch-size 32
```

### `turing bench`
Run high-precision latency and throughput benchmarks:
```bash
turing bench --model llama-3.1-70b --batch-size 16 --seq-len 2048
```

### `turing convert`
Convert HuggingFace safetensors checkpoints into Subspace-pruned format:
```bash
turing convert --input ./llama-3.1-70b --output ./llama-3.1-70b-subspace --sparsity 0.57
```
