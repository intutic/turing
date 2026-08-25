# 💻 CLI Commands Reference

Turing Engine provides a unified command-line tool `turing` for instant terminal chat, continuous batch serving, text generation, offline conversion, and accuracy evaluation.

---

## Available Commands

### 1. `turing chat`
Start an interactive terminal chat with real pretrained weights:
```bash
# Chat with SmolLM2 or DeepSeek-R1 in your terminal:
turing chat --model smollm2
turing chat --model deepseek-r1-1.5b
turing chat --model qwen3-coder-30b
```

### 2. `turing serve`
Start the continuous batching inference server with OpenAI and Anthropic-compatible endpoints:
```bash
turing serve \
  --model deepseek-r1-7b \
  --port 8000 \
  --host 0.0.0.0 \
  --device auto \
  --sparsity 0.50 \
  --max-batch-size 32
```

### 3. `turing generate`
Generate text from a single prompt:
```bash
turing generate --model gpt2 --prompt "Artificial intelligence is" --max-tokens 64
```

### 4. `turing eval-accuracy`
Run live mathematical reasoning (GSM8K) accuracy evaluation:
```bash
turing eval-accuracy --model gpt2 --samples 10
```

### 5. `turing bench`
Run high-precision latency and throughput benchmarks:
```bash
turing bench --model qwen3-coder-30b --batch-size 16 --seq-len 2048
```

### 6. `turing convert`
Convert HuggingFace safetensors checkpoints into Subspace-pruned format:
```bash
turing convert --input ./llama-3.3-70b --output ./llama-3.3-70b-subspace --sparsity 0.57
```
