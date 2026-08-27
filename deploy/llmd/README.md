# Turing Engine on llm-d (Kubernetes Distributed LLM Inference)

This directory contains manifests and configuration for deploying **Turing Engine** as a high-performance model server backend within an **llm-d** (CNCF Sandbox) Kubernetes cluster.

---

## 🚀 Key Advantages of Turing Engine with llm-d

1. **Precise Prefix-Cache Aware Routing**: Turing Engine pods publish real-time KV block creation and eviction events over ZeroMQ (`PUB` on port 5556, `ROUTER` replay on port 5559). llm-d's Endpoint Policy Provider (EPP) builds a live token hash index to route prompt prefixes to pods holding cached states.
2. **SVD-Compressed KV Transfer Wire Format**: Cross-pod KV transfers utilize Turing's Rank-64 INT8 Subspace projection, reducing network payload size by **~75%** compared to standard FP16 transfers.
3. **High Pod Density**: Turing's active-subspace activation gating and hierarchical virtual memory paging enable up to **4× higher model concurrency** per GPU node.

---

## 📦 Quickstart Deployment

### 1. Prerequisites
- Kubernetes cluster with NVIDIA GPU nodes (e.g. GKE L4 / A100 / H100)
- Gateway API Inference Extension (GAIE) installed:
  ```bash
  kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/latest/download/manifests.yaml
  ```
- llm-d router installed via Helm:
  ```bash
  helm install llm-d-router oci://ghcr.io/llm-d/charts/llm-d-router-standalone \
    -n llm-d-optimized-baseline --create-namespace
  ```

### 2. Deploy Turing Engine with llm-d Compatibility

Deploy using the Turing Engine Helm chart with `llmd.enabled=true`:

```bash
helm install turing-model deploy/helm/turing-serving/ \
  --namespace llm-d-optimized-baseline \
  --set llmd.enabled=true \
  --set llmd.guide=optimized-baseline \
  --set llmd.engineType=turing \
  --set model.name="meta-llama/Llama-3.1-70B-Instruct" \
  --set replicaCount=2
```

### 3. Apply the InferencePool

```bash
kubectl apply -f deploy/llmd/inferencepool.yaml
```

### 4. Verify Routing & Health

Check pod readiness:
```bash
kubectl get pods -n llm-d-optimized-baseline -l llm-d.ai/engine-type=turing
```

Test inference through the llm-d EPP router:
```bash
ROUTER_IP=$(kubectl get svc -n llm-d-optimized-baseline optimized-baseline-epp -o jsonpath='{.spec.clusterIP}')

curl -X POST "http://${ROUTER_IP}:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "prompt": "Explain prefix-cache routing in Kubernetes."
  }'
```

---

## 📊 Metrics Reference for EPP Scorers

Turing Engine exports Prometheus metrics on `GET /metrics` formatted for llm-d EPP plugins:

| Turing Prometheus Metric | llm-d EPP Semantic Mapping | Description |
|---|---|---|
| `turing_num_requests_waiting` | `TotalQueuedRequests` | Number of waiting requests in admission queue |
| `turing_num_requests_running` | `TotalRunningRequests` | Number of active requests executing in batch |
| `turing_kv_cache_usage_perc` | `KVCacheUtilization` | Fraction of static KV page pool in use (0.0–1.0) |
| `turing_cache_config_info` | `BlockSize`, `NumGPUBlocks` | Block size tokens (64) and total allocated blocks |
