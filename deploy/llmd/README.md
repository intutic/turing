# Turing Engine on llm-d (Kubernetes Distributed LLM Inference)

This directory contains manifests and configuration for deploying **Turing Engine** as a high-performance model server backend within an [**llm-d**](https://llm-d.ai) Kubernetes cluster. The full guide is [`docs/llmd-integration.md`](../../docs/llmd-integration.md).

---

## 🚀 What the integration does

Turing Engine pods keep an automatic prefix cache and, with `--enable-kv-events`, publish its block creation and
eviction events over ZeroMQ (`PUB` on port 5556, `ROUTER` replay on port 5559) in vLLM's msgpack KV-event format.
llm-d's endpoint picker indexes them and routes each prompt to the pod holding its longest cached prefix.

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
    -n llm-d-serving --create-namespace
  ```

### 2. Deploy Turing Engine with llm-d Compatibility

Deploy using the Turing Engine Helm chart with `llmd.enabled=true`:

```bash
helm install turing-model deploy/helm/turing-serving/ \
  --namespace llm-d-serving \
  --set llmd.enabled=true \
  --set llmd.guide=optimized-baseline \
  --set llmd.engineType=turing \
  --set model.name="Qwen/Qwen2.5-7B-Instruct" \
  --set replicaCount=2
```

### 3. Apply the InferencePool

```bash
kubectl apply -f deploy/llmd/inferencepool.yaml
```

### 4. Verify Routing & Health

Check pod readiness:
```bash
kubectl get pods -n llm-d-serving -l llm-d.ai/engine-type=turing
```

Test inference through the llm-d EPP router:
```bash
ROUTER_IP=$(kubectl get svc -n llm-d-serving llm-d-router-epp -o jsonpath='{.spec.clusterIP}')

curl -X POST "http://${ROUTER_IP}:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "Explain prefix-cache routing in Kubernetes."
  }'
```

---

## 📊 Metrics Reference for EPP Scorers

`GET /metrics` returns Prometheus text; the gauges llm-d's scorers read:

| Turing Prometheus Metric | llm-d EPP Semantic Mapping | Description |
|---|---|---|
| `turing_num_requests_waiting` | `TotalQueuedRequests` | Number of waiting requests in admission queue |
| `turing_num_requests_running` | `TotalRunningRequests` | Number of active requests executing in batch |
| `turing_kv_cache_usage_perc` | `KVCacheUtilization` | Fraction of static KV page pool in use (0.0–1.0) |
| `turing_cache_config_info` | `BlockSize`, `NumGPUBlocks` | Block size (16 tokens, the KV page) and pages in the pool |
