# KServe Kubernetes Deployment Guide

[KServe](https://kserve.github.io/website/) provides standardized, cloud-native model serving on Kubernetes with autoscaling, scale-to-zero, and multi-model mesh support.

---

## 1. Apply Turing Engine `ServingRuntime`

Apply the custom resource definition from [`serving_runtime.yaml`](https://github.com/intutic/turing/blob/master/integrations/kserve/serving_runtime.yaml):

```bash
kubectl apply -f integrations/kserve/serving_runtime.yaml
```

## 2. Deploy an `InferenceService`

```bash
kubectl apply -f integrations/kserve/inferenceservice.yaml
```

## 3. Test the Endpoint

```bash
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')

curl -v -H "Host: turing-llama-70b.default.example.com" \
  -H "Content-Type: application/json" \
  http://${INGRESS_HOST}:${INGRESS_PORT}/v1/chat/completions \
  -d '{
    "model": "llama-3.1-70b",
    "messages": [{"role": "user", "content": "Explain Turing Engine on Kubernetes."}]
  }'
```
