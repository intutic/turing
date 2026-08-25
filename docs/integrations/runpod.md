# RunPod Serverless Deployment Guide

Run Turing Engine as a low-cost serverless worker on RunPod GPUs.

---

## 1. Build and Push Container

```bash
docker build -f deploy/Dockerfile.cuda -t your-dockerhub-user/turing-runpod:v0.1.0 .
docker push your-dockerhub-user/turing-runpod:v0.1.0
```

## 2. RunPod Template Configuration
- **Container Image**: `your-dockerhub-user/turing-runpod:v0.1.0`
- **Container Disk**: 40 GB
- **GPU Type**: 1x NVIDIA L4 (24GB) or RTX 4090
- **Start Command**: `python3 -m integrations.runpod.runpod_handler`
