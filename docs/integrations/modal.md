# Modal Labs Serverless Deployment Guide

Deploy Turing Engine on serverless single-GPU infrastructure (NVIDIA L4 / A10G) with automatic scale-to-zero.

---

## 1. Deploy to Modal

```bash
pip install modal
modal deploy integrations/modal/modal_turing.py
```

## 2. Run Test Prompt

```bash
modal run integrations/modal/modal_turing.py --prompt "Explain the Birkhoff hyper-connections algorithm:"
```
