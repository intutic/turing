# KServe upstream status

kserve/kserve#6074 (adding `turing-runtime` to `config/runtimes/`) was closed as stale on 2026-09-30 without review.
Do not reopen it until both hold:

1. A Turing CUDA image is published (the PR pointed at `ghcr.io/intutic/turing:v0.1.0-cuda`, which never existed).
   Done: `ghcr.io/intutic/turing:v2.0.0-cuda` (2026-11-20).
2. `serving_runtime.yaml` and `inferenceservice.yaml` have been applied on a real cluster with KServe and a GPU node,
   and a `/v1/chat/completions` request through the InferenceService succeeded. Record the KServe and Kubernetes
   versions in the PR.

The PR's description also claimed cluster validation and 70B models on one GPU; neither was true. A new PR describes
only what was run.
