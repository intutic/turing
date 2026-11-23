# Docker

Turing Engine publishes a CPU image and a CUDA image to the GitHub Container Registry.

| Image | Platforms | Built from |
|---|---|---|
| `ghcr.io/intutic/turing:<version>` | linux/amd64, linux/arm64 | CPU PyTorch |
| `ghcr.io/intutic/turing:<version>-cuda` | linux/amd64 | PyTorch for CUDA 12.8 |

`<version>` is the release tag, for example `v2.0.0`; `latest` and `latest-cuda` follow the newest release. The
release workflow (`.github/workflows/release.yml`) publishes both images when a version tag is pushed, starting with
v2.0.0.

The container runs `turing serve` on port 8000 as a non-root user (`turing`). Arguments after the image name are
passed to `turing serve` (default: `--model smollm2`).

## CPU

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/home/turing/.cache/huggingface \
  ghcr.io/intutic/turing:latest --model HuggingFaceTB/SmolLM2-135M
```

The volume reuses your HuggingFace download cache, so weights are fetched once. Without it the container downloads
the model on every start. Pass `-e HF_TOKEN=...` for gated models.

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "smollm2", "messages": [{"role": "user", "content": "The capital of France is"}], "max_tokens": 8}'
```

## CUDA

Needs an NVIDIA driver recent enough for CUDA 12.8 and the NVIDIA Container Toolkit on the host.

!!! warning "The `v2.0.0-cuda` image does not start on a GPU"
    It has no C compiler, which Triton needs at its first kernel (`Failed to find C compiler`: on GKE and under
    `docker run --gpus all` on an L4, 2026-11-23). `deploy/Dockerfile` now installs one and sets the library path
    GKE mounts the driver at, and an image built from it served requests on that L4. Until one is published,
    build the CUDA image yourself:
    `docker build -f deploy/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t turing:cuda .`

```bash
docker run --rm --gpus all -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/home/turing/.cache/huggingface \
  ghcr.io/intutic/turing:latest-cuda --model HuggingFaceTB/SmolLM2-360M-Instruct --device cuda
```

Every `turing serve` flag works after the image name, for example `--kv-cache-dtype int4` or `--max-batch-size 32`.
Run `docker run --rm ghcr.io/intutic/turing:latest --help` for the list.

## Building locally

One Dockerfile, `deploy/Dockerfile`, builds both images. They differ only in the PyTorch wheels: the CUDA image
installs PyTorch's CUDA 12.8 build, which bundles the CUDA runtime.

```bash
docker build -f deploy/Dockerfile -t turing:cpu .
docker build -f deploy/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t turing:cuda .
```

The build compiles the C++ extension into a wheel in a first stage, so the CPU image carries no compiler. The CUDA
image carries `gcc`: Triton compiles its kernels on the GPU node at their first use and needs it.
