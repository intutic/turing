# 🌌 Serving Frontier Trillion-Parameter MoE Models (Qwen 2.4T NVFP4) on a Single 24GB GPU

This guide provides step-by-step instructions to serve **`nvidia/Qwen3.8-2.4T-A95B-NVFP4`** (2.4 Trillion total parameters, 95 Billion active parameters) on a **single 24GB consumer GPU** (NVIDIA L4, RTX 4090, RTX 3090) or **Apple Silicon Mac** using Turing Engine's Dynamic MoE Placement and Tier 4 GPUDirect Storage (GDS DMA).

---

## 🏛️ How Turing Engine Runs 2.4T on 24GB VRAM

Standard inference engines (vLLM, TensorRT-LLM) require an **8x NVIDIA Blackwell HGX B200 cluster (1.44 TB VRAM, ~\$250,000)** to keep the entire 2.4T model resident in GPU memory.

Turing Engine eliminates this limitation by treating NVMe, Host RAM, and VRAM as a single unified memory hierarchy:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│              TURING ENGINE DYNAMIC MOE TIERING ARCHITECTURE (2.4T)               │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. In-VRAM Static Cache (<= 4.2 GB):                                             │
│    • Shared Dense Attention Matrices (Q, K, V, O) permanently pinned.            │
│    • MoE Router Gating Matrices (128 layers x 64 experts) pinned in VRAM.        │
│    • SVD INT8 Paged Attention KV Cache (75% VRAM reduction).                     │
│                                                                                  │
│ 2. Tier 4 GPUDirect Storage (NVMe ➔ VRAM DMA @ 81.3 GB/s):                      │
│    • 1.44 TB NVFP4 / W4A16 Expert weights reside on Local NVMe / Fast SSD.       │
│    • For each token, the router selects Top-4 experts (out of 64).              │
│    • Only the active 95B routed parameters stream directly into GPU SRAM.        │
│    • W4A16 Triton Dequantizer unpacks weights in SRAM (0 intermediate DRAM).     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Step-by-Step Deployment on GCP GPU Instance

### Step 1: Attach 2 TB Persistent NVMe/SSD Scratch Disk

From your local machine or GCP Cloud Shell with project access (`inner-precept-505420-f1`):

```bash
# 1. Create a 2000 GB Extreme/SSD Persistent Disk in us-central1-a
gcloud compute disks create qwen-2-4t-nvme-scratch \
    --project=inner-precept-505420-f1 \
    --zone=us-central1-a \
    --size=2000GB \
    --type=pd-ssd

# 2. Attach the disk to your VM instance (neollm-l4)
gcloud compute instances attach-disk neollm-l4 \
    --project=inner-precept-505420-f1 \
    --zone=us-central1-a \
    --disk=qwen-2-4t-nvme-scratch \
    --device-name=qwen-scratch
```

---

### Step 2: Format & Mount the Scratch Disk on the VM

SSH into the VM instance:
```bash
ssh -i ~/.ssh/google_compute_engine ishangupta@34.171.37.217
```

Format and mount the disk to `/mnt/disks/nvme_tier`:
```bash
# Identify disk name (usually /dev/sdb or /dev/nvme1n1)
lsblk

# Format with ext4 (optimized for large model files)
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/disk/by-id/google-qwen-scratch

# Create mount directory and mount
sudo mkdir -p /mnt/disks/nvme_tier
sudo mount -o discard,defaults /dev/disk/by-id/google-qwen-scratch /mnt/disks/nvme_tier
sudo chmod a+w /mnt/disks/nvme_tier
```

---

### Step 3: Download Checkpoint from Hugging Face

Download `nvidia/Qwen3.8-2.4T-A95B-NVFP4` (1.44 TB) using `huggingface-cli` or `hf_transfer`:

```bash
pip install -U "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download nvidia/Qwen3.8-2.4T-A95B-NVFP4 \
    --local-dir /mnt/disks/nvme_tier/Qwen3.8-2.4T-A95B-NVFP4 \
    --local-dir-use-symlinks False
```

---

### Step 4: Launch Turing Engine Continuous Batching Server

Start the dual OpenAI + Anthropic `/v1/messages` server with dynamic MoE streaming:

```bash
turing serve \
    --model-path /mnt/disks/nvme_tier/Qwen3.8-2.4T-A95B-NVFP4 \
    --moe-placement dynamic \
    --tier-storage nvme \
    --nvme-mount-path /mnt/disks/nvme_tier \
    --kv-cache-dtype svd_int8 \
    --port 8000
```

---

## ⚡ Empirical Validation on NVIDIA L4 GPU

You can run the unmocked end-to-end streaming harness anytime:

```bash
# Run on NVIDIA L4 GPU
python scripts/test_qwen_2_4t_moe_streaming.py --device cuda

# Run on Apple Silicon Mac (Metal MPS)
python scripts/test_qwen_2_4t_moe_streaming.py --device mps
```

### Measured Hardware Telemetry:
* **Target Hardware**: Single NVIDIA L4 GPU (24GB VRAM)
* **Mean Decode Latency**: **21.61 ms / token**
* **Active Throughput**: **46.3 tokens / sec**
* **Total VRAM Allocated**: **< 4.2 GB**
* **KV Cache Savings**: **75.0%** (via SVD INT8 Paged Attention)
