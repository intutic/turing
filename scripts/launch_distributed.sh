#!/usr/bin/env bash
# ==============================================================================
# Multi-Node & Multi-GPU Launch Script for Turing Engine Serving
# Shards large models across TP ranks and PP stages using PyTorch NCCL / torchrun
# ==============================================================================

set -euo pipefail

MODEL="${1:-test-tiny}"
TP_SIZE="${2:-2}"
PP_SIZE="${3:-1}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

TOTAL_GPUS=$(( TP_SIZE * PP_SIZE ))
GPUS_PER_NODE=$(( TOTAL_GPUS / NUM_NODES ))

echo "======================================================================"
echo "🚀 Launching Distributed Turing Engine Serving Runtime"
echo "======================================================================"
echo "  Model Identifier     : ${MODEL}"
echo "  Tensor Parallel (TP) : ${TP_SIZE}"
echo "  Pipeline Stages (PP) : ${PP_SIZE}"
echo "  Total GPUs           : ${TOTAL_GPUS} (${GPUS_PER_NODE} per node)"
echo "  Cluster Node Rank    : ${NODE_RANK} / ${NUM_NODES}"
echo "  Master Address       : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  HTTP Serving Port    : http://${HOST}:${PORT}"
echo "======================================================================"

torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m turing.cli serve \
    --model "${MODEL}" \
    --tensor-parallel "${TP_SIZE}" \
    --pipeline-parallel "${PP_SIZE}" \
    --host "${HOST}" \
    --port "${PORT}"
