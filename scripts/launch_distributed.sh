#!/usr/bin/env bash
# Serve one model across TP x PP processes with torchrun, one process per GPU (gloo on CPU-only hosts). It runs as
# $PYTHON -m torch.distributed.run so every rank uses the interpreter turing is installed in.
# Rank 0 runs the HTTP server and the engine; every other rank runs DistributedInferenceDriver.worker_loop().
# Usage: scripts/launch_distributed.sh MODEL [TP] [PP]
# Multi-node: set NUM_NODES, NODE_RANK and MASTER_ADDR on every node; TP x PP must equal NUM_NODES x GPUs per node.
set -euo pipefail

MODEL="${1:?usage: $0 MODEL [TP] [PP]}"
TP_SIZE="${2:-2}"
PP_SIZE="${3:-1}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

WORLD=$(( TP_SIZE * PP_SIZE ))
if (( WORLD % NUM_NODES )); then
    echo "TP x PP (${WORLD}) must be a multiple of NUM_NODES (${NUM_NODES})" >&2
    exit 1
fi
PER_NODE=$(( WORLD / NUM_NODES ))

echo "turing serve ${MODEL}: TP ${TP_SIZE} x PP ${PP_SIZE} = ${WORLD} ranks, ${PER_NODE} on node ${NODE_RANK}/${NUM_NODES}," \
     "rendezvous ${MASTER_ADDR}:${MASTER_PORT}, HTTP on rank 0 at ${HOST}:${PORT}"

exec "${PYTHON:-python}" -m torch.distributed.run \
    --nproc_per_node="${PER_NODE}" \
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
