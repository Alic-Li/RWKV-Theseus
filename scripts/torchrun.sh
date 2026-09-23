#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec python -m torch.distributed.run --standalone --nproc_per_node="${GPUS:-2}" train.py --config configs/train.json "$@"
