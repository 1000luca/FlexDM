#!/usr/bin/env bash
# Run FlexDM with the Decision Index kit on the GPU that has the most free memory.
#
#   ./run_flexdm.sh OUT_DIR [more `decision_index run` arguments]
#
#   ./run_flexdm.sh runs/flexdm                                 # the public 0.3 suite in ./suite-0.3, as submitted
#   ./run_flexdm.sh runs/flexdm-private --rows private.jsonl.gz # any other rows file
#
# Use the venv from README.md (activated, or PYTHON=/path/to/venv/bin/python) with the model weights already in the
# Hugging Face cache. Set CUDA_VISIBLE_DEVICES yourself to skip the GPU choice. The kit resumes an existing OUT_DIR.
set -euo pipefail

OUT=${1:?usage: run_flexdm.sh OUT_DIR [more decision_index run arguments]}
shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
MIN_FREE_MIB=${MIN_FREE_MIB:-72000}     # vLLM takes gpu_memory_utilization=0.70 of a 96 GB card at start, plus a margin

BIN=$(dirname "$(command -v "$PYTHON")")
export PATH="$BIN:$PATH"                 # FlashInfer's first-run JIT calls ninja, which is installed in the venv
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_HUB_DISABLE_XET=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  read -r GPU FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                         | tr -d ' ' | sort -t, -k2,2nr | head -1 | tr ',' ' ')
  if [ "${FREE:-0}" -lt "$MIN_FREE_MIB" ]; then
    echo "no GPU has $MIN_FREE_MIB MiB free (best: GPU ${GPU:-?} with ${FREE:-0} MiB)" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "using GPU $GPU ($FREE MiB free)" >&2
fi

OPTIONS=(layout=header read_twice=true twice_max_tokens=3000
         temperature_choice=10.9757 temperature_choice_log_slope=-1.7016
         temperature_noul=10.732 noul_bias=0)
ARGS=()
for kv in "${OPTIONS[@]}"; do ARGS+=(--option "$kv"); done

exec "$PYTHON" -m decision_index run --engine jevlab.vllm_reader:VllmReader "${ARGS[@]}" --compact --out "$OUT" "$@"
