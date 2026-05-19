#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "run" ]]; then
  echo "Usage: bash scripts/run_segment_level_fourier_aware.sh run"
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

OUT_DIR="outputs/segment_level_fourier_aware"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

BATCH_TCN="${BATCH_TCN:-320}"
BATCH_ATTENTION="${BATCH_ATTENTION:-250}"

COMMON_ARGS=(
  --data_dir data/LADS
  --device cuda:0
  --fourier_mode dynamic
  --warm_start fourier
  --input_mode fourier_aware
  --train_sampling fixed
  --stride_eval 1
  --max_eval_per_segment 0
  --lambda_visible_residual 0.05
)

echo "Running TCN Fourier-aware clean split..."
python -u -m scripts.run_all_lads_hybrid_tcn \
  "${COMMON_ARGS[@]}" \
  --batch_train "$BATCH_TCN" \
  --batch_eval "$BATCH_TCN" \
  --out_csv "$OUT_DIR/tcn_fourier_aware_dynamic_clean_l0_05.csv" \
  2>&1 | tee "$LOG_DIR/tcn_fourier_aware_dynamic_clean.log"

# echo "Running Transformer Fourier-aware clean split..."
# python -u -m scripts.run_all_lads_hybrid_transformer \
#   "${COMMON_ARGS[@]}" \
#   --batch_train "$BATCH_ATTENTION" \
#   --batch_eval "$BATCH_ATTENTION" \
#   --out_csv "$OUT_DIR/transformer_fourier_aware_dynamic_clean.csv" \
#   2>&1 | tee "$LOG_DIR/transformer_fourier_aware_dynamic_clean.log"

# echo "Running Conv-Transformer Fourier-aware clean split..."
# python -u -m scripts.run_all_lads_hybrid_conv_transformer \
#   "${COMMON_ARGS[@]}" \
#   --batch_train "$BATCH_ATTENTION" \
#   --batch_eval "$BATCH_ATTENTION" \
#   --out_csv "$OUT_DIR/conv_transformer_fourier_aware_dynamic_clean.csv" \
#   2>&1 | tee "$LOG_DIR/conv_transformer_fourier_aware_dynamic_clean.log"
