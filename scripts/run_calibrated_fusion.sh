#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "run" ]]; then
  echo "Usage:"
  echo "  bash scripts/run_calibrated_fusion.sh run"
  echo ""
  echo "Optional environment variables:"
  echo "  ARCHS=\"tcn transformer conv_transformer unet1d\""
  echo "  SEEDS=\"123 456 789\""
  echo "  CUDA_VISIBLE_DEVICES=0"
  echo "  BATCH_TCN=150 BATCH_TRANSFORMER=200 BATCH_CONV_TRANSFORMER=200 BATCH_UNET=200"
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ARCHS="${ARCHS:-tcn transformer conv_transformer unet1d}"
SEEDS="${SEEDS:-123 456 789}"

LAMBDA_VISIBLE_RESIDUAL="${LAMBDA_VISIBLE_RESIDUAL:-0.1}"
ALPHA_STEPS="${ALPHA_STEPS:-21}"
FOURIER_MODE="${FOURIER_MODE:-dynamic}"
DEVICE="${DEVICE:-cuda:0}"

run_one () {
  local arch="$1"
  local seed="$2"

  local module=""
  local out_dir=""
  local prefix=""
  local batch=""
  local extra_args=()

  case "$arch" in
    tcn)
      module="scripts.run_all_lads_hybrid_tcn_calibrated_fusion"
      out_dir="outputs/segment_level_calibrated_fusion_tcn"
      prefix="tcn"
      batch="${BATCH_TCN:-150}"
      ;;
    transformer)
      module="scripts.run_all_lads_hybrid_transformer_calibrated_fusion"
      out_dir="outputs/segment_level_calibrated_fusion_transformer"
      prefix="transformer"
      batch="${BATCH_TRANSFORMER:-200}"
      ;;
    conv_transformer)
      module="scripts.run_all_lads_hybrid_conv_transformer_calibrated_fusion"
      out_dir="outputs/segment_level_calibrated_fusion_conv_transformer"
      prefix="conv_transformer"
      batch="${BATCH_CONV_TRANSFORMER:-200}"
      ;;
    unet1d)
      module="scripts.run_all_lads_hybrid_unet1d_calibrated_fusion"
      out_dir="outputs/segment_level_calibrated_fusion_unet1d"
      prefix="unet1d"
      batch="${BATCH_UNET:-200}"
      extra_args+=(--base_channels 64 --unet_depth 4 --kernel_size 7 --dropout 0.1)
      ;;
    *)
      echo "Unknown architecture: $arch"
      exit 2
      ;;
  esac

  local log_dir="$out_dir/logs"
  mkdir -p "$log_dir"

  local out_csv="$out_dir/${prefix}_calibrated_fusion_${FOURIER_MODE}_clean${seed}.csv"
  local log_file="$log_dir/${prefix}_calibrated_fusion_${FOURIER_MODE}_clean${seed}.log"

  echo "=================================================================="
  echo "[START] arch=$arch seed=$seed $(date -Is)"
  echo "  module -> $module"
  echo "  csv    -> $out_csv"
  echo "  log    -> $log_file"
  echo "=================================================================="

  python -u -m "$module" \
    --data_dir data/LADS \
    --out_csv "$out_csv" \
    --device "$DEVICE" \
    --fourier_mode "$FOURIER_MODE" \
    --train_sampling fixed \
    --batch_train "$batch" \
    --batch_eval "$batch" \
    --lambda_visible_residual "$LAMBDA_VISIBLE_RESIDUAL" \
    --alpha_steps "$ALPHA_STEPS" \
    --seed "$seed" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_file"

  echo "[DONE] arch=$arch seed=$seed $(date -Is)"
}

for seed in $SEEDS; do
  for arch in $ARCHS; do
    run_one "$arch" "$seed"
  done
done

echo "All requested calibrated-fusion experiments completed."
