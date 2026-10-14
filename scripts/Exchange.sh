#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/dataset}"
DEVICE="${DEVICE:-cuda:0}"
args=(--device "$DEVICE")
if [[ "${DRY_RUN:-0}" == "1" ]]; then args+=(--dry_run); fi
if [[ "${SAVE_CHECKPOINT:-0}" == "1" ]]; then args+=(--save_checkpoint); fi
# Shared dataset settings; other parameters use run.py defaults.
common_args=(
  --dataset Exchange
  --data custom
  --data_path exchange_rate.csv
  --root_path "$DATA_ROOT"
  --freq d
  --enc_in 8
  --acr_dropout 0.05
)
for horizon in ${HORIZONS:-96 192 336 720}; do
  horizon_args=()
  case "$horizon" in
    96)
      horizon_args=(
        --csca_filter_scale 1.5
      )
      ;;
    192)
      horizon_args=(
        --crh_mix 0.6
      )
      ;;
    336)
      horizon_args=(
        --d_model 512
        --d_ff 2048
        --dropout 0.015
        --patch_len 12
        --stride 6
        --csca_filter_scale 1.5
        --learning_rate 0.00021
        --crh_mix 0.75
      )
      ;;
    720)
      horizon_args=(
        --csca_filter_scale 0.24
        --acr_graph_top_k 5
        --acr_gamma_init -5.0
        --train_epochs 3
        --patience 1
        --learning_rate 5e-05
        --crh_mix 0.05
      )
      ;;
    *) echo "Unsupported horizon: $horizon" >&2; exit 2 ;;
  esac
  "$PYTHON" -u "$ROOT/run.py" \
    "${common_args[@]}" --pred_len "$horizon" \
    "${horizon_args[@]}" "${args[@]}"
done
