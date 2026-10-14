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
  --dataset ETTm2
  --data ETTm2
  --data_path ETTm2.csv
  --root_path "$DATA_ROOT/ETT-small"
  --curriculum_gamma 1.0
)
for horizon in ${HORIZONS:-96 192 336 720}; do
  horizon_args=()
  case "$horizon" in
    96)
      horizon_args=(
        --patch_len 8
        --stride 4
        --csca_frequency_bins 0.5
        --csca_idrop 0.0
        --acr_gate_bias 0.0
        --acr_graph_top_k 1
        --acr_gamma_init 0.0
        --acr_tau 0.8
        --crh_mix 1.0
        --curriculum_curve soft_bucket
      )
      ;;
    192)
      horizon_args=(
        --patch_len 8
        --stride 4
        --csca_frequency_bins 0.4
        --csca_idrop 0.0
        --acr_gate_bias 0.0
        --acr_gamma_init -2.0
        --acr_dropout 0.05
      )
      ;;
    336)
      horizon_args=(
        --csca_frequency_bins 0.5
        --csca_idrop 0.0
        --csca_filter_scale 0.5
        --acr_gate_bias -3.0
        --acr_gamma_init -4.0
        --weight_decay 0.0001
        --crh_mix 0.75
        --curriculum_curve soft_bucket
        --curriculum_temperature 0.08
      )
      ;;
    720)
      horizon_args=(
        --e_layers 3
        --dropout 0.15
        --patch_len 24
        --stride 12
        --csca_filter_type 1
        --csca_idrop 0.2
        --acr_reduction_ratio 16
        --acr_gate_bias -3.0
        --acr_graph_top_k 0
        --acr_gamma_init -3.5
        --acr_tau 1.0
        --acr_dropout 0.0
        --batch_size 64
        --learning_rate 2.5e-05
        --crh_mix 0.25
        --curriculum_segment_length 48
        --curriculum_horizon_mix 0.5
      )
      ;;
    *) echo "Unsupported horizon: $horizon" >&2; exit 2 ;;
  esac
  "$PYTHON" -u "$ROOT/run.py" \
    "${common_args[@]}" --pred_len "$horizon" \
    "${horizon_args[@]}" "${args[@]}"
done
