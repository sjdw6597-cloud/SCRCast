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
  --dataset Weather
  --data custom
  --data_path weather.csv
  --root_path "$DATA_ROOT/weather"
  --enc_in 21
)
for horizon in ${HORIZONS:-96 192 336 720}; do
  horizon_args=()
  case "$horizon" in
    96)
      horizon_args=(
        --patch_len 8
        --stride 4
        --csca_idrop 0.0
        --csca_filter_scale 0.5
        --acr_gate_bias 0.0
        --acr_gamma_init 0.0
        --train_epochs 5
        --crh_mix 1.0
        --curriculum_gamma 0.75
        --curriculum_curve rank_interp
      )
      ;;
    192)
      horizon_args=(
        --patch_len 8
        --stride 4
        --csca_frequency_bins 0.5
        --csca_idrop 0.0
        --acr_gate_bias 0.0
        --acr_gamma_init 0.0
        --batch_size 64
        --train_epochs 15
        --patience 5
        --learning_rate 5e-05
        --lradj type3
        --crh_mix 1.0
        --curriculum_segment_length 24
        --curriculum_gamma 1.0
        --curriculum_curve soft_bucket
        --curriculum_temperature 0.005
      )
      ;;
    336)
      horizon_args=(
        --d_model 512
        --d_ff 2048
        --csca_frequency_bins 0.5
        --acr_dropout 0.0
        --train_epochs 3
        --weight_decay 1e-05
        --crh_mix 0.25
        --curriculum_segment_length 48
      )
      ;;
    720)
      horizon_args=(
        --d_model 384
        --d_ff 1536
        --csca_frequency_bins 0.5
        --csca_idrop 0.0
        --acr_gamma_init -2.0
        --train_epochs 3
        --crh_mix 1.0
        --curriculum_segment_length 24
        --curriculum_gamma 4.0
        --curriculum_horizon_mix 0.3
        --curriculum_curve rank_interp
      )
      ;;
    *) echo "Unsupported horizon: $horizon" >&2; exit 2 ;;
  esac
  "$PYTHON" -u "$ROOT/run.py" \
    "${common_args[@]}" --pred_len "$horizon" \
    "${horizon_args[@]}" "${args[@]}"
done
