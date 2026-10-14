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
  --dataset ILI
  --data custom
  --data_path national_illness.csv
  --root_path "$DATA_ROOT"
  --freq w
  --seq_len 36
  --label_len 18
  --acr_dropout 0.05
)
for horizon in ${HORIZONS:-24 36 48 60}; do
  horizon_args=()
  case "$horizon" in
    24)
      horizon_args=(
        --csca_frequency_bins 0.85
        --batch_size 16
        --train_epochs 20
        --patience 6
        --learning_rate 0.00012
        --lradj type3
        --crh_mix 0.005
        --curriculum_segment_length 6
        --curriculum_horizon_mix 0.3
        --curriculum_curve soft_bucket
      )
      ;;
    36)
      horizon_args=(
        --d_model 512
        --d_ff 2048
        --patch_len 9
        --stride 3
        --csca_idrop 0.2
        --batch_size 16
        --train_epochs 30
        --patience 8
        --learning_rate 5e-05
        --lradj type3
        --crh_mix 1.0
        --curriculum_horizon_mix 1.0
      )
      ;;
    48)
      horizon_args=(
        --patch_len 12
        --stride 6
        --train_epochs 25
        --patience 8
        --learning_rate 0.000125
        --lradj type3
      )
      ;;
    60)
      horizon_args=(
        --dropout 0.0
        --patch_len 12
        --stride 6
        --csca_idrop 0.3
        --csca_filter_scale 0.025
        --acr_gate_bias -4.0
        --acr_gamma_init 0.0
        --acr_residual_scale 0.5
        --batch_size 16
        --train_epochs 22
        --patience 40
        --learning_rate 7.5e-05
        --lradj cosine
        --crh_mix 1.0
        --curriculum_gamma 2.0
      )
      ;;
    *) echo "Unsupported horizon: $horizon" >&2; exit 2 ;;
  esac
  "$PYTHON" -u "$ROOT/run.py" \
    "${common_args[@]}" --pred_len "$horizon" \
    "${horizon_args[@]}" "${args[@]}"
done
