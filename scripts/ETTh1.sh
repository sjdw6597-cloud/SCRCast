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
  --dataset ETTh1
  --root_path "$DATA_ROOT/ETT-small"
)
for horizon in ${HORIZONS:-96 192 336 720}; do
  horizon_args=()
  case "$horizon" in
    96)
      horizon_args=(
        --d_model 512
        --d_ff 2048
        --e_layers 1
        --patch_len 12
        --stride 6
        --csca_frequency_bins 0.75
        --acr_reduction_ratio 16
        --acr_graph_top_k 0
        --acr_tau 1.0
        --acr_dropout 0.2
        --acr_residual_scale 0.1
        --learning_rate 5e-05
        --crh_bucket_count 32
        --crh_mix 0.65
        --curriculum_gamma 1.0
      )
      ;;
    192)
      horizon_args=(
        --csca_frequency_bins 0.5
        --csca_idrop 0.0
        --acr_reduction_ratio 16
        --acr_graph_top_k 0
        --acr_gamma_init -4.0
        --acr_tau 1.0
        --acr_dropout 0.05
        --weight_decay 0.0001
        --crh_mix 0.0
        --crh_mix_final 0.5
        --crh_anneal_epochs 2
        --curriculum_curve soft_bucket
      )
      ;;
    336)
      horizon_args=(
        --csca_frequency_bins 0.75
        --csca_idrop 0.0
        --acr_reduction_ratio 16
        --acr_gate_bias -3.0
        --acr_graph_top_k 1
        --acr_gamma_init -4.0
        --acr_tau 1.0
        --weight_decay 0.0001
        --crh_mix 0.0
        --crh_mix_final 0.5
        --crh_anneal_epochs 2
      )
      ;;
    720)
      horizon_args=(
        --d_model 512
        --d_ff 2048
        --e_layers 1
        --dropout 0.0
        --csca_idrop 0.0
        --acr_gate_bias 0.0
        --acr_gamma_init 0.0
        --curriculum_segment_length 48
      )
      ;;
    *) echo "Unsupported horizon: $horizon" >&2; exit 2 ;;
  esac
  "$PYTHON" -u "$ROOT/run.py" \
    "${common_args[@]}" --pred_len "$horizon" \
    "${horizon_args[@]}" "${args[@]}"
done
