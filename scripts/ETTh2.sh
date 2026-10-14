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
  --dataset ETTh2
  --data ETTh2
  --data_path ETTh2.csv
  --root_path "$DATA_ROOT/ETT-small"
  --csca_filter_type 1
  --csca_idrop 0.0
)
for horizon in ${HORIZONS:-96 192 336 720}; do
  horizon_args=()
  case "$horizon" in
    96)
      horizon_args=(
        --d_model 512
        --d_ff 2048
        --e_layers 1
        --dropout 0.15
        --acr_gate_bias -1.0
        --acr_dropout 0.0
        --weight_decay 1e-05
        --crh_mix 1.0
        --curriculum_gamma 1.5
      )
      ;;
    192)
      horizon_args=(
        --dropout 0.22
        --patch_len 18
        --stride 9
        --csca_coherence_smooth 5
        --acr_gate_bias 0.0
        --acr_gamma_init -1.25
        --acr_dropout 0.05
        --crh_mix 0.0
        --crh_mix_final 0.5
        --crh_anneal_epochs 2
      )
      ;;
    336)
      horizon_args=(
        --d_model 384
        --d_ff 1536
        --dropout 0.01
        --acr_gate_bias 0.0
        --acr_gamma_init 0.0
        --acr_dropout 0.0
        --batch_size 56
        --train_epochs 15
        --patience 5
        --learning_rate 6e-05
        --lradj type3
        --curriculum_segment_length 48
        --curriculum_gamma 0.25
        --curriculum_horizon_mix 0.3
      )
      ;;
    720)
      horizon_args=(
        --e_layers 3
        --csca_coherence_smooth 7
        --acr_reduction_ratio 4
        --acr_gate_bias 0.0
        --acr_graph_top_k 2
        --acr_gamma_init 0.0
        --acr_tau 0.8
        --learning_rate 5e-05
        --curriculum_segment_length 48
      )
      ;;
    *) echo "Unsupported horizon: $horizon" >&2; exit 2 ;;
  esac
  "$PYTHON" -u "$ROOT/run.py" \
    "${common_args[@]}" --pred_len "$horizon" \
    "${horizon_args[@]}" "${args[@]}"
done
