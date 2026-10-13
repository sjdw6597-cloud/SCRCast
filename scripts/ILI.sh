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
for horizon in ${HORIZONS:-24 36 48 60}; do
  case "$horizon" in
    24)
      "$PYTHON" -u "$ROOT/run.py" \
        --dataset ILI \
        --data custom \
        --data_path national_illness.csv \
        --root_path "$DATA_ROOT" \
        --target OT \
        --freq w \
        --seq_len 36 \
        --label_len 18 \
        --pred_len 24 \
        --enc_in 7 \
        --d_model 256 \
        --d_ff 1024 \
        --n_heads 4 \
        --e_layers 2 \
        --dropout 0.1 \
        --activation gelu \
        --patch_len 16 \
        --stride 8 \
        --csca_filter_type 2 \
        --csca_frequency_bins 0.85 \
        --csca_coherence_smooth 3 \
        --csca_idrop 0.1 \
        --csca_filter_scale 1.0 \
        --csca_global_mix 1.0 \
        --acr_reduction_ratio 8 \
        --acr_gate_bias -2.0 \
        --acr_graph_top_k 3 \
        --acr_gamma_init -3.0 \
        --acr_tau 0.5 \
        --acr_dropout 0.05 \
        --acr_residual_scale 1.0 \
        --batch_size 16 \
        --num_workers 10 \
        --train_epochs 20 \
        --patience 6 \
        --learning_rate 0.00012 \
        --weight_decay 0.0 \
        --lradj type3 \
        --seed 2021 \
        --use_norm 1 \
        --crh_bucket_count 8 \
        --crh_epoch_interval 2 \
        --crh_mix 0.005 \
        --crh_mix_final 0.005 \
        --crh_anneal_epochs 0 \
        --curriculum_segment_length 6 \
        --curriculum_gamma 0.5 \
        --curriculum_horizon_mix 0.3 \
        --curriculum_curve soft_bucket \
        --curriculum_temperature 0.05 \
        "${args[@]}"
      ;;
    36)
      "$PYTHON" -u "$ROOT/run.py" \
        --dataset ILI \
        --data custom \
        --data_path national_illness.csv \
        --root_path "$DATA_ROOT" \
        --target OT \
        --freq w \
        --seq_len 36 \
        --label_len 18 \
        --pred_len 36 \
        --enc_in 7 \
        --d_model 512 \
        --d_ff 2048 \
        --n_heads 4 \
        --e_layers 2 \
        --dropout 0.1 \
        --activation gelu \
        --patch_len 9 \
        --stride 3 \
        --csca_filter_type 2 \
        --csca_frequency_bins 1.0 \
        --csca_coherence_smooth 3 \
        --csca_idrop 0.2 \
        --csca_filter_scale 1.0 \
        --csca_global_mix 1.0 \
        --acr_reduction_ratio 8 \
        --acr_gate_bias -2.0 \
        --acr_graph_top_k 3 \
        --acr_gamma_init -3.0 \
        --acr_tau 0.5 \
        --acr_dropout 0.05 \
        --acr_residual_scale 1.0 \
        --batch_size 16 \
        --num_workers 10 \
        --train_epochs 30 \
        --patience 8 \
        --learning_rate 5e-05 \
        --weight_decay 0.0 \
        --lradj type3 \
        --seed 2021 \
        --use_norm 1 \
        --crh_bucket_count 8 \
        --crh_epoch_interval 2 \
        --crh_mix 1.0 \
        --crh_mix_final 1.0 \
        --crh_anneal_epochs 0 \
        --curriculum_segment_length 0 \
        --curriculum_gamma 0.5 \
        --curriculum_horizon_mix 1.0 \
        --curriculum_curve power \
        --curriculum_temperature 0.05 \
        "${args[@]}"
      ;;
    48)
      "$PYTHON" -u "$ROOT/run.py" \
        --dataset ILI \
        --data custom \
        --data_path national_illness.csv \
        --root_path "$DATA_ROOT" \
        --target OT \
        --freq w \
        --seq_len 36 \
        --label_len 18 \
        --pred_len 48 \
        --enc_in 7 \
        --d_model 256 \
        --d_ff 1024 \
        --n_heads 4 \
        --e_layers 2 \
        --dropout 0.1 \
        --activation gelu \
        --patch_len 12 \
        --stride 6 \
        --csca_filter_type 2 \
        --csca_frequency_bins 1.0 \
        --csca_coherence_smooth 3 \
        --csca_idrop 0.1 \
        --csca_filter_scale 1.0 \
        --csca_global_mix 1.0 \
        --acr_reduction_ratio 8 \
        --acr_gate_bias -2.0 \
        --acr_graph_top_k 3 \
        --acr_gamma_init -3.0 \
        --acr_tau 0.5 \
        --acr_dropout 0.05 \
        --acr_residual_scale 1.0 \
        --batch_size 32 \
        --num_workers 10 \
        --train_epochs 25 \
        --patience 8 \
        --learning_rate 0.000125 \
        --weight_decay 0.0 \
        --lradj type3 \
        --seed 2021 \
        --use_norm 1 \
        --crh_bucket_count 8 \
        --crh_epoch_interval 2 \
        --crh_mix 0.5 \
        --crh_mix_final 0.5 \
        --crh_anneal_epochs 0 \
        --curriculum_segment_length 0 \
        --curriculum_gamma 0.5 \
        --curriculum_horizon_mix 0.7 \
        --curriculum_curve power \
        --curriculum_temperature 0.05 \
        "${args[@]}"
      ;;
    60)
      "$PYTHON" -u "$ROOT/run.py" \
        --dataset ILI \
        --data custom \
        --data_path national_illness.csv \
        --root_path "$DATA_ROOT" \
        --target OT \
        --freq w \
        --seq_len 36 \
        --label_len 18 \
        --pred_len 60 \
        --enc_in 7 \
        --d_model 256 \
        --d_ff 1024 \
        --n_heads 4 \
        --e_layers 2 \
        --dropout 0.0 \
        --activation gelu \
        --patch_len 12 \
        --stride 6 \
        --csca_filter_type 2 \
        --csca_frequency_bins 1.0 \
        --csca_coherence_smooth 3 \
        --csca_idrop 0.3 \
        --csca_filter_scale 0.025 \
        --csca_global_mix 1.0 \
        --acr_reduction_ratio 8 \
        --acr_gate_bias -4.0 \
        --acr_graph_top_k 3 \
        --acr_gamma_init 0.0 \
        --acr_tau 0.5 \
        --acr_dropout 0.05 \
        --acr_residual_scale 0.5 \
        --batch_size 16 \
        --num_workers 10 \
        --train_epochs 22 \
        --patience 40 \
        --learning_rate 7.5e-05 \
        --weight_decay 0.0 \
        --lradj cosine \
        --seed 2021 \
        --use_norm 1 \
        --crh_bucket_count 8 \
        --crh_epoch_interval 2 \
        --crh_mix 1.0 \
        --crh_mix_final 1.0 \
        --crh_anneal_epochs 0 \
        --curriculum_segment_length 0 \
        --curriculum_gamma 2.0 \
        --curriculum_horizon_mix 0.7 \
        --curriculum_curve power \
        --curriculum_temperature 0.05 \
        "${args[@]}"
      ;;
    *) echo "Unsupported horizon: $horizon" >&2; exit 2 ;;
  esac
done
