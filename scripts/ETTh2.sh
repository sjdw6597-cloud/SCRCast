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
for horizon in ${HORIZONS:-96 192 336 720}; do
  case "$horizon" in
    96)
      "$PYTHON" -u "$ROOT/run.py" \
        --dataset ETTh2 \
        --data ETTh2 \
        --data_path ETTh2.csv \
        --root_path "$DATA_ROOT/ETT-small" \
        --target OT \
        --freq h \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --enc_in 7 \
        --d_model 512 \
        --d_ff 2048 \
        --n_heads 4 \
        --e_layers 1 \
        --dropout 0.15 \
        --activation gelu \
        --patch_len 16 \
        --stride 8 \
        --csca_filter_type 1 \
        --csca_frequency_bins 1.0 \
        --csca_coherence_smooth 3 \
        --csca_idrop 0.0 \
        --csca_filter_scale 1.0 \
        --csca_global_mix 1.0 \
        --acr_reduction_ratio 8 \
        --acr_gate_bias -1.0 \
        --acr_graph_top_k 3 \
        --acr_gamma_init -3.0 \
        --acr_tau 0.5 \
        --acr_dropout 0.0 \
        --acr_residual_scale 1.0 \
        --batch_size 32 \
        --num_workers 10 \
        --train_epochs 10 \
        --patience 3 \
        --learning_rate 0.0001 \
        --weight_decay 1e-05 \
        --lradj type1 \
        --seed 2021 \
        --use_norm 1 \
        --crh_bucket_count 8 \
        --crh_epoch_interval 2 \
        --crh_mix 1.0 \
        --crh_mix_final 1.0 \
        --crh_anneal_epochs 0 \
        --curriculum_segment_length 0 \
        --curriculum_gamma 1.5 \
        --curriculum_horizon_mix 0.7 \
        --curriculum_curve power \
        --curriculum_temperature 0.05 \
        "${args[@]}"
      ;;
    192)
      "$PYTHON" -u "$ROOT/run.py" \
        --dataset ETTh2 \
        --data ETTh2 \
        --data_path ETTh2.csv \
        --root_path "$DATA_ROOT/ETT-small" \
        --target OT \
        --freq h \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
        --enc_in 7 \
        --d_model 256 \
        --d_ff 1024 \
        --n_heads 4 \
        --e_layers 2 \
        --dropout 0.22 \
        --activation gelu \
        --patch_len 18 \
        --stride 9 \
        --csca_filter_type 1 \
        --csca_frequency_bins 1.0 \
        --csca_coherence_smooth 5 \
        --csca_idrop 0.0 \
        --csca_filter_scale 1.0 \
        --csca_global_mix 1.0 \
        --acr_reduction_ratio 8 \
        --acr_gate_bias 0.0 \
        --acr_graph_top_k 3 \
        --acr_gamma_init -1.25 \
        --acr_tau 0.5 \
        --acr_dropout 0.05 \
        --acr_residual_scale 1.0 \
        --batch_size 32 \
        --num_workers 10 \
        --train_epochs 10 \
        --patience 3 \
        --learning_rate 0.0001 \
        --weight_decay 0.0 \
        --lradj type1 \
        --seed 2021 \
        --use_norm 1 \
        --crh_bucket_count 8 \
        --crh_epoch_interval 2 \
        --crh_mix 0.0 \
        --crh_mix_final 0.5 \
        --crh_anneal_epochs 2 \
        --curriculum_segment_length 0 \
        --curriculum_gamma 0.5 \
        --curriculum_horizon_mix 0.7 \
        --curriculum_curve power \
        --curriculum_temperature 0.05 \
        "${args[@]}"
      ;;
    336)
      "$PYTHON" -u "$ROOT/run.py" \
        --dataset ETTh2 \
        --data ETTh2 \
        --data_path ETTh2.csv \
        --root_path "$DATA_ROOT/ETT-small" \
        --target OT \
        --freq h \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 336 \
        --enc_in 7 \
        --d_model 384 \
        --d_ff 1536 \
        --n_heads 4 \
        --e_layers 2 \
        --dropout 0.01 \
        --activation gelu \
        --patch_len 16 \
        --stride 8 \
        --csca_filter_type 1 \
        --csca_frequency_bins 1.0 \
        --csca_coherence_smooth 3 \
        --csca_idrop 0.0 \
        --csca_filter_scale 1.0 \
        --csca_global_mix 1.0 \
        --acr_reduction_ratio 8 \
        --acr_gate_bias 0.0 \
        --acr_graph_top_k 3 \
        --acr_gamma_init 0.0 \
        --acr_tau 0.5 \
        --acr_dropout 0.0 \
        --acr_residual_scale 1.0 \
        --batch_size 56 \
        --num_workers 10 \
        --train_epochs 15 \
        --patience 5 \
        --learning_rate 6e-05 \
        --weight_decay 0.0 \
        --lradj type3 \
        --seed 2021 \
        --use_norm 1 \
        --crh_bucket_count 8 \
        --crh_epoch_interval 2 \
        --crh_mix 0.5 \
        --crh_mix_final 0.5 \
        --crh_anneal_epochs 0 \
        --curriculum_segment_length 48 \
        --curriculum_gamma 0.25 \
        --curriculum_horizon_mix 0.3 \
        --curriculum_curve power \
        --curriculum_temperature 0.05 \
        "${args[@]}"
      ;;
    720)
      "$PYTHON" -u "$ROOT/run.py" \
        --dataset ETTh2 \
        --data ETTh2 \
        --data_path ETTh2.csv \
        --root_path "$DATA_ROOT/ETT-small" \
        --target OT \
        --freq h \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 720 \
        --enc_in 7 \
        --d_model 256 \
        --d_ff 1024 \
        --n_heads 4 \
        --e_layers 3 \
        --dropout 0.1 \
        --activation gelu \
        --patch_len 16 \
        --stride 8 \
        --csca_filter_type 1 \
        --csca_frequency_bins 1.0 \
        --csca_coherence_smooth 7 \
        --csca_idrop 0.0 \
        --csca_filter_scale 1.0 \
        --csca_global_mix 1.0 \
        --acr_reduction_ratio 4 \
        --acr_gate_bias 0.0 \
        --acr_graph_top_k 2 \
        --acr_gamma_init 0.0 \
        --acr_tau 0.8 \
        --acr_dropout 0.1 \
        --acr_residual_scale 1.0 \
        --batch_size 32 \
        --num_workers 10 \
        --train_epochs 10 \
        --patience 3 \
        --learning_rate 5e-05 \
        --weight_decay 0.0 \
        --lradj type1 \
        --seed 2021 \
        --use_norm 1 \
        --crh_bucket_count 8 \
        --crh_epoch_interval 2 \
        --crh_mix 0.5 \
        --crh_mix_final 0.5 \
        --crh_anneal_epochs 0 \
        --curriculum_segment_length 48 \
        --curriculum_gamma 0.5 \
        --curriculum_horizon_mix 0.7 \
        --curriculum_curve power \
        --curriculum_temperature 0.05 \
        "${args[@]}"
      ;;
    *) echo "Unsupported horizon: $horizon" >&2; exit 2 ;;
  esac
done
