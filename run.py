"""Train and evaluate SCRCast using command-line arguments."""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from train import train_and_test

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='ETTh1', help='Dataset label used in output names')
    parser.add_argument('--data', type=str, default='ETTh1', choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom'])
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--root_path', type=str, default=str(ROOT / 'dataset/ETT-small'))
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'relu'])
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--csca_filter_type', type=int, default=2, choices=[1, 2])
    parser.add_argument('--csca_frequency_bins', type=float, default=1.0)
    parser.add_argument('--csca_coherence_smooth', type=int, default=3)
    parser.add_argument('--csca_idrop', type=float, default=0.1)
    parser.add_argument('--csca_filter_scale', type=float, default=1.0)
    parser.add_argument('--csca_global_mix', type=float, default=1.0)
    parser.add_argument('--acr_reduction_ratio', type=int, default=16)
    parser.add_argument('--acr_gate_bias', type=float, default=-2.0)
    parser.add_argument('--acr_graph_top_k', type=int, default=0)
    parser.add_argument('--acr_gamma_init', type=float, default=-3.0)
    parser.add_argument('--acr_tau', type=float, default=1.0)
    parser.add_argument('--acr_dropout', type=float, default=0.1)
    parser.add_argument('--acr_residual_scale', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lradj', type=str, default='type1', choices=['type1', 'type2', 'type3', 'cosine'])
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--crh_bucket_count', type=int, default=32)
    parser.add_argument('--crh_epoch_interval', type=int, default=2)
    parser.add_argument('--crh_mix', type=float, default=0.5)
    parser.add_argument('--crh_anneal_epochs', type=int, default=0)
    parser.add_argument('--curriculum_segment_length', type=int, default=0)
    parser.add_argument('--curriculum_gamma', type=float, default=1.0)
    parser.add_argument('--curriculum_horizon_mix', type=float, default=0.7)
    parser.add_argument('--curriculum_curve', type=str, default='power', choices=['power', 'rank_interp', 'soft_bucket'])
    parser.add_argument('--curriculum_temperature', type=float, default=0.05)
    parser.add_argument('--crh_mix_final', type=float, default=None, help='Final CRH mix; defaults to --crh_mix')
    parser.add_argument('--train_sample_stride', type=int, default=1)
    parser.add_argument('--val_sample_stride', type=int, default=1)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--device', default='cuda:0', help='cpu or cuda:N')
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--dry_run', action='store_true', help='Print arguments without reading data or training')
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.crh_mix_final is None:
        args.crh_mix_final = args.crh_mix
    if not 0.0 <= args.crh_mix <= 1.0 or not 0.0 <= args.crh_mix_final <= 1.0:
        raise ValueError('CRH mix values must be in [0, 1]')
    if args.crh_anneal_epochs < 0:
        raise ValueError('crh_anneal_epochs must be non-negative')
    if args.crh_anneal_epochs == 0 and args.crh_mix_final != args.crh_mix:
        raise ValueError('crh_anneal_epochs must be positive when CRH mix values differ')
    return args


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, sort_keys=True, default=str))
        return
    data_file = Path(args.root_path) / args.data_path
    if not data_file.is_file():
        raise FileNotFoundError(f'Missing dataset: {data_file}; see README.md')
    device = torch.device(args.device)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu for a CPU run')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir or ROOT / 'results' / f'{args.dataset}_{args.pred_len}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str) + '\n')
    train_and_test(args, output_dir, device)


if __name__ == '__main__':
    main()
