"""Validation-selected training and streaming test metrics for SCRCast."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data_provider.data_factory import data_provider
from models.SCRCast import Model
from utils.CRH import crh_loss
from utils.tools import EarlyStopping, adjust_learning_rate


def training_loss(prediction, target, args, epoch):
    base = nn.functional.mse_loss(prediction, target)
    crh, stats = crh_loss(
        prediction, target, prediction.detach(), epoch,
        bucket_count=args.crh_bucket_count, epoch_interval=args.crh_epoch_interval,
        segment_length=args.curriculum_segment_length,
        horizon_mix=args.curriculum_horizon_mix, gamma=args.curriculum_gamma,
        curve=args.curriculum_curve, temperature=args.curriculum_temperature)
    # Divide out the active hierarchy's mean weight before mixing with MSE.
    scale = float(args.crh_bucket_count) / stats['active_weight_sum'].to(crh).clamp_min(
        torch.finfo(crh.dtype).eps)
    crh = crh * scale
    final_mix = getattr(args, 'crh_mix_final', args.crh_mix)
    progress = min(max(float(epoch) / args.crh_anneal_epochs, 0.0), 1.0) if args.crh_anneal_epochs else 0.0
    mix = args.crh_mix + progress * (final_mix - args.crh_mix)
    return (1.0 - mix) * base + mix * crh


def forward_batch(model, batch, args, device):
    x, y = [item.float().to(device) for item in batch]
    prediction = model(x)[:, -args.pred_len:]
    return prediction, y[:, -args.pred_len:]


def validation_loss(model, loader, args, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            prediction, target = forward_batch(model, batch, args, device)
            losses.append(nn.functional.mse_loss(prediction.detach(), target.detach()).item())
    model.train()
    return float(np.average(losses))


def test_metrics(model, loader, args, device):
    model.eval()
    count, squared_sum, absolute_sum = 0, 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            prediction, target = forward_batch(model, batch, args, device)
            error = prediction.detach().cpu().numpy() - target.detach().cpu().numpy()
            squared_sum += float(np.square(error).sum(dtype=np.float64))
            absolute_sum += float(np.abs(error).sum(dtype=np.float64))
            count += error.size
    if count == 0:
        raise ValueError('Empty test split')
    return {'mse': squared_sum / count, 'mae': absolute_sum / count}


def train_and_test(args, output_dir, device):
    model = Model(args).float().to(device)
    _, train_loader = data_provider(args, 'train')
    _, val_loader = data_provider(args, 'val')
    _, test_loader = data_provider(args, 'test')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    early_stop = EarlyStopping(patience=args.patience, verbose=True)
    best_epoch = None
    # The best checkpoint exists only during this run unless explicitly retained.
    with tempfile.TemporaryDirectory(prefix='scrcast_ckpt_') as checkpoint_dir:
        for epoch in range(args.train_epochs):
            started = time.monotonic()
            model.train()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                prediction, target = forward_batch(model, batch, args, device)
                loss = training_loss(prediction, target, args, epoch)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            val_loss = validation_loss(model, val_loader, args, device)
            # Preserve the source loop's test traversal and RNG order.
            # The test loss is reported; checkpoint selection uses validation.
            test_loss = validation_loss(model, test_loader, args, device)
            if not np.isfinite(val_loss) or not np.isfinite(test_loss):
                raise RuntimeError('Nonfinite evaluation loss')
            if early_stop.best_score is None or -val_loss >= early_stop.best_score:
                best_epoch = epoch + 1
            print(f'Epoch {epoch + 1} | train={np.average(losses):.7f} val={val_loss:.7f} '
                  f'test={test_loss:.7f} seconds={time.monotonic() - started:.1f}', flush=True)
            early_stop(val_loss, model, checkpoint_dir)
            if early_stop.early_stop:
                break
            adjust_learning_rate(optimizer, epoch + 1, args)
        model.load_state_dict(torch.load(Path(checkpoint_dir) / 'checkpoint.pth',
                                         map_location=device, weights_only=True))
        if args.save_checkpoint:
            torch.save(model.state_dict(), output_dir / 'checkpoint.pth')
        metrics = test_metrics(model, test_loader, args, device)
    result = {'dataset': args.dataset, 'pred_len': args.pred_len, 'seed': args.seed,
              'best_validation_epoch': best_epoch, **metrics}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"mse:{metrics['mse']}, mae:{metrics['mae']}", flush=True)
    return result
