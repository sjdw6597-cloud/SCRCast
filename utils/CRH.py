"""Continuous-rank hierarchical training objective."""
import torch


def _descending_weights(bucket_count, device, dtype):
    if bucket_count < 2:
        raise ValueError('CRH requires at least two predictability buckets')
    weights = torch.linspace(1.0, 0.0, bucket_count, device=device, dtype=dtype)
    weights[-1] = weights[-2] / 2
    return weights


def crh_loss(
    predictions,
    target,
    ranking_predictions,
    epoch,
    bucket_count=8,
    epoch_interval=2,
    segment_length=0,
    horizon_mix=0.7,
    gamma=1.0,
    curve='power',
    temperature=0.05,
):
    """Continuous rank relaxation of hierarchical predictability training.

    The historical stage schedule is retained, but every discrete bucket
    assignment is replaced by an analytic rank-weight curve. This makes the
    training-unit weighting continuous while preserving the scale of each
    hierarchy level.
    """
    if predictions.shape != target.shape:
        raise ValueError('predictions and target must have identical shapes')
    if bucket_count < 2:
        raise ValueError('bucket_count must be at least two')
    if epoch_interval <= 0:
        raise ValueError('epoch_interval must be positive')
    if segment_length < 0:
        raise ValueError('segment_length must be non-negative')
    if not 0.0 <= horizon_mix <= 1.0:
        raise ValueError('horizon_mix must be in [0, 1]')
    if gamma <= 0.0:
        raise ValueError('gamma must be positive')
    if curve not in {'power', 'rank_interp', 'soft_bucket'}:
        raise ValueError(
            "curve must be 'power', 'rank_interp', or 'soft_bucket'"
        )
    if temperature <= 0.0:
        raise ValueError('temperature must be positive')

    unit_loss = _unit_errors(predictions, target, segment_length)
    with torch.no_grad():
        ranking_error = _unit_errors(
            ranking_predictions, target, segment_length
        )
        local_rank = _percentile_rank(ranking_error)
        global_rank = _percentile_rank(
            ranking_error.reshape(-1, 1)
        ).reshape_as(ranking_error)
        mixed_rank = (
            horizon_mix * local_rank
            + (1.0 - horizon_mix) * global_rank
        )
        difficulty = mixed_rank
        rank_curve = difficulty.pow(gamma)

    base_weights = _descending_weights(
        int(bucket_count), predictions.device, predictions.dtype
    )
    stage = min(int(epoch) // int(epoch_interval), int(bucket_count) - 1)
    total = predictions.new_zeros(())
    active_weight_sum = predictions.new_zeros(())
    combined_weights = torch.zeros_like(unit_loss)

    for level in range(stage + 1):
        remaining = int(bucket_count) - level
        easy_weight = base_weights[level]
        hard_weight = base_weights[-1]
        if curve == 'rank_interp':
            anchors = base_weights[level:]
            position = rank_curve * (remaining - 1)
            lower = position.floor().long().clamp_max(remaining - 1)
            upper = (lower + 1).clamp_max(remaining - 1)
            fraction = position - lower.to(position.dtype)
            level_weights = (
                anchors[lower] * (1.0 - fraction)
                + anchors[upper] * fraction
            )
        elif curve == 'soft_bucket':
            anchors = base_weights[level:]
            level_weights = torch.full_like(rank_curve, anchors[0])
            for boundary in range(1, remaining):
                threshold = float(boundary) / float(remaining)
                transition = torch.sigmoid(
                    (rank_curve - threshold) / temperature
                )
                level_weights = level_weights + (
                    anchors[boundary] - anchors[boundary - 1]
                ) * transition
        else:
            level_weights = (
                easy_weight
                + (hard_weight - easy_weight) * rank_curve
            )
        total = total + remaining * (level_weights * unit_loss).mean()
        active_weight_sum = (
            active_weight_sum + remaining * level_weights.mean()
        )
        combined_weights = combined_weights + level_weights

    loss = total / int(bucket_count)
    diagnostics = {
        'stage': stage,
        'active_weight_sum': active_weight_sum.detach(),
        'easy_loss': unit_loss[difficulty == difficulty.min()][0].detach(),
        'hard_loss': unit_loss[difficulty == difficulty.max()][0].detach(),
        'unit_count': unit_loss.new_tensor(unit_loss.numel()).detach(),
        'weight_mean': combined_weights.mean().detach(),
        'weight_std': combined_weights.std(unbiased=False).detach(),
    }
    return loss, diagnostics


def _unit_errors(values, references, segment_length):
    squared_error = (values - references).square()
    if segment_length <= 0 or segment_length >= squared_error.size(1):
        return squared_error.flatten(1).mean(dim=1, keepdim=True)
    return torch.stack([
        segment.flatten(1).mean(dim=1)
        for segment in torch.split(squared_error, segment_length, dim=1)
    ], dim=1)


def _percentile_rank(values):
    """Rank samples independently for every future segment."""
    batch_size = values.size(0)
    if batch_size <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values, dim=0)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(
        batch_size, device=values.device, dtype=order.dtype
    ).view(batch_size, 1).expand_as(order)
    ranks.scatter_(0, order, rank_values)
    return ranks.to(values.dtype) / float(batch_size - 1)
