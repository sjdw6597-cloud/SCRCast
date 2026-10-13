"""
CSCA-style frequency filter attention for SCRCast.

SCRCast uses variables as tokens. This module computes variable-pair filters from the raw lookback window and shares each variable-pair filter across all patch pairs of those two variables.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CSCAIdentityDropout(nn.Module):
    """Drop multiplicative filter entries back to identity instead of zero."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        if self.p >= 1:
            return torch.ones_like(x)
        return 1.0 + F.dropout(x - 1.0, p=self.p, training=True)

class CSCAFullFrequencyLeadLagFilters(nn.Module):
    """CSCA filter_type=2: preserve frequency-specific lead-lag features."""

    def __init__(self, seq_len, d_model, frequency_bins=1.0, idrop=0.0, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.bands_to_keep = max(1, int((seq_len // 2 + 1) * frequency_bins) - 1)
        self.idrop = CSCAIdentityDropout(idrop)
        input_dim = self.bands_to_keep * 2
        self.frequency_filter_network = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(self, x_enc):
        x = x_enc.permute(0, 2, 1)
        window = torch.hann_window(self.seq_len, device=x.device, dtype=x.dtype)
        x_fft = torch.fft.rfft(x * window.view(1, 1, -1), norm='ortho', dim=-1)
        x_fft = x_fft[:, :, 1:self.bands_to_keep + 1]
        cross = x_fft.unsqueeze(2) * torch.conj(x_fft.unsqueeze(1))
        feat = torch.cat([torch.angle(cross), torch.log1p(torch.abs(cross))], dim=-1)
        filters = self.frequency_filter_network(feat)
        n_vars = x.shape[1]
        diag = torch.eye(n_vars, device=x.device, dtype=torch.bool).view(1, n_vars, n_vars, 1)
        filters = filters.masked_fill(diag.expand_as(filters), 1.0)
        return self.idrop(filters)

class CSCAFeatureFilters(nn.Module):
    """CSCA filter_type=1: spectral band features plus instantaneous coupling."""

    def __init__(self, seq_len, d_model, idrop=0.0, dropout=0.0, coherence_smooth=3):
        super().__init__()
        self.idrop = CSCAIdentityDropout(idrop)
        self.coherence_smooth = max(1, int(coherence_smooth))
        self.filter_network = nn.Sequential(nn.LayerNorm(15), nn.Linear(15, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(self, x_enc):
        x = x_enc.permute(0, 2, 1)
        lead_lag = self.compute_spectral_features(x)
        instant = self.compute_instantaneous_coupling(x)
        feat = torch.cat([lead_lag, instant], dim=-1)
        return self.idrop(self.filter_network(feat))

    def compute_spectral_features(self, x):
        B, N, L = x.shape
        window = torch.hann_window(L, device=x.device, dtype=x.dtype)
        x_fft = torch.fft.rfft(x * window.view(1, 1, -1), dim=-1)
        freq_bins = x_fft.shape[-1]
        ultra_low_end = max(1, int(freq_bins * 0.15))
        low_end = max(ultra_low_end, int(freq_bins * 0.2))
        mid_end = max(low_end, int(freq_bins * 0.6))
        high_end = max(mid_end, int(freq_bins * 0.8))
        bands = [x_fft[:, :, 0:ultra_low_end], x_fft[:, :, ultra_low_end:low_end], x_fft[:, :, low_end:mid_end], x_fft[:, :, mid_end:high_end]]
        return torch.cat([self._compute_band_features(b) for b in bands], dim=-1)

    @staticmethod
    def _smooth_frequency(x, kernel_size):
        """Replicate-pad and average the last (frequency) dimension."""
        freq_bins = x.shape[-1]
        kernel_size = min(int(kernel_size), freq_bins)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size <= 1:
            return x
        pad = kernel_size // 2

        def pool(real_x):
            flat = real_x.reshape(-1, 1, freq_bins)
            flat = F.pad(flat, (pad, pad), mode='replicate')
            return F.avg_pool1d(flat, kernel_size=kernel_size, stride=1).reshape_as(real_x)
        if torch.is_complex(x):
            return torch.complex(pool(x.real), pool(x.imag))
        return pool(x)

    def _compute_band_features(self, x_band):
        B, N, freq_bins = x_band.shape
        if freq_bins == 0:
            return torch.zeros(B, N, N, 3, device=x_band.device, dtype=x_band.real.dtype)
        x_i = x_band.unsqueeze(2)
        x_j = x_band.unsqueeze(1)
        cross = x_i * torch.conj(x_j)
        auto_i = torch.abs(x_i) ** 2
        auto_j = torch.abs(x_j) ** 2
        phase = torch.angle(cross)
        mag = torch.abs(cross)
        smooth_cross = self._smooth_frequency(cross, self.coherence_smooth)
        smooth_auto_i = self._smooth_frequency(auto_i, self.coherence_smooth)
        smooth_auto_j = self._smooth_frequency(auto_j, self.coherence_smooth)
        coherence = torch.abs(smooth_cross) ** 2 / (smooth_auto_i * smooth_auto_j + 1e-08)
        coherence = torch.clamp(coherence, 0, 1)
        weights = coherence / (torch.sum(coherence, dim=-1, keepdim=True) + 1e-08)
        avg_phase = torch.sum(phase * weights, dim=-1)
        avg_mag = torch.sum(mag * weights, dim=-1)
        avg_coherence = torch.mean(coherence, dim=-1)
        return torch.stack([avg_phase, torch.log1p(avg_mag), avg_coherence], dim=-1)

    def compute_instantaneous_coupling(self, x):
        B, N, L = x.shape
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-08)
        instant_corr = torch.bmm(x_norm, x_norm.transpose(-2, -1)) / L
        window_size = max(2, min(10, L // 4))
        rolling = x.unfold(-1, window_size, 1).std(dim=-1)
        vol_norm = (rolling - rolling.mean(dim=-1, keepdim=True)) / (rolling.std(dim=-1, keepdim=True) + 1e-08)
        vol_corr = torch.bmm(vol_norm, vol_norm.transpose(-2, -1)) / rolling.shape[-1]
        ranks = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()
        ranks = ranks - ranks.mean(dim=-1, keepdim=True)
        rank_cov = torch.bmm(ranks, ranks.transpose(-2, -1)) / max(1, L - 1)
        rank_std = torch.sqrt(torch.diagonal(rank_cov, dim1=-2, dim2=-1) + 1e-08).unsqueeze(-1)
        rank_corr = rank_cov / (rank_std * rank_std.transpose(-2, -1) + 1e-08)
        rank_corr = torch.nan_to_num(rank_corr, nan=0.0)
        return torch.stack([instant_corr, vol_corr, rank_corr], dim=-1)

def build_csca_filters(configs):
    filter_type = getattr(configs, 'csca_filter_type', 2)
    if filter_type == 1:
        return CSCAFeatureFilters(seq_len=configs.seq_len, d_model=configs.d_model, idrop=getattr(configs, 'csca_idrop', 0.0), dropout=configs.dropout, coherence_smooth=getattr(configs, 'csca_coherence_smooth', 3))
    if filter_type == 2:
        return CSCAFullFrequencyLeadLagFilters(seq_len=configs.seq_len, d_model=configs.d_model, frequency_bins=getattr(configs, 'csca_frequency_bins', 1.0), idrop=getattr(configs, 'csca_idrop', 0.0), dropout=configs.dropout)
    raise ValueError(f'csca_filter_type must be 1 or 2, got {filter_type!r}')

class CSCAFilterAttention(nn.Module):
    """
    Multi-head CSCA attention over joint variable-patch tokens.

    The token sequence is packed as ``[B, C*P, D]`` and is viewed internally
    as ``[B, H, C, P, D_head]``.  The spectral prior has shape
    ``[B, C, C, H, D_head]``: it is a variable-pair relation shared across
    all query/key patch pairs.  This factorized layout is intentional; it is
    not a patch-conditioned filter and must remain unchanged for checkpoint
    and result compatibility.
    """

    def __init__(self, d_model, n_heads=1, dropout=0.1, global_mix=1.0):
        super().__init__()
        assert d_model % n_heads == 0, f'd_model ({d_model}) must be divisible by n_heads ({n_heads})'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.global_mix = float(global_mix)

    def forward(self, q, k, v, filters=None, n_vars=1):
        B, N, D = q.shape
        C = n_vars
        P = N // C
        q_proj = self.q(q).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k_proj = self.k(k).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v_proj = self.v(v).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        if filters is not None and filters.dim() == 4 and (C > 1) and (N == C * P):
            q_var = q_proj.view(B, self.n_heads, C, P, self.d_head)
            k_var = k_proj.view(B, self.n_heads, C, P, self.d_head)
            v_var = v_proj.view(B, self.n_heads, C, P, self.d_head)
            filt = filters.view(B, C, C, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)
            if self.global_mix != 1.0:
                local_scores = torch.einsum('bhcpd,bhcqd->bhcpq', q_var, k_var) / math.sqrt(self.d_head)
                local_weights = F.softmax(local_scores, dim=-1)
                local_weights = F.dropout(local_weights, p=self.dropout, training=self.training)
                local_out = torch.einsum('bhcpq,bhcqd->bhcpd', local_weights, v_var).reshape(B, self.n_heads, N, self.d_head)
            scores = torch.einsum('bhcpd,bhrqd,bhcrd->bhcprq', q_var, k_var, filt)
            scores = scores / math.sqrt(self.d_head)
            scores = scores.reshape(B, self.n_heads, C, P, C * P)
            attn_w = F.softmax(scores, dim=-1)
            attn_w = F.dropout(attn_w, p=self.dropout, training=self.training)
            attn_var = attn_w.view(B, self.n_heads, C, P, C, P)
            value_mix = torch.einsum('bhcprq,bhrqd->bhcprd', attn_var, v_var)
            out = torch.einsum('bhcprd,bhcrd->bhcpd', value_mix, filt)
            out = out.reshape(B, self.n_heads, N, self.d_head)
            if self.global_mix != 1.0:
                out = local_out + self.global_mix * (out - local_out)
        else:
            scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_head)
            attn_w = F.softmax(scores, dim=-1)
            attn_w = F.dropout(attn_w, p=self.dropout, training=self.training)
            out = torch.matmul(attn_w, v_proj)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return (self.out(out), attn_w.mean(dim=1))
CSCAJointVariablePatchAttention = CSCAFilterAttention

class CSCAEncoderLayer(nn.Module):
    """
    Encoder layer using CSCA frequency filter attention.
    """

    def __init__(self, d_model, d_ff, n_heads=1, dropout=0.1, activation='relu', csca_global_mix=1.0, **kwargs):
        super().__init__()
        self.attention = CSCAJointVariablePatchAttention(d_model, n_heads=n_heads, dropout=dropout, global_mix=csca_global_mix)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None, csca_filters=None, n_vars=1, **kwargs):
        new_x, _ = self.attention(x, x, x, csca_filters, n_vars=n_vars)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return (self.norm2(x + y), None)
