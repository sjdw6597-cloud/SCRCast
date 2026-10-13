"""SCRCast: CSCA attention, ACR routing, and a patch forecasting head."""
import torch
from torch import nn
from layers.ACR import ACR_Module
from layers.CSCA import CSCAEncoderLayer, build_csca_filters
from layers.Embed import PatchEmbedding
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.patch_embedding = PatchEmbedding(
            configs.d_model, configs.patch_len, configs.stride,
            configs.stride, configs.dropout)
        patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 2)
        # Construction order preserves the source model's random initialization.
        self.acr = ACR_Module(
            configs.d_model, reduction_ratio=configs.acr_reduction_ratio,
            gate_bias=configs.acr_gate_bias, graph_top_k=configs.acr_graph_top_k,
            gamma_init=configs.acr_gamma_init, tau=configs.acr_tau,
            dropout=configs.acr_dropout, residual_scale=configs.acr_residual_scale)
        self.encoder = Encoder([
            CSCAEncoderLayer(configs.d_model, configs.d_ff, configs.n_heads,
                             configs.dropout, configs.activation,
                             csca_global_mix=configs.csca_global_mix)
            for _ in range(configs.e_layers)
        ], norm_layer=nn.LayerNorm(configs.d_model))
        self.csca_filters = build_csca_filters(configs)
        self.csca_filter_scale = configs.csca_filter_scale
        self.revin = RevIN(configs.enc_in, affine=True)
        self.head = FlattenHead(configs.enc_in, configs.d_model * patch_num,
                                configs.pred_len, configs.dropout)

    def forward(self, x_enc):
        # CSCA observes StandardScaler output before window-level RevIN.
        raw_x = x_enc.detach().clone()
        x = self.revin(x_enc, 'norm')
        patches, channels = self.patch_embedding(x.permute(0, 2, 1))
        batch = patches.shape[0] // channels
        count = patches.shape[1]
        filters = self.csca_filters(raw_x)
        if self.csca_filter_scale != 1.0:
            filters = 1.0 + self.csca_filter_scale * (filters - 1.0)
        patches = patches.reshape(batch, channels, count, -1).reshape(
            batch, channels * count, -1)
        encoded, _ = self.encoder(patches, csca_filters=filters, n_vars=channels)
        encoded = encoded.reshape(batch, channels, count, -1)
        routed, _, _ = self.acr(encoded.permute(0, 2, 1, 3).reshape(
            batch * count, channels, -1))
        routed = routed.reshape(batch, count, channels, -1).permute(0, 2, 3, 1)
        prediction = self.head(routed).permute(0, 2, 1)
        return self.revin(prediction, 'denorm')[:, -self.pred_len:, :]
