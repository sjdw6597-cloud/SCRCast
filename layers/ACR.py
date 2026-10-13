import torch
from torch import nn

class ACR_Module(nn.Module):
    def __init__(
        self,
        d_model,
        reduction_ratio=8,
        gate_bias=-2.0,
        graph_top_k=3,
        gamma_init=0.0,
        tau=0.5,
        dropout=0.1,
        residual_scale=1.0,
    ):
        super(ACR_Module, self).__init__()
        self.d_model = d_model
        self.d_k = max(1, d_model // reduction_ratio)
        self.gate_bias = gate_bias          # static scalar (not learnable)
        self.graph_top_k = graph_top_k
        self.tau = tau
        self.residual_scale = float(residual_scale)
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)

        # CI MLP: channel-independent feature transform
        self.ci_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Dual gate: H content + G strength
        self.gate_h = nn.Linear(d_model, 1)   # gate from feature content
        self.gate_g = nn.Linear(1, 1)          # gate from graph strength

        # Residual mixing coefficient (learnable)
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))

    def _sparsify_graph(self, G):
        if self.graph_top_k is None or self.graph_top_k <= 0:
            return G
        k = min(self.graph_top_k, G.shape[-1])
        if k >= G.shape[-1]:
            return G
        values, indices = torch.topk(G, k=k, dim=-1)
        sparse_G = torch.zeros_like(G)
        sparse_G.scatter_(-1, indices, values)
        return sparse_G / sparse_G.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def fuse(self, H_base, G):
        # 1. Graph routing: aggregate neighbor context
        H_cd = torch.matmul(G, H_base)            # (B*P, V, D)

        # 2. Channel-independent MLP transform
        H_ci = self.ci_mlp(H_base)                # (B*P, V, D)

        # 3. Gate: combine feature content + graph strength
        c = G.shape[-1]
        mask = torch.eye(c, device=H_base.device).bool()
        G_cross = G.masked_fill(mask, 0.0)        # remove diagonal (self-loop)
        g_strength = G_cross.max(dim=-1, keepdim=True)[0]

        gate_logit = self.gate_h(H_base) + self.gate_g(g_strength)
        alpha = torch.sigmoid(gate_logit + self.gate_bias)

        # 4. Mix: routed features + CI features
        H_mix = alpha * H_cd + (1 - alpha) * H_ci

        # 5. Residual: learnable gamma controls bypass
        gamma = torch.sigmoid(self.gamma).view(1, 1, 1)
        if self.residual_scale == 1.0:
            # Keep the historical default computation exactly unchanged.
            H_out = gamma * H_mix + (1 - gamma) * H_base
        else:
            # Scale only the learned ACR residual while preserving the identity path.
            H_out = H_base + self.residual_scale * gamma * (H_mix - H_base)

        return H_out, alpha, gamma

    def forward(self, H):
        Q = self.W_q(H)
        K = self.W_k(H)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)
        G = torch.softmax(scores / self.tau, dim=-1)
        G = self._sparsify_graph(G)

        H_out, alpha, gamma = self.fuse(H, G)

        return H_out, G, alpha
