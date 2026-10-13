from torch import nn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, **kwargs):
        attns = []
        for layer in self.attn_layers:
            x, attn = layer(x, **kwargs)
            attns.append(attn)
        return self.norm(x), attns
