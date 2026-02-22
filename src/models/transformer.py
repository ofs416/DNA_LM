import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self, input_dim, model_dim, num_heads, num_layers, output_dim, embedder
    ):
        super().__init__()
        self.embedder = embedder
        self.W_q = nn.Linear(model_dim, model_dim, bias=False)
        self.W_k = nn.Linear(model_dim, model_dim, bias=False)
        self.W_v = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, input):
        return
