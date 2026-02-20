import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.arange(0, d_model, 2).float() / d_model
        self.pe[:, 0::2] = torch.sin(position * div)
        self.pe[:, 1::2] = torch.cos(position * div)

    def forward(self, tokens):
        return self.pe[: tokens.size(1), :] + tokens
