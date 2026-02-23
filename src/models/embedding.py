import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.tensor(10_000) ** (torch.arange(0, d_model, 2).float() / d_model)
        pos_div = position / div
        pe[:, 0::2] = torch.sin(pos_div)
        pe[:, 1::2] = torch.cos(pos_div)
        self.register_buffer("pe", pe)

    def forward(self, tokens):
        return self.pe[: tokens.size(-2), :] + tokens


if __name__ == "__main__":
    enc = PositionalEncoder(32, 8)
    print(enc.pe.shape)
    tokens = torch.zeros(32, 8)
    print(enc(tokens).shape)
    print(enc.pe.shape)

    enc = PositionalEncoder(32, 8)
    print(enc.pe.shape)
    tokens = torch.zeros(10, 32, 8)
    print(enc(tokens).shape)
    print(enc.pe.shape)
