import torch.nn as nn
import torch
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.W_q = nn.Linear(self.model_dim, self.model_dim * num_heads, bias=False)
        self.W_k = nn.Linear(self.model_dim, self.model_dim * num_heads, bias=False)
        self.W_v = nn.Linear(self.model_dim, self.model_dim * num_heads, bias=False)

    def forward(self, input):
        # Input shape: (batch_size, seq_len, model_dim)

        Q, K, V = self.W_q(input), self.W_k(input), self.W_v(input)
        Q = Q.view(
            input.size(0), input.size(1), self.num_heads, self.model_dim
        ).transpose(1, 2)
        K = K.view(
            input.size(0), input.size(1), self.num_heads, self.model_dim
        ).transpose(1, 2)
        V = V.view(
            input.size(0), input.size(1), self.num_heads, self.model_dim
        ).transpose(1, 2)

        Attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.model_dim**0.5)
        Z = F.softmax(torch.matmul(Attention, V), dim=-1)

        return Z


if __name__ == "__main__":
    model_dim = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    mha = MHA(model_dim, num_heads)
    input = torch.rand(batch_size, seq_len, model_dim)
    output = mha(input)
    print(output.shape)  # Expected shape: (batch_size, num_heads, seq_len, model_dim)
