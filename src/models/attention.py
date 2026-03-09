import torch.nn as nn
import torch
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert (
            self.head_dim * num_heads == model_dim
        ), "model_dim must be divisible by num_heads"

        self.W_q = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.W_k = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.W_v = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.W_o = nn.Linear(self.model_dim, self.model_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        # Input shape: (batch_size, seq_len, model_dim)
        batch_size = query.size(0)

        Q, K, V = self.W_q(query), self.W_k(key), self.W_v(value)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        Z = torch.matmul(attention_weights, V)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        Z = self.W_o(Z)

        return Z, attention_weights


if __name__ == "__main__":
    model_dim = 8
    num_heads = 4
    batch_size = 2
    seq_len = 5

    mha = MHA(model_dim, num_heads)
    input = torch.rand(batch_size, seq_len, model_dim)

    output, attention_weights = mha(input, input, input)
    print(
        f"MHA output shape: {output.shape}"
    )  # Expected shape: (batch_size, num_heads, seq_len, model_dim)
    print(f"Attention weights: {attention_weights.shape}")

    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])

    m_output, m_attention_weights = mha(input, input, input, mask=mask)

    print(f"MHA output with mask shape: {m_output.shape}")
    print(f"Attention weights with mask: {m_attention_weights.shape}")

    print(attention_weights[0])
    print(m_attention_weights[0])
