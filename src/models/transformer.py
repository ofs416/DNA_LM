import torch.nn as nn
import torch
import torch.nn.functional as F

from src.models.embedding import PositionalEncoder


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
        # Z shape: (batch_size, num_heads, seq_len, model_dim)

        return Z


class Transformer(nn.Module):
    def __init__(self, model_dim, num_heads, max_len):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.mha = MHA(model_dim, num_heads)
        self.embedding = PositionalEncoder(max_len, model_dim)
        self.dense = nn.Linear(model_dim * num_heads, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        mha_output_norm = self.mha(self.norm(embeddings))
        resdiual_output = (mha_output_norm + embeddings.unsqueeze(1)).transpose(1, 2)
        # resdiual_output shape: (batch_size, seq_len, num_heads, model_dim)
        dense_output = self.dense(
            resdiual_output.reshape(
                resdiual_output.size(0),
                resdiual_output.size(1),
                resdiual_output.size(2) * resdiual_output.size(3),
            )
        )

        return dense_output


if __name__ == "__main__":
    model_dim = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    mha = MHA(model_dim, num_heads)
    input = torch.rand(batch_size, seq_len, model_dim)
    output = mha(input)
    print(
        f"MHA output shape: {output.shape}"
    )  # Expected shape: (batch_size, num_heads, seq_len, model_dim)

    transformer = Transformer(model_dim, num_heads, max_len=100)

    transformer_output = transformer(input)
    print(
        f"Transformer output shape: {transformer_output.shape}"
    )  # Expected shape: (batch_size, seq_len, model_dim)
