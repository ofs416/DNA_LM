import torch.nn as nn
import torch

from src.models.embedding import AbsPosEncoder
from src.models.attention import MHA


class Transformer(nn.Module):
    def __init__(self, num_unique_tokens, model_dim, num_heads, max_len):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.embedder = nn.Embedding(num_unique_tokens, model_dim)
        self.pos_encoder = AbsPosEncoder(max_len, model_dim)
        self.mha = MHA(model_dim, num_heads)
        self.dense = nn.Linear(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, tokens, mask=None):
        embeddings = self.embedder(tokens)
        pos_embeddings = self.pos_encoder(embeddings)
        embeddings_norm = self.norm(pos_embeddings)
        mha_output_norm, attention_weights = self.mha(
            embeddings_norm, embeddings_norm, embeddings_norm, mask=mask
        )
        resdiual_output = (mha_output_norm + embeddings.unsqueeze(1)).transpose(1, 2)
        # resdiual_output shape: (batch_size, seq_len, num_heads, model_dim)
        dense_output = self.dense(resdiual_output)

        return dense_output


if __name__ == "__main__":
    model_dim = 512
    unique_tokens = 9  # A, C, G, T, N, CLS, SEP, PAD, MASK
    num_heads = 8
    batch_size = 2
    seq_len = 10

    embedder = nn.Embedding(unique_tokens, model_dim)

    input1 = torch.randint(0, unique_tokens, (batch_size, seq_len))  # Random token IDs
    input2 = torch.rand(batch_size, seq_len, model_dim)
    print(
        f"Input1 shape: {embedder(input1).shape}"
    )  # Expected shape: (batch_size, seq_len, model_dim)
    print(
        f"Input2 shape: {input2.shape}"
    )  # Expected shape: (batch_size, seq_len, model_dim)

    transformer = Transformer(unique_tokens, model_dim, num_heads, max_len=100)

    transformer_output = transformer(input1)
    print(
        f"Transformer output shape: {transformer_output.shape}"
    )  # Expected shape: (batch_size, seq_len, model_dim)
