import torch.nn as nn
import torch

from src.models.embedding import AbsPosEncoder
from src.models.attention import MHA


class Transformer(nn.Module):
    def __init__(self, model_dim, num_heads, max_len):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.mha = MHA(model_dim, num_heads)
        self.embedding = AbsPosEncoder(max_len, model_dim)
        self.dense = nn.Linear(model_dim * num_heads, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        embeddings_norm = self.norm(embeddings)
        mha_output_norm, attention_weights = self.mha(
            embeddings_norm, embeddings_norm, embeddings_norm
        )
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
    input = torch.rand(batch_size, seq_len, model_dim)

    transformer = Transformer(model_dim, num_heads, max_len=100)

    transformer_output = transformer(input)
    print(
        f"Transformer output shape: {transformer_output.shape}"
    )  # Expected shape: (batch_size, seq_len, model_dim)
