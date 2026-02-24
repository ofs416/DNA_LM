import torch
from src.models.transformer import Transformer


def test_transformer_output_shape():
    """Test Transformer produces correct output shape"""
    model_dim = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10
    max_len = 100

    transformer = Transformer(model_dim, num_heads, max_len)
    input_tensor = torch.randn(batch_size, seq_len, model_dim)
    output = transformer(input_tensor)

    assert output.shape == (batch_size, seq_len, model_dim)


def test_transformer_gradient_flow():
    """Test that gradients flow through entire Transformer"""
    model_dim = 128
    num_heads = 2
    batch_size = 2
    seq_len = 5
    max_len = 50

    transformer = Transformer(model_dim, num_heads, max_len)
    input_tensor = torch.randn(batch_size, seq_len, model_dim)
    output = transformer(input_tensor)
    loss = output.sum()
    loss.backward()

    assert transformer.mha.W_q.weight.grad is not None
    assert transformer.dense.weight.grad is not None
