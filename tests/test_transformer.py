import torch
from src.models.transformer import MHA, Transformer


def test_mha_different_batch_sizes():
    """Test MHA with various batch sizes"""
    model_dim = 256
    num_heads = 4
    seq_len = 8

    mha = MHA(model_dim, num_heads)

    for batch_size in [1, 4, 16]:
        input_tensor = torch.rand(batch_size, seq_len, model_dim)
        output = mha(input_tensor)
        assert output.shape == (batch_size, num_heads, seq_len, model_dim)


def test_mha_gradient_flow():
    """Test that gradients flow through MHA"""
    model_dim = 128
    num_heads = 2
    batch_size = 2
    seq_len = 5

    mha = MHA(model_dim, num_heads)
    input_tensor = torch.rand(batch_size, seq_len, model_dim, requires_grad=True)
    output = mha(input_tensor)
    loss = output.sum()
    loss.backward()

    assert input_tensor.grad is not None
    assert mha.W_q.weight.grad is not None


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
