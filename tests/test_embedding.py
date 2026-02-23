import torch


from src.models.embedding import PositionalEncoder


def test_positional_encoding():
    encoder = PositionalEncoder(max_len=100, d_model=512)
    pe = encoder.pe
    assert pe.shape == (100, 512), "Positional encoding shape is incorrect"

    pos_0 = pe[0, :]
    assert torch.all(
        (pos_0 == 0.0) | (pos_0 == 1.0)
    ), "Positional encoding for position 0 should be zeros and ones"
