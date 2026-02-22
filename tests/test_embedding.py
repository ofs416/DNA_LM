# test_encoder.py
import sys
import os

import torch


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.embedding import PositionalEncoder


def test_positional_encoding():
    encoder = PositionalEncoder(max_len=100, d_model=512)
    pe = encoder.pe
    assert pe.shape == (100, 512), "Positional encoding shape is incorrect"

    pos_0 = pe[0, :]
    assert torch.all(
        (pos_0 == 0.0) | (pos_0 == 1.0)
    ), "Positional encoding for position 0 should be zeros and ones"
