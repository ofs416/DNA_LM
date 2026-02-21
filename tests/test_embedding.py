# test_encoder.py
import sys
import os

import torch


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.embedding import PositionalEncoder


def test_positional_encoding():
    encoder = PositionalEncoder(d_model=512, max_len=100)
    pe = encoder.pe
    assert pe.shape == (100, 512), "Positional encoding shape is incorrect"

    pos_0 = pe[:, 0]
    assert torch.all(
        pos_0 == 0.0
    ), "Positional encoding for position 0 should be all zeros"
    pos_1 = pe[:, 1]
    assert torch.all(
        pos_1 == 1.0
    ), "Positional encoding for position 1 should be all ones"
