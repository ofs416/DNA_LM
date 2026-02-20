# test_encoder.py
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.embedding import PositionalEncoder

# Test the encoder
encoder = PositionalEncoder(d_model=512, max_len=100)
print(f"Encoder created: {encoder}")
print(f"Expected shape for position 0: {encoder.pe[:, 0].shape}")