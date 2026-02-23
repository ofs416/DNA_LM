import torch

from src.data.tokeniser import NucleotideTokeniser


def test_nucleotide_tokeniser():
    tokeniser = NucleotideTokeniser(max_length=10)
    sequence = "ACGTN"
    tokens = tokeniser.tokenise(sequence)
    expected = torch.tensor([5, 0, 1, 2, 3, 4, 6, 7, 7, 7])
    assert torch.equal(tokens, expected), f"Expected {expected}, but got {tokens}"
