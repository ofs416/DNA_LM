from src.data.tokeniser import NucleotideTokeniser


def test_nucleotide_tokeniser():
    tokeniser = NucleotideTokeniser(max_length=10)
    sequence = "ACGTN"
    tokens = tokeniser.tokenise(sequence)
    assert tokens == [5, 0, 1, 2, 3, 4, 6, 7, 7, 7], "Tokenisation is incorrect"
