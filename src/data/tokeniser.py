import torch


class NucleotideTokeniser:
    def __init__(self, max_length=300):
        self.max_length = max_length
        self.nucleotide_to_id = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "N": 4,  # For unknown nucleotides
            "CLS": 5,  # Classification token
            "SEP": 6,  # Separator token (for the end of sequences)
            "PAD": 7,  # Padding token
            "MASK": 8,  # Mask token (for masked language modeling)
        }
        self.id_to_nucleotide = {v: k for k, v in self.nucleotide_to_id.items()}

    def tokenise(self, sequence):
        token_tensor = torch.tensor(self.nucleotide_to_id["PAD"]).repeat(
            self.max_length
        )
        token_tensor[0] = self.nucleotide_to_id["CLS"]  # Start with CLS token
        for i, nucleotide in enumerate(sequence[: self.max_length - 2], start=1):
            token_tensor[i] = self.nucleotide_to_id.get(
                nucleotide, self.nucleotide_to_id["N"]
            )
        token_tensor[len(sequence) + 1] = self.nucleotide_to_id[
            "SEP"
        ]  # End with SEP token

        return token_tensor


if __name__ == "__main__":
    tokeniser = NucleotideTokeniser(max_length=10)
    sequence = "ACGTN"
    tokens = tokeniser.tokenise(sequence)
    print(tokens)
