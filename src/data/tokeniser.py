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
        tokens = [self.nucleotide_to_id["CLS"]]
        for nucleotide in sequence:
            tokens.append(
                self.nucleotide_to_id.get(nucleotide, self.nucleotide_to_id["N"])
            )
        tokens.append(self.nucleotide_to_id["SEP"])
        while len(tokens) < self.max_length:
            tokens.append(self.nucleotide_to_id["PAD"])

        return tokens


if __name__ == "__main__":
    tokeniser = NucleotideTokeniser(max_length=10)
    sequence = "ACGTN"
    tokens = tokeniser.tokenise(sequence)
    print(tokens)
