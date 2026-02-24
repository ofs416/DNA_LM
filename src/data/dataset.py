import os
import torch
from torch.utils.data import Dataset

from src.data.tokeniser import NucleotideTokeniser


class NucleotideDataset(Dataset):
    def __init__(self, data_dir, max_length=300):
        print(f"Using data directory: {data_dir}")
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokeniser = NucleotideTokeniser(max_length=max_length)
        self.sequences = self._load_sequences()

    def _load_sequences(self):
        sequences = []
        labels = []
        for foldername1 in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, foldername1)
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    if foldername1 == "positive":
                        labels.append(1)
                    elif foldername1 == "negative":
                        labels.append(0)

                    with open(os.path.join(folder_path, filename), "r") as f:
                        sequence = f.read().strip()
                        tokenised_sequence = self.tokeniser.tokenise(sequence)
                        sequences.append(tokenised_sequence)
        return {
            "sequences": sequences,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.sequences["sequences"])

    def __getitem__(self, idx):
        sequence = self.sequences["sequences"][idx]
        label = self.sequences["labels"][idx]
        return sequence, label

    def save(self, save_path):
        torch.save(self.sequences, save_path)

    @classmethod
    def load(cls, load_path):
        sequences = torch.load(load_path)
        dataset = cls(data_dir=load_path, max_length=sequences["sequences"][0].shape[0])
        dataset.sequences = sequences
        return dataset


if __name__ == "__main__":
    print("Running dataset.py...")

    dataset = NucleotideDataset(
        data_dir=r"data\raw\human_nontata_promoters\test", max_length=300
    )
    print(f"Dataset length: {len(dataset)}")

    print(f"Sample: {dataset[0]}")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    features, labels = next(iter(dataloader))
    print(f"Batch features: {features}")
    print(f"Batch labels: {labels}")

    dataset.save("data/processed/human_nontata_promoters/test/nucleotide_dataset.pt")

    loaded_dataset = NucleotideDataset.load(
        "data/processed/human_nontata_promoters/test/nucleotide_dataset.pt"
    )
    print(f"Loaded dataset length: {len(loaded_dataset)}")
    print(f"Loaded sample: {loaded_dataset[0]}")
