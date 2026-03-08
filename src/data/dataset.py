import os
import torch
from torch.utils.data import Dataset

from src.data.tokeniser import NucleotideTokeniser


class NucleotideDataset(Dataset):
    def __init__(self, data_dir, raw=True, max_length=300):
        print(f"Using data directory: {data_dir}")
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokeniser = NucleotideTokeniser(max_length=max_length)

        if raw:
            self.sequences = self._load_sequences()
        else:
            self.sequences = torch.load(data_dir)

    def _load_sequences(self):
        sequences = []
        attention_masks = []
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

                    attention_mask = (tokenised_sequence != 7).long()
                    attention_masks.append(attention_mask)
        return {
            "sequences": sequences,
            "attention_masks": attention_masks,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.sequences["sequences"])

    def __getitem__(self, idx):
        sequence = self.sequences["sequences"][idx]
        attention_mask = self.sequences["attention_masks"][idx]
        label = self.sequences["labels"][idx]
        return sequence, attention_mask, label

    def save(self, save_path):
        torch.save(self.sequences, save_path)


if __name__ == "__main__":
    print("Running dataset.py...")

    dataset = NucleotideDataset(
        data_dir=r"data\raw\human_nontata_promoters\test", raw=True, max_length=256
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset Sample: {dataset[0]}")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    features, masks, labels = next(iter(dataloader))
    print(f"Batch features: {features}")
    print(f"Batch masks: {masks}")
    print(f"Batch labels: {labels}")

    dataset.save("data/processed/human_nontata_promoters/test/nucleotide_dataset.pt")

    # loaded_dataset = NucleotideDataset(
    #     data_dir=r"data\processed\human_nontata_promoters\test\nucleotide_dataset.pt",
    #     raw=False,
    #     max_length=300,
    # )
    # print(f"Loaded dataset length: {len(loaded_dataset)}")
    # print(f"Loaded sample: {loaded_dataset[0]}")
