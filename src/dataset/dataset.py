"""
Dataset utilities.
"""

import pandas as pd
from torch.utils.data import Dataset


class CFGDataset(Dataset):
    """
    Dataset used for CFG training.
    """

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        return {
            "smiles": row["smiles"],
            "qed": float(row["qed"]),
            "logp": float(row["logp"]),
            "tpsa": float(row["tpsa"]),
            "sas": float(row["sas"]),
        }


def load_datasets(cfg):

    train_dataset = CFGDataset(cfg["train"]["path"])
    val_dataset = CFGDataset(cfg["validation"]["path"])
    test_dataset = CFGDataset(cfg["test"]["path"])

    return train_dataset, val_dataset, test_dataset