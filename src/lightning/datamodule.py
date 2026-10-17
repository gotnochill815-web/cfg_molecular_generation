"""
PyTorch Lightning DataModule for ChemFM CFG training.
"""

import lightning as L
from torch.utils.data import DataLoader

from src.training.collator import CFGDataCollator


class CFGDataModule(L.LightningDataModule):
    """
    Lightning DataModule for ChemFM Classifier-Free Guidance training.

    Responsibilities:
        - Hold train/validation/test datasets
        - Build DataLoaders
        - Create the CFG data collator
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        tokenizer,
        batch_size=1,
        num_workers=4,
        dropout_prob=0.2,
        max_length=512,
        conditioning_mode="random",
        pin_memory=True,
        persistent_workers=False ,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pin_memory = pin_memory
        self.persistent_workers = (
            persistent_workers if num_workers > 0 else False
        )

        # CFG dropout and configurable conditioning for training.
        self.train_collator = CFGDataCollator(
            tokenizer=tokenizer,
            dropout_prob=dropout_prob,
            max_length=max_length,
            conditioning_mode=conditioning_mode,
        )

        # Deterministic, fully conditioned validation and testing.
        self.eval_collator = CFGDataCollator(
            tokenizer=tokenizer,
            dropout_prob=0.0,
            max_length=max_length,
            conditioning_mode="all",
        )

    def setup(self, stage=None):
        """
        Datasets are already created in train.py,
        so nothing needs to be done here.
        """
        pass

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self):

        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )