import torch
import h5torch
import h5py
import pytorch_lightning as pl
import numpy as np
import os

class PfamDataset(h5torch.Dataset):
    def __init__(self, path, pad_to = 1024):
        subset = h5py.File(path)["0/protein_length"][:] <= pad_to
        super().__init__(path, subset = subset)
        self.pad_to = pad_to
    def __getitem__(self, index):
        items = super().__getitem__(index)
        sequence = items["central"]
        length = items["0/protein_length"]

        sequence = np.concatenate([sequence, np.ones((self.pad_to - length), dtype=sequence.dtype)])
        return sequence, length

class PfamDataModule(pl.LightningDataModule):
    def __init__(self, rootfolder, pad_to = 1024, batch_size=16, n_workers=4):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        self.train = PfamDataset(os.path.join(rootfolder, "train.h5t"), pad_to = pad_to)
        self.val = PfamDataset(os.path.join(rootfolder, "valid.h5t"), pad_to = pad_to)
        self.test = PfamDataset(os.path.join(rootfolder, "holdout.h5t"), pad_to = pad_to)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )