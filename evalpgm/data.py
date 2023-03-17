import torch
import h5torch
import h5py
import pytorch_lightning as pl
import numpy as np
from functools import partial

tok_to_idx_esm2_150M = {
    '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10,
    'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17,
    'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24,
    'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30,
    '<null_1>': 31, '<mask>': 32
} # following the tokenization of the ESM models.

idx_to_tok_esm2_150M = {v : k for k,v in tok_to_idx_esm2_150M.items()}


def pad_batch(f, sample, pad_to = 1024):
    sequence = sample["central"]
    length = sample["0/protein_length"]
    sequence = np.concatenate([sequence, np.ones((pad_to - length), dtype=sequence.dtype)])
    return sequence, length.astype(np.float32)
    

class Uniref50DataModule(pl.LightningDataModule):
    def __init__(self, path, pad_to = 1024, batch_size=16, n_workers=4, subsample = None):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size
        padder = partial(pad_batch, pad_to = pad_to)

        subset = h5py.File(path)["0/protein_length"][:] <= pad_to
    
        split = h5py.File(path)["unstructured/split"][:] == b"train"
        subset_train = (
            subset & split if subsample is None else
            subset & split & (np.random.rand(len(split)) < subsample)
        )
        self.train = h5torch.Dataset(path, subset = subset, sample_processor = padder)

        split = h5py.File(path)["unstructured/split"][:] == b"val"
        subset_val = (
            subset & split if subsample is None else
            subset & split & (np.random.rand(len(split)) < subsample)
        )
        self.val = h5torch.Dataset(path, subset = subset_val, sample_processor = padder)

        split = h5py.File(path)["unstructured/split"][:] == b"test"
        subset_test = (
            (subset & split) if subsample is None else
            (subset & split & (np.random.rand(len(split)) < subsample))
        )

        self.test = h5torch.Dataset(path, subset = subset_test, sample_processor = padder)
        print(len(self.train), len(self.val), len(self.test))

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