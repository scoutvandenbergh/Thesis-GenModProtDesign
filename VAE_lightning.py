import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re

data = pd.read_csv("pdb_data_seq.csv")
data = data[data['sequence'].notna()]
data["residueCount"] = data["sequence"].str.len().astype(int) #fixed with this

min_length = 64 #discard smaller than this
max_length = 1024 #+ padding
data = data.drop(data[(data["macromoleculeType"] != "Protein") |
                 (data["residueCount"] < min_length) | 
                 (data["residueCount"] > max_length)].index)

data = data.drop(["structureId", "chainId", "macromoleculeType"], axis = 1)
data = data.reset_index()

indices = []
indicesCounter = 0
for sequence in data["sequence"]:
    if any(c in sequence for c in "XUZBO"):
        indices.append(indicesCounter)
    indicesCounter += 1
data = data.drop(indices)
data = data.reset_index()

# calculate the frequency of each AA
aaCount = data['sequence'].str.split('').explode().value_counts()

# convert the AA counts to a dictionary
aaFrequencies = aaCount.to_dict()
_ = aaFrequencies.pop('')
aaTotal = sum(aaFrequencies.values())

for aa in aaFrequencies.items():
    aaFrequencies[aa[0]] = aa[1]/aaTotal

groundTruth = [] #"ARNDCEQGHILKMFPSTWYV"

AA2number = {'A' : 0, 'R': 1, 'N': 2, 'D' : 3, 'C' : 4, 'E' : 5, 'Q': 6, 
             'G' : 7, 'H' : 8, 'I' : 9, 'L' : 10, 'K' : 11, 'M': 12, 'F' : 13, 
             'P' : 14, 'S' : 15, 'T' : 16, 'W' : 17, 'Y' : 18, 'V': 19
             }

for sequence in data["sequence"]:
  testseq = np.array(list(sequence))
  groundTruth.append(np.array([AA2number[letter] for letter in np.array(list(sequence))]))

data["ground_truth"] = groundTruth

paddedSequence = []
for sequence in data["ground_truth"]:
  if len(sequence) < max_length:
    padSeq = np.full(shape = max_length - len(sequence), fill_value= 20)
    sequence = np.concatenate((sequence, padSeq), axis=None)
  paddedSequence.append(sequence)
data["ground_truth_padded"] = paddedSequence

print("--------------------- Pre-processing done -------------------")

p = 0.01
idx = torch.randperm(len(data["ground_truth_padded"]))[:int(len(data["ground_truth_padded"])*p)]

Xtrain = torch.tensor(data["ground_truth_padded"])
Xtrain = Xtrain[idx]
ytrain = torch.tensor(data["ground_truth_padded"])
ytrain = ytrain[idx]

train_set, val_set, test_set = torch.utils.data.random_split(
    idx, [int(len(idx)*0.80), int(len(idx)*0.15), len(idx) - int(len(idx)*0.80) - int(len(idx)*0.15)])

X_train = Xtrain[train_set.indices]
y_train = ytrain[train_set.indices]

X_val = Xtrain[val_set.indices]
y_val = ytrain[val_set.indices]

X_test = Xtrain[test_set.indices]
y_test = ytrain[test_set.indices]

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, pin_memory=True, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, pin_memory=True, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, pin_memory=True, shuffle=True)

print("------------------- Data splitting done --------------------")

from VAE_models_lightning import VAE # change this to take other models

# model
#model = pl.LightningModule(nn.DataParallel(VAE())) for multiple gpu
model = VAE()

print("----------------- importing model architecture done -------------------")

from pytorch_lightning.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=True, mode='min')

print("----------------- implementing early stopping done -----------------------")

trainer = pl.Trainer(
	max_epochs=10,
	accelerator="gpu",
	devices=1, #if multiple gpus
  check_val_every_n_epoch=1,
  callbacks=[early_stopping])

print("------------------- initializing trainer done ---------------------")
#tensorboard --logdir logs in terminal --> not showing up anything in browser
#figure out where params etc are saved, good values for gradient clipping, LR scheduler, ...

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, test_dataloader)
