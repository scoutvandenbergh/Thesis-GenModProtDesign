import esm
#from esm.pretrained import esm2_t48_15B_UR50D #48 layer transformer with 15B parameters trained on UniRef50D

import torch
import numpy as np
from evalpgm.data import Uniref50DataModule
import sys

np.random.seed(42)

data_path = str(sys.argv[1])
output_path = str(sys.argv[2])

embeddings_path = output_path + "/emb_esm2_t33_650M_UR50D_subsample0_01.t"
avg_embeddings_path = output_path + "/avg_per_seq_emb_esm2_t33_650M_UR50D_subsample0_01.t"


batch_size = 16
dm = Uniref50DataModule(data_path, batch_size = batch_size, n_workers = 8, subsample = 0.01)
print("data splitted \n")

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() ##33 layer transformer with 650M params trained on UniRef50D
model.to("cuda:1")
model.eval()  # disables dropout for deterministic results
print("esm2_t33_650M_UR50D loaded \n")

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(torch.from_numpy(np.concatenate([x[0].reshape(1, -1) for x in dm.test])), repr_layers=[33], return_contacts=False)
token_representations = results["representations"][33]

torch.save(results["representations"][33], embeddings_path)
print(results["representations"][33].shape)

# Generate per-sequence representations via averaging
sequence_representations = []
lengths = torch.from_numpy(np.array([int(x[1]) for x in dm.test]))
print(lengths, type(lengths))

for i, tokens_len in enumerate(lengths):
    sequence_representations.append(token_representations[i, 0 : tokens_len].mean(0))
    print(sequence_representations, sequence_representations[0].size())
torch.save(sequence_representations, avg_embeddings_path)
print(sequence_representations)

#Placeholder file, look for correct code @generate_embeddings.py