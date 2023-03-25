import torch
import numpy as np
from evalpgm.data import Uniref50DataModule
from tqdm import tqdm
import esm
import time
import sys

np.random.seed(42)

data_path = str(sys.argv[1])
output_path = str(sys.argv[2])

#embeddings_path = output_path + "/emb_esm2_t33_650M_UR50D_subsample0_01.t"
avg_embeddings_path = output_path + "/avg_per_seq_emb_esm2_t33_650M_UR50D_subsample0_25.t"


batch_size = 16
dm = Uniref50DataModule(data_path, batch_size = batch_size, n_workers = 8, subsample = 0.25)
print("data splitted \n")

test = torch.cat([torch.tensor(x[0]) for x in dm.test_dataloader()])
print(test.shape)
dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, pin_memory=True, shuffle=False)

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() ##33 layer transformer with 650M params trained on UniRef50D
model.to("cuda:0")
model.eval()  # disables dropout for deterministic results
print("esm2_t33_650M_UR50D loaded \n")

representations = []
lengths = torch.stack([torch.tensor(int(x[1])) for x in dm.test])
sequence_representations = []

counter = 0
times_embeddings = []

for batch in tqdm(dataloader):
    with torch.no_grad():
        start = time.time()
        results = model(batch.to("cuda:0"), repr_layers=[33], return_contacts=False)
        stop = time.time()
    times_embeddings.append(stop-start)
    token_representations = torch.tensor(results["representations"][33].cpu())
    
    for i, tokens_len in enumerate(lengths[counter:counter+batch_size]): #fix lengths so this only goes in steps of batch_size, so in first batch loop iteration it has the element 0 until element 15, second batch iteration from 16 to 31 etc
        sequence_representations.append(token_representations[i, 0 : tokens_len ].mean(0))
    counter += batch_size
sequence_representations = torch.stack(sequence_representations, dim=0)

torch.save(sequence_representations, avg_embeddings_path)

print(sequence_representations.shape)
print(sum(times_embeddings), len(times_embeddings))
print("Average time per batch of", batch_size, "proteins through esm2_t33_650M_UR50D: ", sum(times_embeddings)/len(times_embeddings)) 
print("Average time per sequence: ", sum(times_embeddings)/len(times_embeddings)/batch_size)


# Output for 1% of data:
# torch.Size([8262, 1280])
# 681.4922761917114 517
# Average time per batch of 16 proteins through esm2_t33_650M_UR50D:  1.3181668785139486
# Average time per sequence:  0.08238542990712179
# But runtime of tqdm was 42:05 for 517 batches... 

