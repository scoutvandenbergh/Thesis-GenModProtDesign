import torch
import numpy as np
from evalpgm.data import Uniref50DataModule
from tqdm import tqdm
import esm
import time
import sys
from pathlib import Path

def load_model_and_alphabet_local(model_location): 
    """Load from local path. The regression weights need to be co-located""" 
    model_location = Path(model_location) 
    model_data = torch.load(str(model_location), map_location="cpu")
    model_name = model_location.stem
    regression_data = None
    return esm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data) 

np.random.seed(42)

data_path = str(sys.argv[1])
output_path = str(sys.argv[2])
weights_path = str(sys.argv[3])

#embeddings_path = output_path + "/emb_esm2_t33_650M_UR50D_subsample0_01.t"
avg_embeddings_path = output_path + "/COMPLETE_avg_per_seq_emb_esm2_t30_150M_UR50D.t"


batch_size = 16
dm = Uniref50DataModule(data_path, batch_size = batch_size, n_workers = 8, subsample = None)
print("data splitted \n")

test = torch.cat([torch.tensor(x[0]) for x in dm.test_dataloader()])
print(test.shape)
dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, pin_memory=True, shuffle=False) #Should be true I think, to compare 200k test samples vs 5-50-500-5k-50k-300k test
#NOTE: not a problem anymore using emb[torch.randperm(emb.shape[0])]

# Load ESM-2 model
#model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() ##33 layer transformer with 650M params trained on UniRef50D
model, alphabet = load_model_and_alphabet_local(weights_path)

model.to("cuda:0")
model.eval()  # disables dropout for deterministic results
print("esm2_t30_150M_UR50D loaded \n")

representations = []
lengths = torch.stack([torch.tensor(int(x[1])) for x in dm.test])
sequence_representations = []

counter = 0
times_embeddings = []

for batch in tqdm(dataloader):
    with torch.no_grad():
        start = time.time()
        results = model(batch.to("cuda:0"), repr_layers=[30], return_contacts=False)
        stop = time.time()
    times_embeddings.append(stop-start)
    token_representations = torch.tensor(results["representations"][30].cpu())
    
    for i, tokens_len in enumerate(lengths[counter:counter+batch_size]): #fix lengths so this only goes in steps of batch_size, so in first batch loop iteration it has the element 0 until element 15, second batch iteration from 16 to 31 etc
        sequence_representations.append(token_representations[i, 0 : tokens_len ].mean(0))
    counter += batch_size
sequence_representations = torch.stack(sequence_representations, dim=0)

torch.save(sequence_representations, avg_embeddings_path)

print(sequence_representations.shape)
print(sum(times_embeddings), len(times_embeddings))
print("Average time per batch of", batch_size, "proteins through esm2_t30_150M_UR50D: ", sum(times_embeddings)/len(times_embeddings)) 
print("Average time per sequence: ", sum(times_embeddings)/len(times_embeddings)/batch_size)


# Output for 1% of data:
# torch.Size([8262, 1280])
# 681.4922761917114 517
# Average time per batch of 16 proteins through esm2_t33_650M_UR50D:  1.3181668785139486
# Average time per sequence:  0.08238542990712179
# But runtime of tqdm was 42:05 for 517 batches...  

# Output for 25% of data:
# torch.Size([204004, 1280])
# 17058.13907957077 12751
# Average time per batch of 16 proteins through esm2_t33_650M_UR50D:  1.3377883365673884
# Average time per sequence:  0.08361177103546177
# But runtime of tqdm was 17:35:33 for 12751 batches...


# Output for 100% of data using esm2_t33_650M_UR50D model
# torch.Size([817350, 1280])
# 68792.55374884605 51085
# Average time per batch of 16 proteins through esm2_t33_650M_UR50D:  1.3466292208837438
# Average time per sequence:  0.08416432630523399

# Output for 100% of data using esm2_t6_8M_UR50D model
# torch.Size([817350, 320])
# 399.169970035553 51085
# Average time per batch of 16 proteins through esm2_t6_8M_UR50D:  0.007813839092405853
# Average time per sequence:  0.0004883649432753658

# Output for 100% of data using esm2_t12_35M_UR50D model
# torch.Size([817350, 480])
# 614.3640744686127 51085
# Average time per batch of 16 proteins through esm2_t12_35M_UR50D:  0.012026310550427967
# Average time per sequence:  0.000751644409401748
