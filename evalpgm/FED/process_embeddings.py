import torch
import numpy as np
from tqdm import tqdm
import esm
import sys
import time
from pathlib import Path

def load_model_and_alphabet_local(model_location): 
    """Load from local path. The regression weights need to be co-located""" 
    model_location = Path(model_location) 
    model_data = torch.load(str(model_location), map_location="cpu")
    model_name = model_location.stem
    regression_data = None
    return esm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data) 

np.random.seed(42)
torch.manual_seed(42)

def process_embeddings_ESM2_35M(model, data, lengths):
    total_time_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16

    data = torch.tensor(data)
    print("data.shape", data.shape)
    lengths = torch.tensor(lengths)
    print("lengths.shape", lengths.shape)

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=True, shuffle=False)
    print("dataloader ok \n")

    # # Load the ESM-2 model
    #model, alphabet = load_model_and_alphabet_local(weights_path)
    #print("model loaded")
    model.to(device)
    print("model on device", device.type, "\n")
    model.eval()

    # Calculate and save the embeddings for the selected test sequences
    sequence_representations = []
    counter = 0

    for i, batch in enumerate(tqdm(dataloader)):
        if i % 250 == 0:
            print(f"Step{i}")

        with torch.no_grad():
            results = model(batch.to(device), repr_layers=[12], return_contacts=False)
        token_representations = torch.tensor(results["representations"][12].cpu())
        for i, tokens_len in enumerate(lengths[counter:counter+batch_size]):
            sequence_representations.append(token_representations[i, 0 : tokens_len ].mean(0))
        counter += batch_size
    sequence_representations = torch.stack(sequence_representations, dim=0)

    total_time_stop = time.time()

    print("Took ", total_time_stop-total_time_start, "seconds to run the entire script.")

    print("sequence_representations.shape", sequence_representations.shape)

    return sequence_representations