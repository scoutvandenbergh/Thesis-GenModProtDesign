import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import datetime
import time


# GENERATE NOVEL PROTEIN SEQUENCES

def generate_sequences(model, amount = 25000, temperature = 0.8):
    sampledProteins = []
    batch_size = 256

    start = time.time()

    z = torch.randn(amount, 128).cuda()

    dataloader = torch.utils.data.DataLoader(z, batch_size=batch_size, shuffle=False) #use dataloader to keep VRAM requirements low during inference. 25k at once = > 11k VRAM, 256 at once is about .5GB VRAM

    # generate new samples
    for batch in dataloader: #no tqdm to limit amount of unneccessary output
        with torch.no_grad():
            output = model.decode(batch)
            softmax = nn.Softmax(dim=-1)
            output_softmax = softmax(output/temperature).double().cpu().transpose(-2, -1) # transpose required to sample from Categorical, this allows vectorized sampling instead of for loops (seconds instead of hours)

            for seq in output_softmax:
                dist = torch.distributions.Categorical(seq)
                sampledFromSoftmax = dist.sample().cpu().numpy().tolist()
                sampledProteins.append(sampledFromSoftmax)

    stop = time.time()

    print(f"Time elapsed to temperature sample {amount} novel protein sequences:{datetime.timedelta(seconds=stop-start)}")
    sampledProteins_tensor = torch.tensor(sampledProteins)
    print(sampledProteins_tensor.shape) #otherwise a nested list

    return sampledProteins_tensor