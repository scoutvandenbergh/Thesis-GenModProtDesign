import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import datetime
import time


# GENERATE NOVEL PROTEIN SEQUENCES

def generate_sequences(model, amount = 25000, temperature = 0.8, bottleneck_shape = (128,)):
    sampledProteins = []
    batch_size = 256

    start = time.time()

    z = torch.randn(amount, *bottleneck_shape).cuda()
    model.eval()
    roundedLengths = []

    dataloader = torch.utils.data.DataLoader(z, batch_size=batch_size, shuffle=False)

    # generate new samples
    for batch in dataloader:
        with torch.no_grad():
            output, length_pred = model.decode(batch)
            lengths = length_pred.squeeze()*1024
            rounding = [round(num) for num in lengths.tolist()]
            roundedLengths.extend(rounding)
            softmax = nn.Softmax(dim=-1)
            output_softmax = softmax(output.transpose(-2, -1)/temperature).double().cpu()

            # Check if the output from softmax sums to 1, if not normalize, fix by GPT4 , required due to numerical issues for certain temperatures
            if not torch.allclose(output_softmax.sum(-1), torch.tensor(1.0).double(), atol=1e-5):
                print("Sum of softmax outputs does not equal 1. Normalizing.")
                output_softmax = output_softmax + 1e-8  # Add a small epsilon
                output_softmax = output_softmax / output_softmax.sum(-1, keepdim=True)

            for i, seq in enumerate(output_softmax):
                dist = torch.distributions.Categorical(seq)
                sampledFromSoftmax = dist.sample().cpu().numpy().tolist()
                sampledFromSoftmax = sampledFromSoftmax[:roundedLengths[i]] + [1]*(1024-roundedLengths[i])
                sampledProteins.append(sampledFromSoftmax)

    stop = time.time()

    print(f"Time elapsed to temperature sample {amount} novel protein sequences:{datetime.timedelta(seconds=stop-start)}")
    
    sampledProteins_tensor = torch.tensor(sampledProteins)
    print(sampledProteins_tensor.shape)
    return sampledProteins_tensor, roundedLengths, np.mean(roundedLengths) 
 