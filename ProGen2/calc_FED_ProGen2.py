from scipy import linalg
import numpy as np
import torch
import time
import sys
import os

def calc_FED(mu_real, sigma_real, mu_gen, sigma_gen):
    eps = 1e-06

    assert mu_real.shape == mu_gen.shape, 'Training and test mean vectors have different lengths'
    assert sigma_real.shape == sigma_gen.shape, 'Training and test covariances have different dimensions'

    covmean, _ = linalg.sqrtm(np.dot(sigma_real, sigma_gen), disp=False)

    # Product might be almost singular
    if not np.isfinite(covmean).all():
        msg = ('FED calculation produces singular product, adding %s to diagonal of cov estimates') % eps
        print(msg)

        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm(np.dot((sigma_real + offset), (sigma_gen + offset)))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    FED = np.sum((mu_real - mu_gen)**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    print("Fréchet ESM Distance: ", FED)
    return FED

folder_path = str(sys.argv[1])
path_real_embeddings = str(sys.argv[2])
ESM_models = ["8M", "35M", "150M", "650M", "3B", "15B"]
output_path = str(sys.argv[3])

filesnamesESM_real_embeddings = sorted(os.listdir(path_real_embeddings))
paths_real = []

for file_name in filesnamesESM_real_embeddings:
    file_path_real = os.path.join(path_real_embeddings, file_name)
    paths_real.append(file_path_real)

print(paths_real)

# Get a list of filenames in the folder
filenames = sorted(os.listdir(folder_path))
model_count = 0

avg_FED_per_ESM_model = []
stdev_FED_per_ESM_model = []
avg_time_per_ESM_model = []
stdev_time_per_ESM_model = []

# Iterate over the filenames and load each file using torch.load
for filename in filenames:
    file_path = os.path.join(folder_path, filename)

    progen2_small_emb_650M = torch.load(file_path)
    real_emb_650M = torch.load(paths_real[model_count])

    nan_mask = np.isnan(progen2_small_emb_650M.numpy())
    valid_rows_mask = np.logical_not(np.any(nan_mask, axis=1))
    progen2_small_emb_650M_clean = progen2_small_emb_650M[valid_rows_mask]

    # Print the shape of the cleaned tensor
    print(progen2_small_emb_650M_clean.shape)

    cycles = 10

    real_set_size = 100000

    list_FED = []
    list_times = []

    for i in range(cycles): #cycle over random seed for better reproducibility
        print("Cycle", i+1, "ESM2 model", ESM_models[model_count])
        start = time.time()
        real_emb_650M = real_emb_650M[torch.randperm(real_emb_650M.shape[0])] #Shuffle (dataloader was not shuffled)
        real = real_emb_650M[0:real_set_size].numpy()

        #assert avg_emb_esm2_t33_650M_UR50D.shape[0] == np.concatenate((gen, real), axis=0).shape[0], "Missing a sample?"

        mu_real, sigma_real = np.mean(real, axis = 0), np.cov(real, rowvar=False)
        mu_gen, sigma_gen = np.mean(progen2_small_emb_650M_clean.numpy(), axis = 0), np.cov(progen2_small_emb_650M_clean.numpy(), rowvar=False)

        list_FED.append(calc_FED(mu_real, sigma_real, mu_gen, sigma_gen))
        stop = time.time()
        list_times.append(stop-start)

    with open(f"{output_path}testtCalcFED.txt", 'a') as f:
        f.write(f"Embedding used from ESM2 model with {ESM_models[model_count]} parameters \n")
        f.write(f"Average FED over {cycles} cycles: {np.round(sum(list_FED)/len(list_FED), decimals=3)} \n")
        f.write(f"Standard deviation of FED over {cycles} cycles: {np.std(list_FED)} \t rounded: {np.round(np.std(list_FED), decimals=3)} \n")
        f.write(f"Time elapsed for {cycles} cycles: {sum(list_times)} seconds. \t Time per cycle: {np.round(sum(list_times)/len(list_times), decimals=3)} seconds. \t stdev: {np.round(np.std(list_times), decimals=3)} \n \n")
        f.flush()
    avg_FED_per_ESM_model.append(np.round(sum(list_FED)/len(list_FED), decimals=3))
    stdev_FED_per_ESM_model.append(np.round(np.std(list_FED), decimals=3))
    avg_time_per_ESM_model.append(np.round(sum(list_times)/len(list_times), decimals=3))
    stdev_time_per_ESM_model.append(np.round(np.std(list_times), decimals=3))

    model_count += 1

print(avg_FED_per_ESM_model)
print(stdev_FED_per_ESM_model)
print(avg_time_per_ESM_model)
print(stdev_time_per_ESM_model)

torch.save(avg_FED_per_ESM_model, f"{output_path}avg_FED_per_ESM_model.t")
torch.save(stdev_FED_per_ESM_model, f"{output_path}stdev_FED_per_ESM_model.t")
torch.save(avg_time_per_ESM_model, f"{output_path}avg_time_per_ESM_model.t")
torch.save(stdev_time_per_ESM_model, f"{output_path}stdev_time_per_ESM_model.t")
