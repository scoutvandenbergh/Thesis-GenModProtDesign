from scipy import linalg
import numpy as np
import torch
import time

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

def calc_avg_FED(path_real_embeddings, generated_embeddings, cycles = 10, real_set_size = 100000):
    real_emb = torch.load(path_real_embeddings)
    generated_emb = generated_embeddings

    nan_mask = np.isnan(generated_emb.numpy())
    valid_rows_mask = np.logical_not(np.any(nan_mask, axis=1))
    generated_emb = generated_emb[valid_rows_mask]
    print("generated_emb.shape, some might be deleted due to NaNs", generated_emb.shape)

    cycles = cycles

    real_set_size = real_set_size

    list_FED = []
    list_times = []

    for i in range(cycles): #cycle over random seed for better reproducibility
        print("Cycle", i+1)
        start = time.time()
        real_emb = real_emb[torch.randperm(real_emb.shape[0])] #Shuffle (dataloader was not shuffled)
        real = real_emb[0:real_set_size].numpy()

        mu_real, sigma_real = np.mean(real, axis = 0), np.cov(real, rowvar=False)
        mu_gen, sigma_gen = np.mean(generated_emb.numpy(), axis = 0), np.cov(generated_emb.numpy(), rowvar=False)

        list_FED.append(calc_FED(mu_real, sigma_real, mu_gen, sigma_gen))
        stop = time.time()
        list_times.append(stop-start)

    return sum(list_FED)/len(list_FED), np.std(list_FED)