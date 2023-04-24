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

#avg_emb_esm2_t33_650M_UR50D = torch.load("Thesis-GenModProtDesign/evalpgm/FED/avg_per_seq_emb_esm2_t33_650M_UR50D_subsample0_01.t") # Average FED over 50 cycles: 0.277, stdev: 0.007
#avg_emb_esm2_t33_650M_UR50D = torch.load("Thesis-GenModProtDesign/evalpgm/FED/avg_per_seq_emb_esm2_t33_650M_UR50D_subsample0_25.t") # Average FED over 50 cycles: 0.012, stdev = 0.000
avg_emb_esm2_t33_650M_UR50D = torch.load("Thesis-GenModProtDesign/evalpgm/FED/COMPLETE_avg_per_seq_emb_esm2_t33_650M_UR50D.t")

# with open('FED_650M_200k_50_cycles.txt', 'a') as f:
#     f.write(f"FED calculation using esm2_t33_650M_UR50D embeddings on full test data split as 163474 'generated' samples and 653896 'real' samples. \n")
#     f.flush()

cycles = 10

real_set_size = 200000
gen_set_sizes = [10, 50, 500, 5000, 50000, 200000]
factors = [1, 2, 0.5]
ratios = [size/real_set_size for size in gen_set_sizes]

for factor in factors:
    list_FED_ratios = []
    list_FED_ratios_stdevs = []

    times_ratios = []
    times_ratios_stdevs = []
    for index_real_set_size, size_gen_set in enumerate(gen_set_sizes):
        list_FED_iter = []
        list_times_iter = []

        for i in range(cycles): #cycle over random seed for better reproducibility
            start = time.time()
            avg_emb_esm2_t33_650M_UR50D = avg_emb_esm2_t33_650M_UR50D[torch.randperm(avg_emb_esm2_t33_650M_UR50D.shape[0])] #Shuffle (dataloader was not shuffled)
            gen = avg_emb_esm2_t33_650M_UR50D[0:int(size_gen_set*factor)].numpy()
            real = avg_emb_esm2_t33_650M_UR50D[int(size_gen_set*factor):int(size_gen_set*factor)+real_set_size[index_real_set_size]].numpy()

            #assert avg_emb_esm2_t33_650M_UR50D.shape[0] == np.concatenate((gen, real), axis=0).shape[0], "Missing a sample?"

            mu_real, sigma_real = np.mean(real, axis = 0), np.cov(real, rowvar=False)
            mu_gen, sigma_gen = np.mean(gen, axis = 0), np.cov(gen, rowvar=False)

            list_FED_iter.append(calc_FED(mu_real, sigma_real, mu_gen, sigma_gen))
            stop = time.time()
            list_times_iter.append(stop-start)

            # with open('FED_650M_testing_ratios_10_cycles.txt', 'a') as f:
            #     f.write(f"Cycle {i}: \t Fréchet ESM Distance = {calc_FED(mu_real, sigma_real, mu_gen, sigma_gen)} \n")
            #     f.flush()

    #print("Average FED over", cycles, "cycles:", np.round(sum(list_FED)/len(list_FED), decimals=3), " \t stdev:", np.round(np.std(list_FED), decimals=3), " \t Time for", cycles, " cycles:", sum(times), "seconds. \t Time per cycle:", np.round(sum(times)/len(times), decimals=3), "seconds. \t stdev:", np.round(np.std(times), decimals=3))

        with open('Thesis-GenModProtDesign/evalpgm/FED/repeats_FED_650M_testing_ratios_10_cycles.txt', 'a') as f:
            f.write(f"Average FED over {cycles} cycles: {np.round(sum(list_FED_iter)/len(list_FED_iter), decimals=3)} \n")
            f.write(f"Standard deviation of FED over {cycles} cycles: {np.std(list_FED_iter)} \t rounded: {np.round(np.std(list_FED_iter), decimals=3)} \n")
            f.write(f"Time elapsed for {cycles} cycles: {sum(list_times_iter)} seconds. \t Time per cycle: {np.round(sum(list_times_iter)/len(list_times_iter), decimals=3)} seconds. \t stdev: {np.round(np.std(list_times_iter), decimals=3)} \n")
            f.flush()
        list_FED_ratios.append(np.round(sum(list_FED_iter)/len(list_FED_iter), decimals=3))
        list_FED_ratios_stdevs.append(np.round(np.std(list_FED_iter), decimals=3))
        times_ratios.append(np.round(sum(list_times_iter)/len(list_times_iter), decimals=3))
        times_ratios_stdevs.append(np.round(np.std(list_times_iter), decimals=3))

    for i in range(len(ratios)):
        with open('Thesis-GenModProtDesign/evalpgm/FED/repeats_FED_650M_testing_ratios_10_cycles.txt', 'a') as f:
            f.write(f"Factor: {factor} \t Ratio: {ratios[i]} \t avg FED: {list_FED_ratios[i]} \t stdev FED: {list_FED_ratios_stdevs[i]} \t Time required: {times_ratios[i]} \t stdev time: {times_ratios_stdevs[i]} \n")
            f.flush()