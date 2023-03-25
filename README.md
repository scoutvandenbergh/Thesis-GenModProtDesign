## Usage

In your environment, navigate to the root folder of this repo and
```bash
pip install -e .
```

Now you have installed anything under "evalpgm" as a package and can use it in any folder.

## Package structure:

- `data` contains datamodules
- `models` contains models
- `scripts/` contains scripts to process data and train models.

## How to download and process Uniref50
- [Download link for Uniref](https://www.uniprot.org/help/downloads) (Uniref50 fasta format, accessed 18 Jan 2023)
- [Download link for pre-training split](https://github.com/facebookresearch/esm#available-pretraining-split) (UniRef50 IDs of evaluation set, accessed 18 Jan 2023)
- unzip both files using `gunzip *file*`
- Run `python evalpgm/process_Uniref.py *path_to_uniref_fasta* *path_to_split_txt*`

This last script will create a HDF5 file `uniref50.h5t` containing the Uniref50 Database.

## Train a VAE
- Run `python evalpgm/scripts/train_VAE.py *uniref_file* *logfolder*`
- E.g. `nohup python evalpgm/scripts/train_VAE.py uniref50.h5t logs/ &`



## TODO list
- [x] Add a second decoder that decodes one number: the protein length. Optimize its MSE of true protein lengths jointly with the reconstruction loss + KL Div Loss. Also cut off the reconstruction loss up to where the protein ends.
- [x] Transformers: apply rotary positional embeddings, gated linear units, memory efficient self attention.
- [ ] FID --> FAD/FED
   - [ ] Average of 50 runs (200k test vs 5-50-500-5k-50k-300k test)
      - Determine how many "generated" samples required to get FED close to 0.
      - Other approaches were:
         - 25k test vs 25k gen
         - 50k random disturbed vs 200k full dataset
         - 160k disturbed test set vs 20k test set
         - 50k CIFAR-10 noised test vs 10k CIFAR-10 test
         - FAD: ratio eval/background = 7h/540h = 0.013 (optimal) or 25 min/540h = 7.7e-4, (10k train vs 10k test) for 50 rounds, average it
         - FCD: approach mentioned above, find # generated required to have FCD = O between test sets
      - Question: Can we pre-compute the 200k test embeddings and just sample the generated embeddings, resulting in using the same 200k test embeddings 50 times and having new randomly sampled generated embeddings 50 times?

   - [ ] esm2_t6_8M_UR50D vs esm2_t12_35M_UR50D vs esm2_t30_150M_UR50D vs esm2_t33_650M_UR50D vs esm2_t36_3B_UR50D vs esm2_t48_15B_UR50D vs esm2_t36_3B_UR50D+ESMFold
      - Track runtime to calculate embeddings averaged per sequence + time to compute FED once or 50x for the optimal gen vs test ratio @150M or @650M ESM2 model
   - [ ] Compare
      - basic VAE
      - beta-VAE last semester
      - beta-VAE this semester
      - beta-VAE this semester + length pred
      - beta-VAE + transformers + RoPE
      - beta-VAE + transformers + other positional embeddings
      - proteinGAN
      - ProGen2
      - ...

- [ ] VAE --> VQ-VAE
- [ ] VQ-VAE --> conditional VQ-VAE
- [ ] FAD/FID --> FJAD/FJED
- [ ] Apart from `data.py` and `models.py`, there should be a `gen.py` script that introduces some functions or classes that aid in generating proteins.
