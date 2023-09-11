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
- Run `python evalpgm/scripts/train_VAE.py *uniref_file* *logfolder* *length_decoder_boolean*`
- E.g. `nohup python evalpgm/scripts/train_VAE.py uniref50.h5t logs/ True &`
