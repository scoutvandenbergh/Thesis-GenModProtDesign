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

## How to download and process Pfam
- [Download link](https://github.com/songlab-cal/tape#raw-data) (download Raw Pre-training corpus)
- Unpack using `tar -zvxf *file*`
- Run `python evalpgm/process_TAPE.py *pfamfolder*`

## Train a VAE
- Run `python evalpgm/process_TAPE.py *pfamfolder* *logfolder*`
- E.g. `nohup python evalpgm/scripts/train_VAE.py ../pfam/ ../logs_tester/ &`



## TODO list
- [ ] Add a second decoder that decodes one number: the protein length. Optimize its MSE of true protein lengths jointly with the reconstruction loss + KL Div Loss. Also cut off the reconstruction loss up to where the protein ends.
- [ ] Apart from `data.py` and `models.py`, there should be a `gen.py` script that introduces some functions or classes that aid in generating proteins.

