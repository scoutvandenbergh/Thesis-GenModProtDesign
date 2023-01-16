# first download and unpack TAPE

import json
import numpy as np
import os
import h5torch
import sys

root_folder = str(sys.argv[1])


tok_to_idx_esm2_150M = {
    '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10,
    'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17,
    'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24,
    'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30,
    '<null_1>': 31, '<mask>': 32
} # following the tokenization of the ESM models.

for split in ["holdout", "valid", "train"]:

    d = json.load(open(os.path.join(root_folder, "pfam_" + split + ".json")))

    for i in d:
        i["tokenized"] = np.array([tok_to_idx_esm2_150M[aa] for aa in i["primary"]], dtype='int8')

    f = h5torch.File(os.path.join(root_folder, split+".h5t"), 'w')
    f.register([i["tokenized"] for i in d], "central", mode = "vlen", dtype_save = "int8", dtype_load = "int64")
    f.register(np.array([i["protein_length"] for i in d]), axis = 0, name = "protein_length", dtype_save = "int64", dtype_load = "int64")
    f.register(np.array([i["clan"] for i in d]), axis = 0, name = "clan", dtype_save = "int64", dtype_load = "int64")
    f.register(np.array([i["family"] for i in d]), axis = 0, name = "family", dtype_save = "int64", dtype_load = "int64")
    f.register(np.array([int(i["id"]) for i in d]), axis = 0, name = "id", dtype_save = "int64", dtype_load = "int64")
    f.close()