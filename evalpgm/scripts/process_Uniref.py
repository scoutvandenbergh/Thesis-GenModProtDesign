from Bio import SeqIO
from evalpgm.data import tok_to_idx_esm2_150M
import numpy as np
import parse
import sys
import h5torch


path_to_file = str(sys.argv[1])
path_to_split = str(sys.argv[2])

ids = []
clusternames = []
n_members = []
taxons = []
taxon_ids = []
representative_prot_ids = []
seqs = []
seq_lens = []

f = h5torch.File("uniref50.h5t", "w")

with open(path_to_file, mode = "r") as f_:
    for ix, record in enumerate(SeqIO.parse(f_, "fasta")):
        sequence = record.seq
        length = len(sequence)
        id, clustername, n_member, taxon, taxon_id, representative = parse.parse("{} {} n={} Tax={} TaxID={} RepID={}", record.description)
        seq_encoded = np.array([tok_to_idx_esm2_150M[aa] for aa in sequence], dtype='int8')

        ids.append(id)
        clusternames.append(clustername)
        n_members.append(int(n_member))
        taxons.append(taxon)
        taxon_ids.append(taxon_id)
        representative_prot_ids.append(representative)
        seqs.append(seq_encoded)
        seq_lens.append(length)
        if (ix + 10) % 10000 == 0:
            print(ix, flush = True)
            if "central" not in f:
                f.register(seqs, "central", mode = "vlen", dtype_save = "int8", dtype_load = "int64", length = 54_465_398)
            else:
                f.append("central", seqs)
            seqs = []



f.append("central", seqs)
f.register(np.array(seq_lens), axis = 0, name = "protein_length", dtype_save = "int64", dtype_load = "int64")
f.register(np.array(ids).astype(bytes), axis = "unstructured", name = "id", dtype_save = "bytes", dtype_load = "str")

with open(path_to_split) as f_:
    lines = f_.readlines()
valids_ = np.array([k[1:].rstrip() for k in lines], dtype="bytes")

indices = np.isin(f["unstructured/id"][:], valids_)
indices = np.where(indices)[0]

val_indices, test_indices = np.split(np.random.permutation(indices), [indices.shape[0] //2])
k = np.full(f["central"].shape, "train", dtype="object")
k[val_indices] = "val"
k[test_indices] = "test"
k = k.astype(bytes)
f.register(k, axis = "unstructured", name = "split", dtype_save = "bytes", dtype_load = "str")
f.close()