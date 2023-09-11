# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import time
import random
import argparse
import numpy as np

import torch
import sys
import itertools

from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
import datetime


########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic



########################################################################
# model


def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


########################################################################
# sample


def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):

    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        return tokenizer.decode_batch(as_lists(tokens_batch))


def truncate(sample, terminals):
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample


def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)


tok_to_idx_esm2_150M = {
    '<cls>': 0, '1': 1, '<eos>': 2, '<unk>': 3,
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10,
    'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17,
    'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24,
    'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30,
    '<null_1>': 31, '<mask>': 32
} # following the tokenization of the ESM models.

idx_to_tok_esm2_150M = {v : k for k,v in tok_to_idx_esm2_150M.items()}


########################################################################
# main


def main():

    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B
    
    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-large')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    parser.add_argument('--sanity', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    ckpt = f'./checkpoints/{args.model}'

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # (3) load

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

    # (4) sanity

    if args.sanity:

        with print_time('sanity cross-entropy'):

            def ce(tokens):
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                        logits = model(target, labels=target).logits

                        # shift
                        logits = logits[:-1, ...]
                        target = target[1:]

                        return cross_entropy(logits=logits, target=target).item()

            x_uniref90bfd30 = '2GFLPFRGADEGLAAREAATLAARGTAARAYREDSWAVPVPRGLLGDLTARVAALGAASPPPADPLAVTLDLHHVTAEVALTTVLDAATLVHGQTRVLSAEDAAEAATAAAAATEAYLERLQDFVLFMSASVRVWRRGNAAGATGPEWDQWYTVADRDALGSAPTHLAVLGRQADALCHFVLDRVAWGTCGTPLWSGDEDLGNVVATFAGYADRLATAPRDLIM1'
            x_oas = '1EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPWKGLEYVSAISSNGGSTYYANSVKGRFTISRDNSKNTLYLQMGSLRAEDMAVYYCARDESGYSYGWGYYFDYWGQGTLVTVSS2'
            x_bfd90 = '1TAPRSTRASGSEGSRPPGIPAKGRRCLPSRAGSVTPRFRHARQGTATVAKEQGRKLIASNRKARHDYHIEDTFEAGLVLTGTEVKSLRMGRASLIDGYAVFYGEELWLEGVHIPEYLNGNWTNHTPRRRRKLLLNRSELTKLAHKTSESGHTIVPLALYFKDGRAKVEIAVAKGKKAYDKRHALRERQDQREV2'

            checkpoint_x_ce = {
                'progen2-small': (x_uniref90bfd30, 2.4),
                'progen2-medium': (x_uniref90bfd30, 1.9),
                'progen2-base': (x_uniref90bfd30, 1.9),
                'progen2-large': (x_uniref90bfd30, 1.8),
                'progen2-xlarge': (x_uniref90bfd30, 1.0),
                'progen2-oas': (x_oas, 0.3),
                'progen2-BFD90': (x_bfd90, 1.3),
            }

            ce_eval = ce(checkpoint_x_ce[args.model][0])
            ce_target = checkpoint_x_ce[args.model][1]

            print(ce_target, ce_eval, abs(ce_eval - ce_target))

            assert abs(ce_eval - ce_target) < 0.1

    # (5) sample
    time_batch = []
    start_global = time.time()

    with print_time('sampling'):
        sequences = []
        truncations_list = []

        gen_per_loop = 10
        loops = args.num_samples / gen_per_loop
        print("loops", loops)

        for i in range(0, int(loops)):
            time_start = time.time()
            completions = sample(device=device, model=model, tokenizer=tokenizer, 
                                context=args.context, pad_token_id=tokenizer.encode('<|pad|>').ids[0], 
                                num_return_sequences=gen_per_loop, temp=args.t, top_p=args.p, 
                                max_length=args.max_length + 1)
            time_stop = time.time()
            time_batch.append(time_stop-time_start)
            print((i+1)*gen_per_loop, "protein sequences generated. \t Time elapsed", 
                  round(sum(time_batch), 2), "seconds. \t Seconds per sequence", round(sum(time_batch)/((i+1)*gen_per_loop), 2), "of length", args.max_length)
            sys.stdout.flush()
            truncations = [truncate(completion, terminals=['1', '2']) for completion in completions]
            truncations_list.append(truncations)

        for (i, truncation) in enumerate(list(itertools.chain(*truncations_list))):
            seq_tokens = truncation[1:len(truncation)]
            sequences.append(seq_tokens)

        print("rng seed", args.rng_seed)
        stop_global = time.time()


        generation_time_per_seq = sum(time_batch)/args.num_samples
        print("generation time per seq before correction", round(generation_time_per_seq, 3), "seconds. \t stdev", round((np.std(time_batch)/gen_per_loop), 3))
        sys.stdout.flush()

        torch.save(sequences, f"{args.output_path}/RAW_gen_{args.num_samples}_sequences_length_{args.max_length}_seed{args.rng_seed}.t")
        print("Saved", len(sequences), "generated sequences of length", args.max_length, "succesfully in ", f"{args.output_path}/RAW_gen_{args.num_samples}_sequences_length_{args.max_length}_seed{args.rng_seed}.t")
        sys.stdout.flush()

        torch.save(time_batch, f"{args.output_path}/times_for_{args.num_samples}_sequences_length_{args.max_length}_seed{args.rng_seed}.t")

        print("Saved the times required to calculate each batch of", gen_per_loop, "sequences!")
        print(f"{args.num_samples} took {datetime.timedelta(seconds=stop_global-start_global)}")
        sys.stdout.flush()

            


if __name__ == '__main__':
    main()
    print('done.')
