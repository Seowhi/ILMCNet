
"""
Created on Wed Sep 23 18:33:22 2020

@author: mheinzinger
"""

import argparse
import os
import re
import time
from pathlib import Path
import config.args as args

import numpy as np
import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# # device = torch.device('cpu')
# print("Using device: {}".format(device))
device = args.device


max_length = 1590

model_path = "model/prot_t5_xl_half_uniref50-enc"

#       python get_prottrans.py -i /data/cullpdb.fasta -o /t5/cullpdb.npz
#       python get_prottrans.py -i /data/CB513.fasta -o /t5/cb513.npz

def find_divisors(n):
    if n % 2 == 0:
        return 2
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return i
    return n

def get_T5_model(model_dir, transformer_link=model_path):
    print("Loading: {}".format(transformer_link))
    if model_dir is not None:
        print("##########################")
        print("Loading cached model from: {}".format(model_dir))
        print("##########################")
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
    # only cast to full-precision if no GPU is available
    if device == torch.device("cpu"):
        print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    return model, vocab


def read_fasta(fasta_path):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single
        sequence, depending on input file.
    '''

    sequences = dict()
    sequence_count = 0
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                sequences[uniprot_id] = ''
                sequence_count += 1
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[uniprot_id] += ''.join(line.split()).upper().replace("-",
                                                                               "")  # drop gaps and cast to upper-case

    return sequences, sequence_count


def get_embeddings(seq_path,
                   emb_path,
                   model_dir,
                   per_protein,  # whether to derive per-protein (mean-pooled) embeddings
                   max_residues=4000,  # number of cumulative residues per batch
                   max_seq_len=max_length,  # max length after which we switch to single-sequence processing to avoid OOM
                   max_batch=5  # max number of sequences per single batch  #一定要设置一个可以除尽的数字，不然有的数据会不进行加载
                   ):


    seq_dict = dict()
    emb_dict = dict()
    emb_dict = {
               "prottrans_embs": dict()
               }

    # Read in fasta
    seq_dict, seq_count = read_fasta(seq_path)
    # max_batch = 1
    max_batch = find_divisors(seq_count)
    model, vocab = get_T5_model(model_dir)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    print('########################################')
    print('Example sequence: {}\n{}'.format(next(iter(
        seq_dict.keys())), next(iter(seq_dict.values()))))
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long = sum([1 for _, seq in seq_dict.items() if len(seq) > max_seq_len])
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(seq_dict[kv[0]]), reverse=True)

    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    start = time.time()
    batch = list()
    # print(seq_dict)
    # exit()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        # if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
        if len(batch) >= max_batch:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, max_length=max_length, padding="max_length")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)


            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={}). Try lowering batch size. ".format(pdb_id, seq_len) +
                      "If single sequence processing does not work, you need more vRAM to process your protein.")
                continue

            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # slice-off padded/special tokens
                # emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                emb = embedding_repr.last_hidden_state[batch_idx, :max_length]
                # print("emb shape:", emb.shape)

                if per_protein:
                    emb = emb.mean(dim=0)

                if len(emb_dict) == 0:
                    print("Embedded protein {} with length {} to emb. of shape: {}".format(
                        identifier, s_len, emb.shape))

                emb_dict["prottrans_embs"][identifier] = emb.detach().cpu().numpy().squeeze()

    end = time.time()

    __save_embeddings(emb_dict["prottrans_embs"], emb_path)

    # with h5py.File(str(emb_path), "w") as hf:
    #     for sequence_id, embedding in emb_dict.items():
    #         # noinspection PyUnboundLocalVariable
    #         hf.create_dataset(sequence_id, data=embedding)

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(emb_dict)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(
        end - start, (end - start) / len(emb_dict), avg_length))
    return True

def __save_embeddings(emb_dict,out_path):
    np.savez_compressed(out_path, **emb_dict)

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=(
            'data/cullpdb/cullpdb.fasta ' +
            'data/cullpdb/output.npz' +
            'Rostlab/prot_t5_xl_uniref50'))

    # Required positional argument
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='A path to a fasta-formatted text file containing protein sequence(s).')

    # Optional positional argument
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='A path for saving the created embeddings as NumPy npz file.')

    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                        default=None,
                        help='A path to a directory holding the checkpoint for a pre-trained model')

    # Optional argument
    parser.add_argument('--per_protein', type=int,
                        default=0,
                        help="Whether to return per-residue embeddings (0: default) or the mean-pooled per-protein representation (1).")
    return parser


def main():


    # parser = create_arg_parser()
    # args = parser.parse_args()
    #
    # seq_path = Path(args.input)
    # emb_path = Path(args.output)
    # model_dir = Path(args.model) if args.model is not None else None
    #
    # per_protein = False if int(args.per_protein) == 0 else True

    seq_path = f'data/NEW364.fasta'
    emb_path = f't5/NEW364.npz'
    # get_embeddings(seq_path, emb_path, model_dir, per_protein=per_protein)
    get_embeddings(seq_path, emb_path, None, per_protein=False)



def get_single_embeddings(seq):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained(model_path).to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    # model.to(torch.float32) if device == torch.device("cpu")

    # prepare your protein sequences as a list
    sequence_examples = [seq]

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
    embedding = embedding_repr.last_hidden_state[0, :len(seq)]  # shape (7 x 1024)
    return embedding


if __name__ == '__main__':
    #       python get_prottrans.py -i /data/cullpdb.fasta -o /t5/cullpdb.npz
    main()



