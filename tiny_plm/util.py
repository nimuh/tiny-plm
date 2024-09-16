import torch
import pandas as pd


class ProteinTokenizer:
    def __init__(self, kegg_df):
        ko_ids = kegg_df.KO.dropna().reset_index(drop=True)
        
        # Create a dictionary mapping KO IDs to indices
        self.AA_to_idx = dict(zip(ko_ids.values, ko_ids.index.tolist()))
        
        # Add amino acids and special tokens
        amino_acids = "ARNDCQEGHILKMFPSTWYV"
        special_tokens = ["<MASK>", "<PAD>"]
        
        for i, aa in enumerate(amino_acids):
            self.AA_to_idx[aa] = len(ko_ids) + i
        
        for i, token in enumerate(special_tokens):
            self.AA_to_idx[token] = len(ko_ids) + len(amino_acids) + i
        
        # Create reverse mapping
        self.idx_to_AA = {v: k for k, v in self.AA_to_idx.items()}

    def encode(self, seq):
        tokenized = [self.AA_to_idx[aa] for aa in seq]
        return torch.tensor(tokenized)

    def decode(self, int_seq):
        return ''.join(self.idx_to_AA[idx] for idx in int_seq.tolist())


def create_protein_batches(data_length, batch_size=8):
    sampler = torch.utils.data.RandomSampler(range(data_length))
    return list(torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False))


def pad_batch(tok_seqs, labels, max_size, pad_value=21):
    def pad_sequence(seq):
        return torch.nn.functional.pad(seq, (0, max_size - len(seq)), value=pad_value)

    batch_padded = torch.stack([pad_sequence(seq) for seq in tok_seqs])
    batch_padded_label = torch.stack([pad_sequence(label) for label in labels])

    return batch_padded, batch_padded_label
