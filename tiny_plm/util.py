import torch
import pandas as pd


class ProteinTokenizer():
    def __init__(self, kegg_df):
        #kegg_df = pd.read_csv('../data/prok.csv', nrows=100)
        ko_ids = kegg_df.KO.dropna().reset_index(drop=True)
        print(ko_ids.index[:20].tolist())
        self.AA_to_idx = dict(zip(ko_ids.values, ko_ids.index.tolist()))
        self.AA_to_idx['A'] = len(ko_ids)
        self.AA_to_idx['R'] = 1 + len(ko_ids)
        self.AA_to_idx['N'] = 2 + len (ko_ids)
        self.AA_to_idx['D'] = 3 + len(ko_ids)
        self.AA_to_idx['C'] = 4 + len(ko_ids)
        self.AA_to_idx['Q'] = 5 + len(ko_ids)
        self.AA_to_idx['E'] = 6 + len(ko_ids)
        self.AA_to_idx['G'] = 7 + len(ko_ids)
        self.AA_to_idx['H'] = 8 + len(ko_ids)
        self.AA_to_idx['I'] = 9 + len(ko_ids)
        self.AA_to_idx['L'] = 10 + len(ko_ids)
        self.AA_to_idx['K'] = 11 + len(ko_ids)
        self.AA_to_idx['M'] = 12 + len(ko_ids)
        self.AA_to_idx['F'] = 13 + len(ko_ids)
        self.AA_to_idx['P'] = 14 + len(ko_ids)
        self.AA_to_idx['S'] = 15 + len(ko_ids)
        self.AA_to_idx['T'] = 16 + len(ko_ids)
        self.AA_to_idx['W'] = 17 + len(ko_ids)
        self.AA_to_idx['Y'] = 18 + len(ko_ids)
        self.AA_to_idx['V'] = 19 + len(ko_ids)
        self.AA_to_idx['<MASK>'] = 20 + len(ko_ids)
        self.AA_to_idx['<PAD>'] = 21 + len(ko_ids)
        print(self.AA_to_idx)
        self.idx_to_AA = {v: k for k, v in self.AA_to_idx.items()}
        print(self.idx_to_AA)

    def encode(self, seq, mask_frac=0.15):
        #mask_choice = torch.randn((1,len(seq))) < mask_frac
        #masked_seq = ['<MASK>' if mask_choice[0][i] else seq[i] for i in range(len(seq))]
        #print(masked_seq)
        tokenized_masked = [self.AA_to_idx[seq]]
        #label = [self.AA_to_idx[s] for s in seq]
        return torch.tensor(tokenized_masked), None#torch.tensor(label)
    
    def decode(self, int_seq):
        aa_seq = [self.idx_to_AA[idx] for idx in int_seq.tolist()]
        return aa_seq
    

def protein_batch_create(length_of_data, batch_size=8):
    sampler = torch.utils.data.RandomSampler(range(length_of_data))
    batch_idxs = list(torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False))
    return batch_idxs

def pad_batch(tok_seqs, labels, max_size, pad_value=21):
    padded = [torch.cat([item, torch.tensor([pad_value]).expand(max_size-len(item))]) for item in tok_seqs]
    batch_padded = torch.cat([item[None] for item in padded])
    padded_label = [torch.cat([item, torch.tensor([pad_value]).expand(max_size-len(item))]) for item in labels]
    batch_padded_label = torch.cat([item[None] for item in padded_label])
    return batch_padded, batch_padded_label


