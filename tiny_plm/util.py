import torch


class ProteinTokenizer():
    def __init__(self):
        self.AA_to_idx = {}
        self.AA_to_idx['A'] = 0
        self.AA_to_idx['R'] = 1
        self.AA_to_idx['N'] = 2
        self.AA_to_idx['D'] = 3
        self.AA_to_idx['C'] = 4
        self.AA_to_idx['Q'] = 5
        self.AA_to_idx['E'] = 6
        self.AA_to_idx['G'] = 7
        self.AA_to_idx['H'] = 8
        self.AA_to_idx['I'] = 9
        self.AA_to_idx['L'] = 10
        self.AA_to_idx['K'] = 11
        self.AA_to_idx['M'] = 12
        self.AA_to_idx['F'] = 13
        self.AA_to_idx['P'] = 14
        self.AA_to_idx['S'] = 15
        self.AA_to_idx['T'] = 16
        self.AA_to_idx['W'] = 17
        self.AA_to_idx['Y'] = 18
        self.AA_to_idx['V'] = 19
        self.AA_to_idx['<MASK>'] = 20
        self.AA_to_idx['<PAD>'] = 21

        self.idx_to_AA = {v: k for k, v in self.AA_to_idx.items()}

    def encode(self, seq, mask_frac=0.15):
        mask_choice = torch.randn((1,len(seq))) < mask_frac
        masked_seq = ['<MASK>' if mask_choice[0][i] else seq[i] for i in range(len(seq))]
        tokenized_masked = [self.AA_to_idx[s] for s in masked_seq]
        label = [self.AA_to_idx[s] for s in seq]
        return torch.tensor(tokenized_masked),torch.tensor(label)
    
    def decode(self, int_seq):
        aa_seq = [self.idx_to_AA[idx] for idx in int_seq.tolist()]
        return aa_seq
    

def protein_batch_create(length_of_data, batch_size=8):
    sampler = torch.utils.data.RandomSampler(range(length_of_data))
    batch_idxs = list(torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False))
    return batch_idxs


