import torch
import pandas as pd


class ProteinTokenizer:
    def __init__(self, kegg_df):
        # Add amino acids first
        amino_acids = "ARNDCQEGHILKMFPSTWYV"
        self.AA_to_idx = {}
        for i, aa in enumerate(amino_acids):
            self.AA_to_idx[aa] = i
        
        # Add KO IDs
        ko_ids = kegg_df.KO.dropna().reset_index(drop=True).unique()
        start_idx = len(amino_acids) - 1
        for i, ko_id in enumerate(ko_ids):
            self.AA_to_idx[ko_id] = start_idx + i
        
        # Add special tokens
        special_tokens = ["<MASK>", "<PAD>"]
        start_idx += len(ko_ids)
        for i, token in enumerate(special_tokens):
            self.AA_to_idx[token] = start_idx + i
        
        # Create reverse mapping
        self.idx_to_AA = {v: k for k, v in self.AA_to_idx.items()}

    def encode(self, seq):
        #print(seq)
        #print(self.AA_to_idx)
        tokenized = [self.AA_to_idx[seq]] # for aa in seq]
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


def load_mhc_class_1_data():
    from tdc.multi_pred import PeptideMHC
    
    # Load MHC Class I dataset
    data = PeptideMHC(name='MHC1_IEDB-IMGT_Nielsen')
    
    # Split the data
    split = data.get_split()
    train_data = split['train']
    valid_data = split['valid']
    test_data = split['test']
    
    return train_data, valid_data, test_data

class MHCDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        peptide = item['Peptide']
        mhc = item['MHC']
        label = item['Y']
        
        # Tokenize peptide and MHC
        peptide_tokens = self.tokenizer.encode(peptide)
        mhc_tokens = self.tokenizer.encode(mhc)
        
        # Combine peptide and MHC tokens
        combined_tokens = torch.cat([peptide_tokens, mhc_tokens])
        
        return combined_tokens, torch.tensor(label, dtype=torch.float)

def create_mhc_dataloader(data, tokenizer, batch_size=32):
    dataset = MHCDataset(data, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: pad_batch(*zip(*batch), max_size=max(len(x[0]) for x in batch))
    )
    return dataloader

def test_mhc_dataset():
    # Load the data
    train_data, _, _ = load_mhc_class_1_data()
    
    # Create a dummy tokenizer (replace with your actual tokenizer)
    from transformers import EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # Create the dataset
    dataset = MHCDataset(train_data, tokenizer)
    
    print(train_data)
    print(f"Dataset size: {len(dataset)}")
    
    # Iterate over the first 3 samples
    for i in range(3):
        tokens, label = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"Tokens: {tokens}")
        print(f"Label: {label}")
        print(f"Tokens shape: {tokens.shape}")
        print(f"Label shape: {label.shape}")

