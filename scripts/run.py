from tiny_plm.model import PLM
from tiny_plm.config import PLMConfig
from tiny_plm.util import ProteinTokenizer, protein_batch_create
import pandas as pd
import torch


# TODO
# add cmd args to this script

DATA = '../data/all_peptides.csv'
epochs = 500

df_seqs = pd.read_csv(DATA, nrows=100)
batch_idxs = protein_batch_create(df_seqs.shape[0])

config = PLMConfig(n_head=8, n_layer=2, vocab_size=22)
model = PLM(config=config)

tokenizer = ProteinTokenizer()
padded_value = 21
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for batch in batch_idxs:
        seqs = df_seqs.iloc[batch].sequence
        max_size = max([len(seq) for seq in seqs])
        tok_seqs = []
        labels = []

        for seq in seqs:
            tok_masked, label = tokenizer.encode(seq)
            tok_seqs.append(tok_masked)
            labels.append(label)

        padded = [torch.cat([item, torch.tensor([padded_value]).expand(max_size-len(item))]) for item in tok_seqs]
        batch_padded = torch.cat([item[None] for item in padded])

        padded_label = [torch.cat([item, torch.tensor([padded_value]).expand(max_size-len(item))]) for item in labels]
        batch_padded_label = torch.cat([item[None] for item in padded_label])

        y = model(batch_padded)
        pred = y.view(y.size(0)*y.size(1), -1)
        targets = batch_padded_label.view(batch_padded_label.size(0)*batch_padded_label.size(1))
        loss = torch.nn.functional.cross_entropy(pred, targets, ignore_index=21)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"Perplexity: {torch.exp(loss)}  CE Loss: {loss}")
