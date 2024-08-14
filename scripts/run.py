from tiny_plm.model import PLM
from tiny_plm.config import PLMConfig
from tiny_plm.util import ProteinTokenizer, protein_batch_create, pad_batch
import pandas as pd
import torch


# TODO
# add cmd args to this script
# write some decoding function for trained model to see what the model is predicting (curious)
# idea: play with higher masking rates and test on kegg dataset, maybe high masking rates could 
# help with better generalization?

DATA = '../data/all_peptides.csv'
M_RATIO = 0.50
epochs = 50

df_seqs = pd.read_csv(DATA, nrows=100)
batch_idxs = protein_batch_create(df_seqs.shape[0])

config = PLMConfig(n_head=8, n_layer=2, vocab_size=22)
model = PLM(config=config)

tokenizer = ProteinTokenizer()
padded_value = 21
toks, _ = tokenizer.encode("L", mask_frac=0.0)
toks = toks.unsqueeze(dim=0)

print(toks)
while toks.size(1) < 50:
    with torch.no_grad():
        logits = model(toks)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        toks = torch.cat((toks, xcol), dim=1)
print(toks)
print(''.join(tokenizer.decode(toks[0])))


#opt = torch.optim.AdamW(model.parameters(), lr=1e-3)


"""
for epoch in range(epochs):
    for batch in batch_idxs:
        seqs = df_seqs.iloc[batch].sequence
        max_size = max([len(seq) for seq in seqs])
        tok_seqs = []
        labels = []

        for seq in seqs:
            tok_masked, label = tokenizer.encode(seq, mask_frac=M_RATIO)
            tok_seqs.append(tok_masked)
            labels.append(label)

        batch_padded, batch_padded_label = pad_batch(tok_seqs, labels, max_size)

        y = model(batch_padded)
        pred = y.view(y.size(0)*y.size(1), -1)
        targets = batch_padded_label.view(batch_padded_label.size(0)*batch_padded_label.size(1))
        loss = torch.nn.functional.cross_entropy(pred, targets, ignore_index=21)

        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"Perplexity: {torch.exp(loss)}  CE Loss: {loss}")
"""