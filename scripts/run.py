from tiny_plm.model import PLM
from tiny_plm.config import PLMConfig
from tiny_plm.util import ProteinTokenizer, create_protein_batches
import pandas as pd
import torch


# TODO
# add cmd args to this script
# write some decoding function for trained model to see what the model is predicting (curious)
# idea: play with higher masking rates and test on kegg dataset, maybe high masking rates could
# help with better generalization?

DATA = "data/test_set_at_10_idx_conserved.csv"
M_RATIO = 0.50
epochs = 50

# Load and prepare data
df_seqs = pd.read_csv(DATA, nrows=100)
batch_idxs = create_protein_batches(df_seqs.shape[0])

# Initialize model and tokenizer
config = PLMConfig(n_head=8, n_layer=2, vocab_size=len(set(df_seqs.KO)) + 20)
model = PLM(config=config)
tokenizer = ProteinTokenizer(kegg_df=df_seqs)


def generate_sequence(model, initial_tokens, max_length):
    with torch.no_grad():
        while initial_tokens.size(1) < max_length:
            logits = model(initial_tokens)
            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            top_k = 10
            top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
            next_token = torch.multinomial(top_k_probs, 1)
            next_token = torch.gather(top_k_indices, -1, next_token)
            initial_tokens = torch.cat((initial_tokens, next_token), dim=1)
    return initial_tokens


# Encode initial sequence
initial_sequence = "K02910"
toks = tokenizer.encode(initial_sequence, is_ko=True)  # , mask_frac=0.0)
toks = toks.unsqueeze(dim=0)

print(toks)

# Generate sequence
max_length = 50
generated_sequence = generate_sequence(model, toks, max_length)

# Decode and print results
gen_seq = tokenizer.decode(generated_sequence[0][1:])
print(f"Initial sequence: {initial_sequence}")
print(f"Generated sequence: {gen_seq}")
