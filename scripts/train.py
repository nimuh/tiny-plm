import torch
import torch.nn as nn
import torch.optim as optim
from tiny_plm.model import PLM
from tiny_plm.config import PLMConfig
from tiny_plm.util import ProteinTokenizer, create_protein_batches, pad_batch
import pandas as pd

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DATA_PATH = "data/test_set_at_10_idx_conserved.csv"
MASK_RATIO = 0.15

# Load and prepare data
df_seqs = pd.read_csv(DATA_PATH)
batch_idxs = create_protein_batches(df_seqs.shape[0], batch_size=BATCH_SIZE)

# Initialize model, tokenizer, and optimizer
config = PLMConfig(n_head=8, n_layer=2, vocab_size=len(set(df_seqs.KO)) + 20)
model = PLM(config=config)
tokenizer = ProteinTokenizer(kegg_df=df_seqs)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.AA_to_idx["<PAD>"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def mask_sequence(seq, mask_ratio):
    mask = torch.rand(seq.shape) < mask_ratio
    masked_seq = seq.clone()
    masked_seq[mask] = tokenizer.AA_to_idx["<MASK>"]
    return masked_seq, mask

def train():
    model.train()
    total_loss = 0

    for batch in batch_idxs:
        optimizer.zero_grad()
        
        # Prepare batch
        ko_ids = df_seqs.iloc[batch].KO.tolist()
        aa_seqs = df_seqs.iloc[batch].sequence.tolist()  # Assuming AA column exists for amino acid sequences
        
        # Tokenize KO IDs and amino acid sequences
        tokenized_ko = [tokenizer.encode(ko) for ko in ko_ids]
        tokenized_aa = [[tokenizer.encode(aa) for aa in seq] for seq in aa_seqs]
        print(tokenized_aa)
        
        # Combine KO ID (as prompt) with amino acid sequence
        combined_seqs = [torch.cat([ko, aa]) for ko, aa in zip(tokenized_ko, tokenized_aa)]
        
        # Pad sequences
        padded_seqs, _ = pad_batch(combined_seqs, combined_seqs, max(len(seq) for seq in combined_seqs))
        
        # Prepare input (KO ID + masked AA) and target (full sequence)
        input_seqs = padded_seqs.clone()
        target_seqs = padded_seqs.clone()
        
        # Mask amino acid part of input sequences
        for i, seq in enumerate(input_seqs):
            ko_length = len(tokenized_ko[i])
            aa_part = seq[ko_length:]
            masked_aa, _ = mask_sequence(aa_part.unsqueeze(0), MASK_RATIO)
            input_seqs[i, ko_length:] = masked_aa.squeeze(0)
        
        # Forward pass
        logits = model(input_seqs.to(device))
        
        # Calculate loss (ignore KO ID tokens in loss computation)
        loss = 0
        for i, (logit, target) in enumerate(zip(logits, target_seqs)):
            ko_length = len(tokenized_ko[i])
            loss += criterion(logit[ko_length:], target[ko_length:].to(device))
        loss /= len(batch)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(batch_idxs)

# Training loop
for epoch in range(EPOCHS):
    loss = train()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "trained_plm_model.pth")

# Generate example sequence
def generate_sequence(initial_ko, max_length=50):
    model.eval()
    with torch.no_grad():
        initial_tokens = tokenizer.encode(initial_ko).unsqueeze(0).to(device)
        for _ in range(max_length - 1):
            logits = model(initial_tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            initial_tokens = torch.cat([initial_tokens, next_token], dim=1)
    return tokenizer.decode(initial_tokens[0])

# Example usage
initial_ko = "K14331"
generated_seq = generate_sequence(initial_ko)
print(f"Generated sequence for {initial_ko}: {generated_seq}")
