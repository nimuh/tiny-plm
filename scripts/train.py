import torch
import torch.nn as nn
import torch.optim as optim
from tiny_plm.model import PLM
from tiny_plm.config import PLMConfig
from tiny_plm.util import ProteinTokenizer, create_protein_batches, pad_batch
import pandas as pd
import torch.nn.functional as F

# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DATA_PATH = "data/test_set_at_10_idx_conserved.csv"
MASK_RATIO = 0.15
TRAIN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(model, tokenizer, optimizer, criterion, batch_idxs, df_seqs):
    model.train()
    total_loss = 0

    # Add tqdm progress bar
    from tqdm import tqdm

    progress_bar = tqdm(batch_idxs, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        # Prepare batch
        ko_ids = df_seqs.iloc[batch].KO.tolist()
        aa_seqs = df_seqs.iloc[batch].sequence.tolist()

        # Tokenize KO IDs and amino acid sequences
        tokenized_ko = [tokenizer.encode(ko, is_ko=True) for ko in ko_ids]
        tokenized_aa = [tokenizer.encode(aa) for aa in aa_seqs]
        # print(tokenized_aa)

        # Combine KO ID (as prompt) with amino acid sequence
        combined_seqs = [
            torch.cat([ko, aa]) for ko, aa in zip(tokenized_ko, tokenized_aa)
        ]

        # Pad sequences
        padded_seqs, _ = pad_batch(
            combined_seqs, combined_seqs, max(len(seq) for seq in combined_seqs)
        )

        # Prepare input (KO ID + masked AA) and target (full sequence)
        input_seqs = padded_seqs.clone()
        target_seqs = padded_seqs.clone()

        # Prepare input for causal language modeling
        input_seqs = padded_seqs[:, :-1]  # Use all tokens except the last one as input
        target_seqs = padded_seqs[:, 1:]  # Use all tokens except the first one as target

        # Forward pass
        logits = model(input_seqs.to(device))

        # Calculate loss (ignore KO ID tokens in loss computation)
        loss = 0
        for i, (logit, target) in enumerate(zip(logits, target_seqs)):
            #ko_length = len(tokenized_ko[i])
            loss += criterion(logit, target.to(device))
        loss /= len(batch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return total_loss / len(batch_idxs)


# Generate example sequence
# TODO: Add topk to generate_sequence for more diverse outputs
def generate_sequence(
    initial_ko, config, tokenizer, model_path, model=None, max_length=300
):
    if model_path:
        trained_model = PLM(config=config)
        trained_model.load_state_dict(torch.load(model_path))
        trained_model.eval()
    else:
        trained_model = model
    with torch.no_grad():
        initial_tokens = tokenizer.encode(initial_ko).unsqueeze(0).to(device)
        print(initial_tokens)
        for _ in range(max_length):
            logits = trained_model(initial_tokens)
            print(logits.shape)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            initial_tokens = torch.cat([initial_tokens, next_token], dim=1)
    return tokenizer.decode(initial_tokens[0][6:])


def main():
    # Load and prepare data
    df_seqs = pd.read_csv(DATA_PATH)
    # Filter df_seqs to only contain samples with sequences that are less than 1024 in length
    df_seqs = df_seqs[df_seqs["sequence"].str.len() < 1024].reset_index(drop=True)

    batch_idxs = create_protein_batches(df_seqs.shape[0], batch_size=BATCH_SIZE)

    # Initialize model, tokenizer, and optimizer
    config = PLMConfig(n_head=8, n_layer=2, vocab_size=len(set(df_seqs.KO)) + 23)
    model = PLM(config=config)

    tokenizer = ProteinTokenizer(kegg_df=df_seqs)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.AA_to_idx["<PAD>"])

    model.to(device)

    if TRAIN:
        # Training loop
        for epoch in range(EPOCHS):
            loss = train(model, tokenizer, optimizer, criterion, batch_idxs, df_seqs)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

        torch.save(model.state_dict(), "trained_plm_model.pth")

    # Example usage
    initial_ko = "K14331"
    generated_seq = generate_sequence(
        initial_ko, config, tokenizer, model_path="trained_plm_model.pth"
    )  # , model=model)
    print(f"Generated sequence for {initial_ko}: {generated_seq}")


if __name__ == "__main__":
    main()
