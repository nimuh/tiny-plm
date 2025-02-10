import torch
import torch.nn as nn
import torch.optim as optim
from tiny_plm.model import PLM
from tiny_plm.config import PLMConfig
from tiny_plm.util import ProteinTokenizer, create_protein_batches, pad_batch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import datasets
import matplotlib.pyplot as plt



# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DATA_PATH = "data/train_set_at_10_idx_conserved.csv"
MASK_RATIO = 0.15
TRAIN = True
NLAYER = 3
NHEAD = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, tokenizer, optimizer, criterion, batch_idxs, df_seqs):
    model.train()
    total_loss = 0
    batch_losses = []

    progress_bar = tqdm(batch_idxs, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        # Prepare batch
        ko_ids = df_seqs.iloc[batch].KO.tolist()
        aa_seqs = df_seqs.iloc[batch].sequence.tolist()

        # Tokenize KO IDs and amino acid sequences
        tokenized_ko = [tokenizer.encode(ko, is_ko=True) for ko in ko_ids]
        tokenized_aa = [tokenizer.encode(aa) for aa in aa_seqs] + [
            tokenizer.encode("<EOS>")
        ]
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
        target_seqs = padded_seqs[
            :, 1:
        ]  # Use all tokens except the first one as target

        # Forward pass
        logits = model(input_seqs.to(device))

        # Calculate loss (ignore KO ID tokens in loss computation)
        loss = 0
        for i, (logit, target) in enumerate(zip(logits, target_seqs)):
            # ko_length = len(tokenized_ko[i])
            loss += criterion(logit, target.to(device))
        loss /= len(batch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)

        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{batch_loss:.4f}"})

    

    return total_loss / len(batch_idxs), batch_losses


# Generate example sequence
# TODO: Add topk to generate_sequence for more diverse outputs
# TODO: Add temperature to generate_sequence for more diverse outputs
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
        initial_tokens = (
            tokenizer.encode(initial_ko, is_ko=True).unsqueeze(0).to(device)
        )
        for _ in range(max_length):
            logits = trained_model(initial_tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            if next_token == tokenizer.AA_to_idx["<EOS>"]:
                break
            initial_tokens = torch.cat([initial_tokens, next_token], dim=1)
    return tokenizer.decode(initial_tokens[0][6:])


def generate_and_save_sequences(
    df, config, tokenizer, model_path, output_file, max_length=300
):
    """
    Generate sequences for each KO in the dataset and save them to a FASTA file.

    Args:
    df (pd.DataFrame): DataFrame containing KO IDs.
    config (PLMConfig): Configuration for the model.
    tokenizer (ProteinTokenizer): Tokenizer for encoding/decoding sequences.
    model_path (str): Path to the trained model.
    output_file (str): Path to save the output FASTA file.
    max_length (int): Maximum length of generated sequences.
    """
    model = PLM(config=config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    with open(output_file, "w") as f:
        progress_bar = tqdm(
            df["KO"].unique(), desc="Generating sequences", unit="sequence"
        )
        for ko in progress_bar:
            generated_seq = generate_sequence(
                ko,
                config,
                tokenizer,
                model_path=None,
                model=model,
                max_length=max_length,
            )
            f.write(f">{ko}\n{generated_seq}\n")
            progress_bar.set_postfix({"KO": ko})

    print(f"Generated sequences saved to {output_file}")

def print_dataset(df):
    print("###########################################")
    print(df)
    print('###########################################')


def main():

    #ds = datasets.load_dataset('tattabio/OMG', streaming=True)['train']
    #print(next(iter(ds)))
    #print(gget.search(["115413"], "homo_sapiens"))

    # scrape JGI/MGNify

    
    
    # Load and prepare data
    df_seqs = pd.read_csv(DATA_PATH)
    # Filter df_seqs to only contain samples with sequences that are less than 1024 in length
    df_seqs = df_seqs[df_seqs["sequence"].str.len() < 1024].reset_index(drop=True)

    print_dataset(df_seqs)

    batch_idxs = create_protein_batches(df_seqs.shape[0], batch_size=BATCH_SIZE)

    # Initialize model, tokenizer, and optimizer
    config = PLMConfig(
        n_head=NHEAD, n_layer=NLAYER, vocab_size=len(set(df_seqs.KO)) + 23
    )  # might need to change this
    model = PLM(config=config)

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    tokenizer = ProteinTokenizer(kegg_df=df_seqs)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.AA_to_idx["<PAD>"])

    model.to(device)

    epoch_batch_losses = []
    if TRAIN:
        # Training loop
        for epoch in range(EPOCHS):
            loss, batch_losses = train(model, tokenizer, optimizer, criterion, batch_idxs, df_seqs)
            epoch_batch_losses.extend(batch_losses)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

            if (epoch + 1) % 20 == 0:
                torch.save(
                    model.state_dict(), 
                    f"trained_plm_model_prok_{NLAYER}layers_{NHEAD}heads_E{epoch+1}.pth"
                )
                
        torch.save(
            model.state_dict(),
            f"trained_plm_model_prok_{NLAYER}layers_{NHEAD}heads_E{EPOCHS}.pth",
        )

        # Plot batch losses with smoothing
        plt.figure(figsize=(10, 5))
        # Plot original values
        plt.plot(batch_losses, alpha=0.3, label='Original')
        # Plot smoothed values using moving average
        window_size = 50
        smoothed_losses = pd.Series(batch_losses).rolling(window=window_size, min_periods=1).mean()
        plt.plot(smoothed_losses, linewidth=2, label='Smoothed')
        plt.title(f'Batch Losses During Training - {NLAYER} layers {NHEAD} heads {EPOCHS} epochs')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('batch_losses.png')
        plt.close()

    generate_and_save_sequences(
        df_seqs,
        config,
        tokenizer,
        f"trained_plm_model_{NLAYER}layers_{NHEAD}heads_E{EPOCHS}.pth",
        "generated_sequences.fasta",
        max_length=100,
    )

    

if __name__ == "__main__":
    main()
