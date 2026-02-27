
# corrected_transformer_final.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import math
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# ---------------------------
PAD_IDX = 0
special_tokens = [
    '<PAD>', '<ADJLIST_START>', '<ADJLIST_END>',
    '<ORIGIN_START>', '<ORIGIN_END>',
    '<TARGET_START>', '<TARGET_END>',
    '<PATH_START>', '<PATH_END>', '<-->', ';'   # added BOS/EOS
]
coords = [f'({i},{j})' for i in range(6) for j in range(6)]
vocab_list = special_tokens + coords
token2id = {tok: idx for idx, tok in enumerate(vocab_list)}
id2token = {idx: tok for tok, idx in token2id.items()}
vocab_size = len(vocab_list)

MAZE_TYPES = ['forked', 'forkless']
maze2id = {t: i for i, t in enumerate(MAZE_TYPES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------


class MazeDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_tokens = eval(row['input_sequence'])
        tgt_tokens = eval(row['output_path'])
        # Provide BOS and EOS for decoder training
        tgt_tokens = ['<PATH_START>'] + tgt_tokens
        maze_type = maze2id[row['maze_type']]

        src_ids = [token2id[t] for t in src_tokens]
        tgt_ids = [token2id[t] for t in tgt_tokens]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long), torch.tensor(maze_type, dtype=torch.long)


def collate_fn(batch):
    inputs, outputs, maze_types = zip(*batch)
    input_lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX)
    outputs_padded = pad_sequence(outputs, batch_first=True, padding_value=PAD_IDX)
    maze_types_tensor = torch.stack(maze_types)
    return inputs_padded, outputs_padded, maze_types_tensor, input_lengths


# Positional Encoding
class Positional_Encoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


def decoder_future_mask(target_length, device):
    mask = torch.triu(torch.ones((target_length, target_length), device=device), diagonal=1).bool()
    return mask

class TransformerModel(nn.Module):
    def __init__(self, vocab_dim, nheads=8, embed_dim=128, ff_dim=512, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # embedding with padding index
        self.embedding = nn.Embedding(vocab_dim, embed_dim, padding_idx=PAD_IDX)
        self.pos_enc = Positional_Encoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nheads,
                                                   dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nheads,
                                                   dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.num_layers)

        self.output = nn.Linear(embed_dim, vocab_dim)

        for parm in self.parameters():
            if parm.dim() > 1:
                nn.init.xavier_uniform_(parm)

    def forward(self, source_tokens, target_tokens):
        """
        source_tokens: (B, S)
        target_tokens: (B, T)  (must include BOS at index 0)
        returns logits: (B, T, V)
        """
        device = source_tokens.device
        padding_in_src = (source_tokens == PAD_IDX)  # (B, S)
        padding_in_tgt = (target_tokens == PAD_IDX)  # (B, T)

        src_emb = self.embedding(source_tokens)*math.sqrt(self.embed_dim)
        src_emb = self.pos_enc(src_emb)

        tgt_emb = self.embedding(target_tokens)*math.sqrt(self.embed_dim)
        tgt_emb = self.pos_enc(tgt_emb)

        encd_output = self.encoder(src_emb, mask=None, src_key_padding_mask=padding_in_src)

        target_len = target_tokens.size(1)
        decd_mask = decoder_future_mask(target_len, device)
        # decd_mask = bool_mask.float().masked_fill(bool_mask, float('-inf'))

        decd_output = self.decoder(tgt_emb,
                                   memory=encd_output,
                                   tgt_mask=decd_mask,
                                   memory_mask=None,
                                   tgt_key_padding_mask=padding_in_tgt,
                                   memory_key_padding_mask=padding_in_src)
        logits = self.output(decd_output)
        return logits




def create_loaders(train_df, test_df, batch_size=32):
    train_ds = MazeDataset(train_df)
    test_ds = MazeDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    return train_loader, test_loader

# i need to find train token accuracy and sequence accuracy as well
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    total_tokens, correct_tokens = 0,0
    total_seqs, correct_seqs = 0,0
    for source, targ, maze_type,input_lengths in tqdm(loader, desc="Training"):
        source, targ = source.to(device), targ.to(device)
        targ_input = targ[:,:-1].contiguous()
        targ_output = targ[:, 1:].contiguous()
        optimizer.zero_grad()
        prob_outputs = model(source, targ_input)  # (B, T-1, V)
        V = prob_outputs.size(-1)
        logits_flat = prob_outputs.view(-1, V)
        targ_flat = targ_output.view(-1)
        loss = criterion(logits_flat, targ_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

        # token accuracy 
        preds = prob_outputs.argmax(dim=-1)
        mask = (targ_output != PAD_IDX)
        correct_tkn_cnt = (preds[mask] == targ_output[mask]).sum().item()
        correct_tokens += correct_tkn_cnt
        total_tokens += mask.sum().item()

        # sequence exact-match
        batch_size = preds.size(0)
        correct_seq_count = 0
        for b in range(batch_size):
            pred_b = preds[b][mask[b]]
            true_b = targ_output[b][mask[b]]
            if pred_b.numel() == 0:
                match = False
            else:
                match = torch.all(pred_b == true_b).item()
            if match: 
                correct_seq_count += 1
        correct_seqs += correct_seq_count
        total_seqs += batch_size

    train_loss = epoch_loss/len(loader)
    train_token_acc = correct_tokens/total_tokens if total_tokens > 0 else 0.0
    train_seq_acc = correct_seqs/total_seqs if total_seqs > 0 else 0.0

    return train_loss,train_token_acc,train_seq_acc


@torch.no_grad()
# here actually loss and accuracy (both token and sequence)
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens, correct_tokens = 0, 0
    total_seqs, correct_seqs = 0, 0
    all_preds_flat, all_targ_flat = [], []

    for source, targ, maze_type,input_lengths in tqdm(loader, desc="Evaluating"):
        source, targ = source.to(device), targ.to(device)
        targ_input = targ[:, :-1].contiguous()
        targ_output = targ[:, 1:].contiguous()

        prob_outputs = model(source, targ_input)  # (B, T-1, V)
        V = prob_outputs.size(-1)

        logits_flat = prob_outputs.view(-1, V)
        targ_flat = targ_output.view(-1)
        loss = criterion(logits_flat, targ_flat)
        total_loss += loss.item()

        preds = prob_outputs.argmax(dim=-1)

        # token accuracy
        mask = (targ_output != PAD_IDX)
        correct_tkn_cnt = (preds[mask] == targ_output[mask]).sum().item()
        correct_tokens += correct_tkn_cnt
        total_tokens += mask.sum().item()

        # sequence exact-match
        batch_size = preds.size(0)
        correct_seq_count = 0
        for b in range(batch_size):
            pred_b = preds[b][mask[b]]
            true_b = targ_output[b][mask[b]]
            #i want to predict pre_b and true_b are same or not
            if pred_b.numel() == 0:
                match = False
            else:
                match = torch.all(pred_b == true_b).item()
            if match: 
                correct_seq_count += 1
        correct_seqs += correct_seq_count
        total_seqs += batch_size

        # F1 collection
        true_flat = targ_output[mask].cpu().tolist()
        pred_flat = preds[mask].cpu().tolist()
        all_targ_flat.extend(true_flat)
        all_preds_flat.extend(pred_flat)

    avg_loss = total_loss / len(loader)
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    seq_acc = correct_seqs / total_seqs if total_seqs > 0 else 0.0

    if len(all_targ_flat) > 0:
        preds_arr = np.array(all_preds_flat)
        trgs_arr = np.array(all_targ_flat)
        f1_micro = f1_score(trgs_arr, preds_arr, average='micro', zero_division=0)
        f1_macro = f1_score(trgs_arr, preds_arr, average='macro', zero_division=0)
    else:
        f1_micro = 0.0
        f1_macro = 0.0

    return avg_loss, token_acc, seq_acc, f1_micro, f1_macro


def run_model(train_loader, test_loader, vocab_size, device, epochs=20, lr=1e-4):
    model = TransformerModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "test_loss": [], "test_tok_acc": [], "test_seq_acc": [], "test_f1": [], "train_tok_acc": [], "train_seq_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss,train_token_acc,test_seq_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_tok_acc, test_seq_acc, test_f1_micro, test_f1_macro = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_tok_acc"].append(train_token_acc)
        history["train_seq_acc"].append(test_seq_acc)
        history["test_loss"].append(test_loss)
        history["test_tok_acc"].append(test_tok_acc)
        history["test_seq_acc"].append(test_seq_acc)
        history["test_f1"].append(test_f1_micro)

        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f}  "
            f"test_loss={test_loss:.4f}  "
            f"train_seq_acc={train_token_acc*100:.2f}%  "
            f"test_tok_acc={test_tok_acc*100:.2f}%  "
            f"test_seq_acc={test_seq_acc*100:.2f}%  "
            f"test_f1={test_f1_micro*100:.2f}%"
        )
    return model, history

def plot_history(history):
    epochs = []
    for i in range(len(history['train_loss'])):
        epochs.append(i+1)
    # Plot Loss
    plt.plot(epochs,history['train_loss'], label='Train Loss')
    plt.plot(epochs,history['test_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.savefig('trans_loss_plot.png')
    plt.show()

    # token accuracy and test 
    plt.plot(epochs,history['train_tok_acc'],label='Train Token Accuracy')
    plt.plot(epochs,history['test_tok_acc'],label='Test Token Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Token Accuracy')
    plt.title('Token Accuracy over Epochs')
    plt.legend()
    plt.savefig('trans_token_accuracy_plot.png')
    plt.show()

    # sequence accuracy
    
    plt.plot(epochs, history['train_seq_acc'], 'b--', label='Train Sequence Accuracy')
    plt.plot(epochs, history['test_seq_acc'], 'r-', label='Test Sequence Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Sequence Accuracy')
    plt.title('Sequence Accuracy over Epochs')
    plt.legend()
    plt.savefig('trans_sequence_accuracy_plot.png')
    plt.show()

    


if __name__ == "__main__":
    train_csv = "datasets/train_6x6_mazes.csv"
    test_csv = "datasets/test_6x6_mazes.csv"

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    train_loader, test_loader = create_loaders(df_train, df_test, batch_size=32)

    model, history = run_model(train_loader, test_loader, vocab_size, device, epochs=20, lr=1e-4)
    plot_history(history)
    # save model
    torch.save(model.state_dict(), "transformer_model.pth")
