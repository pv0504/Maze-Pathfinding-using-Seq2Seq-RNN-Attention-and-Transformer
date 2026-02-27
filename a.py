
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
# i need to find f1 score 
import numpy as np
from sklearn.metrics import f1_score
 
# --- Vocabulary / tokens (same as yours) ---
PAD_IDX = 0
special_tokens = [
    '<PAD>', '<ADJLIST_START>', '<ADJLIST_END>',
    '<ORIGIN_START>', '<ORIGIN_END>',
    '<TARGET_START>', '<TARGET_END>',
    '<PATH_START>', '<PATH_END>', '<-->',';'
]
coords = [f'({i},{j})' for i in range(6) for j in range(6)]
vocab_list = special_tokens + coords
token2id = {tok: idx for idx, tok in enumerate(vocab_list)}
id2token = {idx: tok for tok, idx in token2id.items()}
vocab_size = len(vocab_list)

MAZE_TYPES = ['forked', 'forkless']
maze2id = {t: i for i, t in enumerate(MAZE_TYPES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd 
# ----------------- Encoder -----------------
class Encoder(nn.Module): 
    def __init__(self, vocab_size, hidden_size, embed_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=self.num_layers, batch_first=True)

    def forward(self, x, lengths=None):
        """
        x: (B, S) token ids
        lengths: (B,) actual lengths (before padding) or None
        returns:
          outputs: (B, S, H)  -- top-layer hidden at every time step (padded)
          hidden: (L, B, H)   -- final hidden state for each layer
        """
        emb = self.embedding(x)  # (B, S, E)
        if lengths is not None:
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, hidden = self.rnn(packed)
            outputs, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, S, H)
        else:
            outputs, hidden = self.rnn(emb)  # outputs: (B,S,H), hidden: (L,B,H)

        return outputs, hidden

# ----------------- Decoder -----------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        attn_dim = hidden_size // 2
        self.Wa = nn.Linear(hidden_size,attn_dim,bias=False)   
        self.Ua = nn.Linear(hidden_size,attn_dim,bias=False)   
        self.v = nn.Linear(attn_dim,1,bias=False)              # final scoring

        # RNN takes (embed + context) as input
        self.rnn = nn.RNN(embed_size+hidden_size,hidden_size,num_layers=self.num_layers,batch_first=True)

        # output projection
        self.wc = nn.Linear(hidden_size*2,hidden_size,bias=False)
        self.wo = nn.Linear(hidden_size,vocab_size,bias=False)

    def forward(self, prev_hidden, prev_output, encoder_outputs, encoder_mask):
        """
        prev_hidden: (L,B,H)
        prev_output: (B,) token ids of previous predicted/teacher token
        encoder_outputs: (B,S,H)
        encoder_mask: (B,S) -> 1 for valid positions, 0 for PAD
        returns:
          logits: (B, V)
          new_hidden: (Layers,B,H)
        """
        B,S,H = encoder_outputs.size()
        # embedding of previous output
        embed_prev = self.embedding(prev_output)  # (B, E)

        #top layer hidden as query
        s_prev_top = prev_hidden[-1]  # (B, H)
        s_expanded = s_prev_top.unsqueeze(1)   # (B,1,H)

        # attention (Bahdanau)
        proj_s = self.Wa(s_expanded)            # (B,1,attn_dim)
        proj_h = self.Ua(encoder_outputs)       # (B,S,attn_dim)
        energy = torch.tanh(proj_s + proj_h)    # (B,S,attn_dim)
        score = self.v(energy).squeeze(-1)      # (B,S)
        score = score.masked_fill(encoder_mask == 0, -1e9)
        alpha = torch.softmax(score, dim=1)     # (B,S)

        context = (alpha.unsqueeze(2) * encoder_outputs).sum(dim=1)  # (B,H)

        rnn_input = torch.cat([embed_prev,context],dim=-1).unsqueeze(1)  # (B,1,E+H)
        out_seq,new_hidden = self.rnn(rnn_input, prev_hidden)  # out_seq: (B,1,H)
        s_t = out_seq.squeeze(1)  # (B,H)

        # output MLP
        pred_input = torch.cat([context, s_t], dim=-1)  # (B, 2H)
        g = self.wc(pred_input) # (B, H)
        attention_out = torch.tanh(g)
        probs = self.wo(attention_out) # (B, V)

        return probs, new_hidden

# ----------------- Seq2Seq wrapper -----------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, device, num_layers=2, maze_type_size=2, maze_embed_dim=8):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.device = device
        self.num_layers = num_layers
        self.maze_embedding = nn.Embedding(maze_type_size, maze_embed_dim)
        self.proj_maze = nn.Linear(maze_embed_dim, encoder.rnn.hidden_size)

    def forward(self, src_seq, target_seq, maze_type, teacher_ratio=0.5, src_lengths=None):
        """
        src_seq: (B, S_src)
        target_seq: (B, S_tgt)
        maze_type: (B,)
        src_lengths: (B,) lengths before padding (optional)
        """
        B, T_out = target_seq.shape
        V = self.vocab_size

        encoder_outputs, last_hidden = self.encoder(src_seq, lengths=src_lengths)
        encoder_mask = (src_seq != PAD_IDX).long()

        # maze conditioning
        maze_emb = self.maze_embedding(maze_type)        # (B, maze_embed_dim)
        maze_proj = self.proj_maze(maze_emb)             # (B, H)
        add_tensor = torch.zeros_like(last_hidden)       # (L, B, H)
        add_tensor[-1] = maze_proj
        last_hidden = last_hidden + add_tensor

        # prepare outputs
        final_outputs = torch.zeros(B, T_out, V, device=self.device)

        # compute per-sample last non-PAD index (last token in src_seq)
        mask = (src_seq != PAD_IDX)
        lengths = mask.sum(dim=1)
        lengths = torch.clamp(lengths, min=1)           # safety
        last_indices = lengths - 1                      # (B,)

        idx = torch.arange(B, device=src_seq.device)
        prev_output = src_seq[idx, last_indices]        # (B,), should be <PATH_START>
        prev_hidden = last_hidden

        for i in range(T_out):
            logits, prev_hidden = self.decoder(prev_hidden, prev_output, encoder_outputs, encoder_mask)
            final_outputs[:, i, :] = logits
            teacher_force = random.random() < teacher_ratio
            top1 = logits.argmax(dim=1)
            prev_output = target_seq[:, i] if teacher_force else top1

        return final_outputs
# ----------------- Dataset -----------------
class MazeDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_tokens = eval(row['input_sequence'])
        tgt_tokens = eval(row['output_path'])
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

# ----------------- Training / Eval -----------------
def train_epoch(model, loader, optimizer, criterion, teacher_ratio, device, clip=1.0):
    model.train()
    epoch_loss = 0.0
    total_tokens,correct_tokens=0,0
    total_seq,correct_seq=0,0
    for src, trg, maze_type, src_lengths in tqdm(loader, desc="Training"):
        src, trg, maze_type, src_lengths = src.to(device), trg.to(device), maze_type.to(device), src_lengths.to(device)
        optimizer.zero_grad()
        outputs = model(src, trg, maze_type, teacher_ratio, src_lengths)
        #why are we avoiding first position? and the first position in trg is not start token.

        output_dim = outputs.shape[-1]

        outputs_flat = outputs.reshape(-1, output_dim)
        trg_flat = trg.reshape(-1)
        loss = criterion(outputs_flat, trg_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        # token accuracy 
        preds = outputs.argmax(dim=-1)
        tar_seq = trg
        mask = (tar_seq != PAD_IDX)
        correct = ((preds == tar_seq) & mask).sum().item()
        correct_tokens += correct
        total_tokens += mask.sum().item()
        # sequence exact match (ignoring pad positions)
       
        # sequence exact-match
        batch_size = preds.size(0)
        correct_seq_count = 0
        for b in range(batch_size):
            pred_b = preds[b][mask[b]]
            true_b = trg[b][mask[b]]
            if pred_b.numel() == 0:
                match = False
            else:
                match = torch.all(pred_b == true_b).item()
            if match: 
                correct_seq_count += 1
        correct_seq += correct_seq_count
        total_seq += batch_size

    train_loss = epoch_loss / len(loader)
    token_acc = correct_tokens/total_tokens if total_tokens > 0 else 0.0
    seq_acc = correct_seq/total_seq if total_seq > 0 else 0.0

    return train_loss, token_acc, seq_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    total_tokens = 0
    total_correct_tokens = 0
    total_seq = 0
    total_correct_seq = 0
    all_preds_flat,all_targ_flat = [],[]
    with torch.no_grad():
        for src, trg, maze_type, src_lengths in tqdm(loader, desc="Evaluating"):
            src = src.to(device)
            trg = trg.to(device)
            maze_type = maze_type.to(device)
            src_lengths = src_lengths.to(device)

            outputs = model(src, trg, maze_type, teacher_ratio=0.0, src_lengths=src_lengths)
            output_dim = outputs.shape[-1]

            logits_flat = outputs.reshape(-1, output_dim)
            trg_flat = trg.reshape(-1)
            loss = criterion(logits_flat, trg_flat)
            epoch_loss += loss.item()

            preds = outputs.argmax(dim=-1)   # (B, T)
            mask = (trg != PAD_IDX)          # (B, T)

            total_correct_tokens += ((preds == trg) & mask).sum().item()
            total_tokens += mask.sum().item()

            # sequence exact-match
            batch_size = preds.size(0)
            per_batch_correct = 0
            for b in range(batch_size):
                pred_b = preds[b][mask[b]]
                true_b = trg[b][mask[b]]
                if pred_b.numel() == 0:
                    match = False
                else:
                    match = bool(torch.all(pred_b == true_b).item())
                if match:
                    per_batch_correct += 1
            total_correct_seq += per_batch_correct
            total_seq += batch_size

            # all_targ_flat.extend(trg_flat[mask].cpu().tolist())
            all_targ_flat.extend(trg[mask].cpu().tolist())
            all_preds_flat.extend(preds[mask].cpu().tolist())

    token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    seq_acc = total_correct_seq / total_seq if total_seq > 0 else 0.0
    avg_loss = epoch_loss / len(loader)

    if len(all_targ_flat) > 0:
        preds_arr,targs_arr = np.array(all_preds_flat),torch.tensor(all_targ_flat)
        f1 = f1_score(targs_arr, preds_arr, average='micro', zero_division=0)
    else:
        f1 = 0.0
    return avg_loss, token_acc, seq_acc,f1


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
    plt.savefig('rnn_loss_plot.png')
    plt.show()

    # token accuracy and test 
    plt.plot(epochs,history['train_tok_acc'],label='Train Token Accuracy')
    plt.plot(epochs,history['test_tok_acc'],label='Test Token Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Token Accuracy')
    plt.title('Token Accuracy over Epochs')
    plt.legend()
    plt.savefig('rnn_token_accuracy_plot.png')
    plt.show()

    # sequence accuracy
    plt.plot(epochs, history['train_seq_acc'], label='Train Sequence Accuracy')
    plt.plot(epochs, history['test_seq_acc'], label='Test Sequence Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Sequence Accuracy')
    plt.title('Sequence Accuracy over Epochs')
    plt.legend()
    plt.savefig('rnn_sequence_accuracy_plot.png')
    plt.show()
# ----------------- Main -----------------
def main():
    # Hyperparams
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    EMBED_DIM = 128
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    TEACHER_FORCE_RATIO = 0.5
    MAZE_EMB_DIM = 8
    SEED = 42

    # reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # load data
    train_df = pd.read_csv('COL774/train_6x6_mazes.csv')
    test_df = pd.read_csv('COL774/test_6x6_mazes.csv')

    train_dataset = MazeDataset(train_df)
    test_dataset = MazeDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # model
    encoder = Encoder(vocab_size, HIDDEN_DIM, EMBED_DIM, num_layers=NUM_LAYERS).to(device)
    decoder = Decoder(vocab_size, HIDDEN_DIM, EMBED_DIM, num_layers=NUM_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder, vocab_size, device, num_layers=NUM_LAYERS, maze_type_size=len(MAZE_TYPES), maze_embed_dim=MAZE_EMB_DIM).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    history = {
        "train_loss": [], "test_loss": [],
        "train_tok_acc": [], "test_tok_acc": [],
        "train_seq_acc": [], "test_seq_acc": [],
        "f1_scores": []
    }

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_tok_acc, train_seq_acc = train_epoch(
            model, train_loader, optimizer, criterion, TEACHER_FORCE_RATIO, device
        )

        test_loss, test_tok_acc, test_seq_acc,f1 = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_tok_acc"].append(train_tok_acc)
        history["test_tok_acc"].append(test_tok_acc)
        history["train_seq_acc"].append(train_seq_acc)
        history["test_seq_acc"].append(test_seq_acc)
        history["f1_scores"].append(f1)

        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Train Tok Acc: {train_tok_acc:.4f} | Test Tok Acc: {test_tok_acc:.4f} | "
              f"Train Seq Acc: {train_seq_acc:.4f} | Test Seq Acc: {test_seq_acc:.4f} | Test f1 Score: {f1:.4f}")
    plot_history(history)
    # save model 
    torch.save(model.state_dict(), 'rnn_model.pth')


if __name__ == "__main__":
    main()


