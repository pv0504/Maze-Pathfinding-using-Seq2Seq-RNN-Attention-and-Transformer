import sys
import ast
import math
import torch
import torch.nn as nn
import pandas as pd
from a import Encoder, Decoder
from b import Positional_Encoding
import matplotlib.pyplot as plt 

# ---------------------------
# Vocabulary / constants
# ---------------------------
PAD_IDX = 0
special_tokens = [
    '<PAD>', '<ADJLIST_START>', '<ADJLIST_END>',
    '<ORIGIN_START>', '<ORIGIN_END>',
    '<TARGET_START>', '<TARGET_END>',
    '<PATH_START>', '<PATH_END>', '<-->', ';'
]
coords = [f'({i},{j})' for i in range(6) for j in range(6)]
vocab_list = special_tokens + coords
token2id = {tok: idx for idx, tok in enumerate(vocab_list)}
id2token = {idx: tok for tok, idx in token2id.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

    @torch.no_grad()
    def predict(self, src_ids, max_len=200):
        
        src = src_ids.unsqueeze(0).to(device)  # (1,S)
        encoder_outputs, hidden = self.encoder(src)  # outputs: (1,S,H) hidden: (L,1,H)
        encoder_mask = (src != PAD_IDX).long()       # (1,S)

        B = src.size(0)

        # i am consider here decoder with <PATH_START>
        prev_output = torch.full((B,), token2id['<PATH_START>'], dtype=torch.long, device=device) 
        prev_hidden = hidden

        out_tokens = []
        for _ in range(max_len):
            logits, prev_hidden = self.decoder(prev_hidden, prev_output, encoder_outputs, encoder_mask)
            next_tok = logits.argmax(dim=1).item()
            out_tokens.append(next_tok)
            prev_output = torch.tensor([next_tok], device=device,dtype=torch.long)
            if next_tok in id2token and id2token[next_tok] == '<PATH_END>':
                break
            elif next_tok not in id2token:
                print(f"predicted token id {next_tok} not in vocabulary. Stopping generation.")
                break 
        return out_tokens

# ---------------------------
# Transformer (same arch used in training)
# ---------------------------




def future_mask(sz):
    return torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)

class TransformerModel(nn.Module):
    def __init__(self, vocab_dim, nheads=8, embed_dim=128, ff_dim=512, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # embedding with padding index
        self.embedding = nn.Embedding(vocab_dim, embed_dim, padding_idx=PAD_IDX)
        self.pos_enc = Positional_Encoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nheads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nheads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.num_layers)

        self.output = nn.Linear(embed_dim, vocab_dim)

        for parm in self.parameters():
            if parm.dim() > 1:
                nn.init.xavier_uniform_(parm)

    @torch.no_grad()
    def predict(self, src_ids, max_len=200):
        # src_ids: 1D tensor (S,)
        src = src_ids.unsqueeze(0).to(device)  # (1,S)
        src_emb = self.embedding(src) * math.sqrt(self.embed_dim)
        src_emb = self.pos_enc(src_emb)
        src_key_padding_mask = (src == PAD_IDX)  # (1,S)
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        ys = torch.tensor([[token2id['<PATH_START>']]], device=device)  
        for _ in range(max_len):
            tgt_emb = self.embedding(ys) * math.sqrt(self.embed_dim)
            tgt_emb = self.pos_enc(tgt_emb)
            tgt_mask = future_mask(ys.size(1)).to(device)
            out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
            out = self.output(out)  # (1, T, vocab_dim)
            next_tok = out[:, -1].argmax(dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)
            if next_tok in id2token and id2token[next_tok] == '<PATH_END>':
                break
            elif next_tok not in id2token:
                print(f"predicted token id {next_tok} not in vocabulary. Stopping generation.")
                break 
        # i kept <PATH_START> token at start, so dropping..
        return ys[0, 1:].tolist()


# load model 

def load_model(model_path, model_type):
    if model_type == "rnn":
        encoder = Encoder(vocab_size=len(vocab_list), hidden_size=512, embed_size=128, num_layers=2)
        decoder = Decoder(vocab_size=len(vocab_list), hidden_size=512, embed_size=128, num_layers=2)
        model = Seq2Seq(
            encoder,
            decoder,
            vocab_size=len(vocab_list),
            device=device,
            maze_type_size=2,
            maze_embed_dim=8
        ).to(device)

    else:
        model = TransformerModel(vocab_dim=len(vocab_list)).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    model_path,model_type,input_csv,output_csv = sys.argv[1:5]
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} examples from {input_csv}")
    model = load_model(model_path,model_type=model_type)
    model.to(device)
    model.eval()
    predictions = []

    for i,r in df.iterrows():
        assert r is not None
        src_tokens = eval(r["input_sequence"])
        src_ids = torch.tensor([token2id[t] for t in src_tokens], dtype=torch.long)
        out_ids = model.predict(src_ids, max_len=200)
        out_tokens = [id2token[i] for i in out_ids]
        
        out_row = r.to_dict()
        out_row["output_path"] = str(out_tokens) 
        predictions.append(out_row)


    out_df = pd.DataFrame(predictions)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote predictions to: {output_csv}")

if __name__ == "__main__":
    main()



