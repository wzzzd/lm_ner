# coding: UTF-8
import math
import torch
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.crf import CRF



class TransformerCRF(nn.Module):
    def __init__(self, config):
        super(TransformerCRF, self).__init__()
        self.vocab_size = len(pkl.load(open(config.path_vocab, 'rb')))
        self.num_label = len([ x for x in open(config.path_tgt_map, 'r').readlines() if x.strip()])
        self.embed = 300
        self.hidden_size = 300
        self.n_head = 5
        self.dim_feedforward = 1024
        self.num_layers = 4
        self.dropout_rate = 0.1

        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed, padding_idx=self.vocab_size - 1)
        self.dropout = nn.Dropout(self.dropout_rate)
        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_head, dim_feedforward=self.dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.pos_encoder = PositionalEncoding(d_model=self.hidden_size, dropout=self.dropout_rate)
        # self.pos_encoder = Positional_Encoding(embed=self.embed, pad_size=128, dropout=self.dropout_rate, device=config.device)
        # full connect
        self.classifier = nn.Linear(self.hidden_size, self.num_label)
        # CRF
        self.crf = CRF(num_tags=self.num_label, batch_first=True)
 

    def forward(self, input_ids, labels=None, attention_mask=None):
        embeds = self.embedding(input_ids)               # (batch_size, seq_len, emb_size)
        embeds = embeds.transpose(0,1)                         # (seq_len, batch_size, emb_size)
        embeds = self.pos_encoder(embeds)                   # (seq_len, batch_size, emb_size)
        # embeds = embeds.transpose(0,1)
        trans_out = self.transformer_encoder(embeds)        # (seq_len, batch_size, emb_size)
        trans_out = trans_out.transpose(0,1)                # (batch_size, seq_len, emb_size)
        trans_out = self.dropout(trans_out)                 # (batch_size, seq_len, emb_size)
        logits = self.classifier(trans_out)                 # (batch_size, seq_len, emb_size)
        output = (None, logits)
        if labels is not None:
            size = logits.size()
            labels = labels[:,:size[1]]
            loss = -1 * self.crf(emissions = logits, tags=labels, mask=attention_mask)
            output = (loss, logits)
        return output



class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        y = self.pe[:x.size(0), :]
        x = x + y
        return self.dropout(x)


# class Positional_Encoding(nn.Module):
#     def __init__(self, embed, pad_size, dropout, device):
#         super(Positional_Encoding, self).__init__()
#         self.device = device
#         self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
#         self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
#         self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
#         out = self.dropout(out)
#         return out