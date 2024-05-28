from torch import Tensor
import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        pos_embedding = torch.zeros(max_len, d_model)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor):
        return self.dropout(x + self.pos_embedding[:, :x.size(1), :])


class TransformerTorch(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int,
                 n_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 max_seq_len: int = 1000,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 batch_first=True):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=n_head,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=batch_first)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model, dropout=dropout)
        self.initialize()

    def initialize(self):
        for para in self.parameters():
            if para.dim() > 1:
                nn.init.xavier_uniform_(para)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, None, None, None)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
