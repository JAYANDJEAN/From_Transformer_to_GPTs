from torch import nn
import torch
from models import *


class Seq2SeqTransformer(nn.Module):

    def __init__(self, num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 n_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 batch_first=False):
        super().__init__()

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(100, emb_size, dropout=dropout, batch_first=batch_first)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=emb_size,
                                                          ffn_hidden=dim_feedforward,
                                                          n_head=n_head,
                                                          drop_prob=dropout)
                                             for _ in range(num_encoder_layers)])

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model=emb_size,
                                                          ffn_hidden=dim_feedforward,
                                                          n_head=n_head,
                                                          drop_prob=dropout)
                                             for _ in range(num_decoder_layers)])

        self.linear = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask)

        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, src_emb, src_mask, tgt_mask)

        # pass to LM head
        output = self.linear(tgt_emb)
        return output
