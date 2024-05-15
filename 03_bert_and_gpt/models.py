from transformers import AutoModel
from torch import nn, Tensor
import torch
import math


class PositionalEncoding(nn.Module):
    r"""PositionalEncoding
    batch_first = true
    """

    def __init__(self, max_len: int, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        pos_embedding = torch.zeros(max_len, d_model)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        # The pos_embedding is registered as a buffer through the register_buffer method,
        # rather than being a learnable parameter of the model.
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor):
        return self.dropout(x + self.pos_embedding[:, :x.size(1), :])


class Seq2Seq_Pre_Encode(nn.Module):
    def __init__(self, model_path, tgt_vocab_size, n_layers=2, n_heads=4, dropout_rate=0.1):
        super(Seq2Seq_Pre_Encode, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_path)
        config = self.encoder.config.to_dict()
        hidden_size = config['hidden_size']

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dropout=dropout_rate,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.generator = nn.Linear(hidden_size, tgt_vocab_size)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(1000, hidden_size, dropout=dropout_rate)
        self.initialize()

    def initialize(self):
        for para in self.decoder.parameters():
            if para.dim() > 1:
                nn.init.xavier_uniform_(para)
        for para in self.generator.parameters():
            if para.dim() > 1:
                nn.init.xavier_uniform_(para)
        for para in self.tgt_emb.parameters():
            if para.dim() > 1:
                nn.init.xavier_uniform_(para)

    def encode(self, src_ids: Tensor, src_mask: Tensor):
        encoder_outputs = self.encoder(input_ids=src_ids, attention_mask=src_mask)
        return encoder_outputs.last_hidden_state

    def decode(self, tgt_ids: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt_emb = self.positional_encoding(self.tgt_emb(tgt_ids))
        return self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, memory_mask=None)

    def forward(self, src_ids: Tensor, src_mask: Tensor, tgt_ids: Tensor, tgt_mask: Tensor):
        encoder_outputs = self.encoder(input_ids=src_ids, attention_mask=src_mask)
        last_hidden_state = encoder_outputs.last_hidden_state
        tgt_emb = self.positional_encoding(self.tgt_emb(tgt_ids))
        decoder_outputs = self.decoder(tgt=tgt_emb, memory=last_hidden_state, tgt_mask=tgt_mask, memory_mask=None)
        return self.generator(decoder_outputs)


class TransformerTorch(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int,
                 n_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 max_seq_len: int = 100,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=n_head,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
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
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
