import math
import torch
from torch import Tensor
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, dropout: float, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        pos_embedding = torch.zeros(max_len, d_model)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        index = 0 if self.batch_first else 1
        pos_embedding = pos_embedding.unsqueeze(index)
        self.dropout = nn.Dropout(dropout)
        # pos_embedding被通过register_buffer方法注册为一个缓冲，而不是模型的可学习参数。
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor):
        if self.batch_first:
            return self.dropout(x + self.pos_embedding[:, :x.size(1), :])
        else:
            return self.dropout(x + self.pos_embedding[:x.size(0), :, :])


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    @staticmethod
    def forward(q: Tensor, k: Tensor, v: Tensor, mask=None):
        # q, k, v: [batch_size, n_head, seq_len, d_tensor]
        # and n_head * d_tensor = d_model
        # mask = [seq_len, seq_len]
        score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        if mask is not None:
            # score = score.masked_fill(mask != 0, -1e9)
            score += mask
        attention_weights = torch.softmax(score, dim=-1)
        attended_values = torch.matmul(attention_weights, v)
        return attended_values, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by num_heads"
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def split(self, x: Tensor):
        """
        split tensor by number of head
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, head, seq_len, d_tensor]
        """
        batch_size, length, d_model = x.size()
        d_tensor = d_model // self.n_head
        x = x.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return x

    @staticmethod
    def concat(x: Tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param x: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, n_head, length, d_tensor = x.size()
        d_model = n_head * d_tensor
        x = x.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return x

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
        # q, k, v: [batch_size, seq_len, d_model]
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)

        return self.fc(out)


class FeedForward(nn.Module):
    def __init__(self, dim_feedforward: int, d_model: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, n_head: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(dim_feedforward=dim_feedforward, d_model=d_model, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        _x = x
        x = self.dropout1(self.attention(q=x, k=x, v=x, mask=src_mask))
        x = self.norm1(x + _x)

        _x = x
        x = self.dropout2(self.ffn(x))
        x = self.norm2(x + _x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, n_head: int, dropout: float):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FeedForward(dim_feedforward=dim_feedforward, d_model=d_model, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, trg_mask):
        _x = tgt
        x = self.dropout1(self.self_attention(q=tgt, k=tgt, v=tgt, mask=trg_mask))
        x = self.norm1(x + _x)

        _x = x
        # memory_mask: (T, S). 我看一般实现是加src_mask，但我觉得不用，加了也是memory_mask
        x = self.dropout2(self.cross_attention(q=x, k=memory, v=memory, mask=None))
        x = self.norm2(x + _x)

        _x = x
        x = self.dropout3(self.ffn(x))
        x = self.norm3(x + _x)
        return x


class TransformerScratch(nn.Module):

    def __init__(self, num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int,
                 n_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 batch_first=True):
        super().__init__()
        assert batch_first, "must batch first"
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(100, d_model, dropout=dropout, batch_first=batch_first)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                          dim_feedforward=dim_feedforward,
                                                          n_head=n_head,
                                                          dropout=dropout)
                                             for _ in range(num_encoder_layers)])

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                          dim_feedforward=dim_feedforward,
                                                          n_head=n_head,
                                                          dropout=dropout)
                                             for _ in range(num_decoder_layers)])
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def initialize(self):
        for para in self.parameters():
            if para.dim() > 1:
                nn.init.xavier_uniform_(para)

    def encode(self, src: Tensor, src_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask)
        return src_emb

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, memory, tgt_mask)
        return tgt_emb

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        memory = self.encode(src, src_mask)
        outs = self.decode(tgt, memory, tgt_mask)
        return self.generator(outs)


class TransformerTorch(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int,
                 n_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
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
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(100, d_model, dropout=dropout, batch_first=batch_first)
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
