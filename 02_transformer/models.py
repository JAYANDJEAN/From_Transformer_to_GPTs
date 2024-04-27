import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F

'''
if batch_first=False 
data after TokenEmbedding is (SEQ_LEN, BATCH_SIZE, D_MODEL)
after PositionalEncoding is still (SEQ_LEN, BATCH_SIZE, D_MODEL)
next is MultiHeadAttention, so MultiHeadAttention need to consider batch_first
'''


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
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor):
        if self.batch_first:
            return self.dropout(x + self.pos_embedding[:, :x.size(1), :])
        else:
            return self.dropout(x + self.pos_embedding[:x.size(0), :, :])


# 以下只处理batch_size=True，如果不是，在前面就给换过来。
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
        # q, k, v: [batch_size, n_head, seq_len, d_tensor]
        # and n_head * d_tensor = d_model
        # mask = [seq_len, seq_len]
        score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(score, dim=-1)
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

    def concat(self, x: Tensor):
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
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
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

    def forward(self, dec, enc, trg_mask, src_mask):
        _x = dec
        print(dec.shape)
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            # memory_mask: (T, S).
            x = self.cross_attention(q=x, k=enc, v=enc, mask=None)

            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


MAX_LEN = 100
D_MODEL = 512
BATCH_SIZE = 128
SEQ_LEN = 23
N_HEAD = 8
DIM_FF = 256
DROPOUT = 0.1
TGT_SEQ_LEN = 17


def positional_encoding():
    def positional_encoding(max_len, d_model):
        pos_enc = np.zeros((max_len, d_model))
        for k in range(max_len):
            for i in range(0, d_model, 2):
                pos_enc[k, i] = np.sin(k / (100 ** ((2 * i) / d_model)))
                pos_enc[k, i + 1] = np.cos(k / (100 ** ((2 * i) / d_model)))
        return pos_enc

    pe1 = PositionalEncoding(MAX_LEN, D_MODEL, 0.1, batch_first=False)
    pe2 = PositionalEncoding(MAX_LEN, D_MODEL, 0.1, batch_first=True)
    pos_enc_list = [positional_encoding(MAX_LEN, D_MODEL),
                    pe1.pos_embedding.squeeze().numpy(),
                    pe2.pos_embedding.squeeze().numpy()]
    title_list = ['Explicit Loop Positional Encoding',
                  'PyTorch Positional Encoding seq_len',
                  'PyTorch Positional Encoding batch_first']

    input_tensor = torch.randn(SEQ_LEN, BATCH_SIZE, D_MODEL)
    output_tensor = pe1(input_tensor)
    assert output_tensor.shape == input_tensor.shape

    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    output_tensor = pe2(input_tensor)
    assert output_tensor.shape == input_tensor.shape

    plt.figure(figsize=(10, 8))
    for i in range(len(pos_enc_list)):
        plt.subplot(2, 2, 1 + i)
        plt.imshow(pos_enc_list[i].T, cmap='viridis', aspect='auto', origin='lower')
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title(title_list[i])
        plt.colorbar()

    plt.tight_layout()
    plt.savefig('positional_encoding.png')


def scale_dot_product_attention():
    attention = ScaleDotProductAttention()

    q = torch.randn(BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    k = torch.randn(BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    v = torch.randn(BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    attended_values, attention_weights = attention(q, k, v)
    assert attended_values.shape == (BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    assert attention_weights.shape == (BATCH_SIZE, N_HEAD, SEQ_LEN, SEQ_LEN)

    mask = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.bool)
    mask[:, -1] = 1  # 在最后一个位置上添加 mask，用于测试

    # 测试有 mask 的情况
    attended_values_masked, attention_weights_masked = attention(q, k, v, mask)
    assert attended_values_masked.shape == (BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    assert attention_weights_masked.shape == (BATCH_SIZE, N_HEAD, SEQ_LEN, SEQ_LEN)


def multi_head_attention():
    model = MultiHeadAttention(D_MODEL, N_HEAD)
    # Generate random input tensors
    q = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    k = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    v = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    out = model(q, k, v)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def encoder_layer():
    model = EncoderLayer(D_MODEL, DIM_FF, N_HEAD, DROPOUT)
    src_emb = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    src_mask = torch.ones(SEQ_LEN, SEQ_LEN)
    out = model(src_emb, src_mask)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def decoder_layer():
    model = DecoderLayer(D_MODEL, DIM_FF, N_HEAD, DROPOUT)
    src_emb = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    src_mask = torch.ones(SEQ_LEN, SEQ_LEN)
    tgt_emb = torch.randn(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)
    tgt_mask = torch.ones(TGT_SEQ_LEN, TGT_SEQ_LEN)

    out = model(tgt_emb, src_emb, tgt_mask, src_mask)
    print(out.shape)


if __name__ == '__main__':
    # test_positional_encoding()
    # test_scale_dot_product_attention()
    # multi_head_attention()
    decoder_layer()
