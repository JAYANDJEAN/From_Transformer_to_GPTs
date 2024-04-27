import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F


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


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
        # q, k, v: [batch_size, n_head, seq_len, d_model]
        # mask = [seq_len, seq_len]
        score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # Mask value chosen to be large negative
        attention_weights = F.softmax(score, dim=-1)
        attended_values = torch.matmul(attention_weights, v)
        return attended_values, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
        q = self.w_q(q).chunk(self.n_head, dim=-1)
        k = self.w_k(k).chunk(self.n_head, dim=-1)
        v = self.w_v(v).chunk(self.n_head, dim=-1)
        q = torch.cat(q, dim=0)
        k = torch.cat(k, dim=0)
        v = torch.cat(v, dim=0)

        out, attention = self.attention(q, k, v, mask=mask)
        # Split and concatenate the output
        out = torch.chunk(out, self.n_head, dim=0)
        out = torch.cat(out, dim=-1)
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

    def forward(self, x: Tensor, src_mask):
        # Compute self attention
        attn_output = self.attention(x, x, x, mask=src_mask)
        x = self.dropout1(attn_output) + x
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = self.dropout2(ffn_output) + x
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, n_head: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.ffn = FeedForward(dim_feedforward=dim_feedforward, d_model=d_model, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, dec, enc, src_mask, trg_mask):
        # Compute self attention
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout(x)
        x = self.norm1(x + dec)
        if enc is not None:
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout(x)
            x = self.norm2(x + x)  # x = self.norm2(x + dec)

        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + x)  # x = self.norm3(x + dec)

        return x


MAX_LEN = 100
D_MODEL = 512
BATCH_SIZE = 128
SEQ_LEN = 23
N_HEAD = 8
DIM_FF = 256
DROPOUT = 0.1


def test_positional_encoding():
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


def test_scale_dot_product_attention():
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


def test_multi_head_attention():
    model = MultiHeadAttention(D_MODEL, N_HEAD)
    # Generate random input tensors
    q = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    k = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    v = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    out = model(q, k, v)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def test_encoder_layer():
    model = EncoderLayer(D_MODEL, DIM_FF, N_HEAD, DROPOUT)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    src_mask = torch.ones(SEQ_LEN, SEQ_LEN)
    out = model(x, src_mask)
    print(out.shape)


if __name__ == '__main__':
    test_positional_encoding()
    test_scale_dot_product_attention()
    test_multi_head_attention()
    test_encoder_layer()
