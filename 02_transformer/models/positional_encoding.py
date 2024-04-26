import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, emb_size: int, dropout: float, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, emb_size, 2) * (math.log(10000.0) / emb_size))
        pos_embedding = torch.zeros(max_len, emb_size)
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


def positional_encoding(max_len, emb_size):
    pos_enc = np.zeros((max_len, emb_size))
    for k in range(max_len):
        for i in range(0, emb_size, 2):
            pos_enc[k, i] = np.sin(k / (100 ** ((2 * i) / emb_size)))
            pos_enc[k, i + 1] = np.cos(k / (100 ** ((2 * i) / emb_size)))
    return pos_enc


if __name__ == '__main__':
    max_len = 100
    emb_size = 512
    batch_size = 128
    seq_len = 23
    pe1 = PositionalEncoding(max_len, emb_size, 0.1, batch_first=False)
    pe2 = PositionalEncoding(max_len, emb_size, 0.1, batch_first=True)
    pos_enc_list = [positional_encoding(max_len, emb_size),
                    pe1.pos_embedding.squeeze().numpy(),
                    pe2.pos_embedding.squeeze().numpy()]
    title_list = ['Explicit Loop Positional Encoding',
                  'PyTorch Positional Encoding seq_len',
                  'PyTorch Positional Encoding batch_first']

    input_tensor = torch.randn(seq_len, batch_size, emb_size)
    output_tensor = pe1(input_tensor)
    assert output_tensor.shape == input_tensor.shape

    input_tensor = torch.randn(batch_size, seq_len, emb_size)
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
    plt.show()
