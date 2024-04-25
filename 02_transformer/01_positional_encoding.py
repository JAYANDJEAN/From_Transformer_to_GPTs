import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

max_len = 100
emb_size = 512


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, emb_size):
        super(PositionalEncoding, self).__init__()
        pos_embedding = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2, dtype=torch.float) * -(math.log(10000.0) / emb_size))
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1)]



def positional_encoding(max_len, emb_size):
    pos_enc = np.zeros((max_len, emb_size))
    for k in range(max_len):
        for i in range(0, emb_size, 2):
            pos_enc[k, i] = np.sin(k / (100 ** ((2 * i) / emb_size)))
            pos_enc[k, i + 1] = np.cos(k / (100 ** ((2 * i) / emb_size)))
    return pos_enc


pos_enc_1 = positional_encoding(max_len, emb_size)
pos_encoding = PositionalEncoding(max_len, emb_size)
pos_enc_2 = pos_encoding.pos_embedding.squeeze().numpy()

input_tensor = torch.randn(128, 23, emb_size)
output_tensor = pos_encoding(input_tensor)
assert output_tensor.shape == (128, 23, emb_size)

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.imshow(pos_enc_1.T, cmap='viridis', aspect='auto', origin='lower')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Explicit Loop Positional Encoding')
plt.colorbar(label='Value')
plt.subplot(1, 2, 2)
plt.imshow(pos_enc_2.T, cmap='viridis', aspect='auto', origin='lower')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('PyTorch Positional Encoding')
plt.colorbar(label='Value')

plt.tight_layout()
plt.show()
