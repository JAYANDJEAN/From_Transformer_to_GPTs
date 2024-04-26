import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # Compute self attention
        attn_output = self.attention(x, x, x, mask=src_mask)

        # Add and normalize
        x = self.dropout1(attn_output) + x
        x = self.norm1(x)

        # Positionwise feed forward network
        ffn_output = self.ffn(x)

        # Add and normalize
        x = self.dropout2(ffn_output) + x
        x = self.norm2(x)
        return x


if __name__ == '__main__':
    d_model = 128
    ffn_hidden = 256
    n_head = 8
    seq_length = 10
    batch_size = 16

    # Create EncoderLayer model
    model = EncoderLayer(d_model, ffn_hidden, n_head, drop_prob=0.1)

    # Generate random input tensor and mask
    x = torch.randn(batch_size, seq_length, d_model)
    src_mask = torch.ones(seq_length, seq_length)

    # Call forward method
    out = model(x, src_mask)
    print(out.shape)
