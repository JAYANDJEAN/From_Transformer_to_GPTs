import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, src_mask, trg_mask):
        # Compute self attention
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout(x)
        x = self.norm1(x + dec)

        # Compute encoder-decoder attention
        if enc is not None:
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout(x)
            x = self.norm2(x + x)  # x = self.norm2(x + dec)

        # Positionwise feed forward network
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + x)  # x = self.norm3(x + dec)

        return x
