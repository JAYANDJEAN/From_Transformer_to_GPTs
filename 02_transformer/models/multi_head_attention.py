import torch
import torch.nn as nn
from scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        '''
        :param q:
        :param k:
        :param v:
        :param mask:
        :return:
        这里的拆分和拼接的用处是：
        '''
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
        return self.w_concat(out)


if __name__ == '__main__':
    d_model = 512
    n_head = 8
    seq_length = 20
    batch_size = 128

    # Create MultiHeadAttention model
    model = MultiHeadAttention(d_model, n_head)

    # Generate random input tensors
    q = torch.randn(batch_size, seq_length, d_model)
    k = torch.randn(batch_size, seq_length, d_model)
    v = torch.randn(batch_size, seq_length, d_model)

    out = model(q, k, v)
    print(out.shape)
