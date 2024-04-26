import torch
import torch.nn.functional as F
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    """
    Compute scaled dot product attention.
    Query : given sentence that we focus on (decoder)
    Key : every sentence to check relationship with Query (encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        # Dot product of Query with Key^T to compute similarity
        # input is 4 dimension tensor
        # [batch_size, n_head, seq_length, emb_size]
        score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # Mask value chosen to be large negative

        attention_weights = F.softmax(score, dim=-1)
        attended_values = torch.matmul(attention_weights, v)

        return attended_values, attention_weights


if __name__ == '__main__':
    batch_size = 128
    seq_length = 20
    emb_size = 512
    n_head = 8

    attention = ScaleDotProductAttention()

    q = torch.randn(batch_size, n_head, seq_length, emb_size)
    k = torch.randn(batch_size, n_head, seq_length, emb_size)
    v = torch.randn(batch_size, n_head, seq_length, emb_size)

    # 测试没有 mask 的情况
    attended_values, attention_weights = attention(q, k, v)
    print("Without Mask:")
    print("Attended Values Shape:", attended_values.shape)  # 预期输出：(batch_size, n_head, seq_length, emb_size)
    print("Attention Weights Shape:", attention_weights.shape)  # 预期输出：(batch_size, n_head, seq_length, seq_length)

    # 创建测试 mask 数据
    mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
    mask[:, -1] = 1  # 在最后一个位置上添加 mask，用于测试

    # 测试有 mask 的情况
    attended_values_masked, attention_weights_masked = attention(q, k, v, mask)
    print("\nWith Mask:")
    print("Attended Values Shape:", attended_values_masked.shape)  # 预期输出：(batch_size, n_head, seq_length, emb_size)
    print("Attention Weights Shape:",
          attention_weights_masked.shape)  # 预期输出：(batch_size, n_head, seq_length, seq_length)
