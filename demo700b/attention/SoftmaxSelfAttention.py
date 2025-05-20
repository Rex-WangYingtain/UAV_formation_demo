import torch
import torch.nn as nn
import math

class SoftmaxSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x 形状: [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, d_model]
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]

        # 计算注意力分数：q 和 k 的点积，然后除以 sqrt(d_model)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_weights = self.softmax(attn_scores)  # [batch_size, seq_len, seq_len]

        # 加权求和值
        output = torch.matmul(attn_weights, v)  # [batch_size, seq_len, d_model]

        return output
