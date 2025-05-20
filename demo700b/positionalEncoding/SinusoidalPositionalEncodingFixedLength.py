import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncodingFixedLength(nn.Module):
    def __init__(self, input_dim, d_model, step_back: int = 4):
        super().__init__()
        position = torch.arange(step_back).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(100.0) / d_model))
        self.pe = torch.zeros(step_back, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term) * 10
        self.pe[:, 1::2] = torch.cos(position * div_term) * 10

        self.fc = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x 形状: [batch_size, seq_len, d_model]
        x = self.fc(x)
        x = x + self.pe
        return x
