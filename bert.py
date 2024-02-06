import math
import torch
import torch.nn.functional as F

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, n_heads, d_model, dropout= 0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.Q = torch.nn.Linear(d_model, d_model)
        self.K = torch.nn.Linear(d_model, d_model)
        self.V = torch.nn.Linear(d_model, d_model)

        self.linear = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        query = self.Q(query)
        key = self.K(key)
        value = self.V(value)

        query = query.view(query.shape[0], -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        attention = (query @ key.permute(0, 1, 3, 2)) / math.sqrt(self.d_k)
        print(attention.shape)
        attention_masked = attention.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(attention_masked, dim = -1)
        scores = self.dropout(scores)
        
        output = scores @ value
        output = output.permute(0, 2, 1, 3).contiguous().view(output.shape[0], -1, self.n_heads * self.d_k)

        return self.linear(output)

