import math
import torch
import torch.nn.functional as F
from embedding import BERTEmbedding

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
        attention_masked = attention.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(attention_masked, dim = -1)
        scores = self.dropout(scores)
        
        output = scores @ value
        output = output.permute(0, 2, 1, 3).contiguous().view(output.shape[0], -1, self.n_heads * self.d_k)

        return self.linear(output)
    
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, middle_dim, dropout = 0.1):
        super().__init__()
        
        self.l1 = torch.nn.Linear(d_model, middle_dim)
        self.l2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        return self.l2(self.dropout(self.activation(self.l1(x))))
    

class Encoder(torch.nn.Module):
    def __init__(self, n_heads, d_model, dropout = 0.1):
        super().__init__()

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.multi_attention = MultiHeadedAttention(n_heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_model * 4, dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.multi_attention(x, x, x, mask)
        output = self.layer_norm1( attention + self.dropout(attention))
        output = self.layer_norm2(output + self.dropout(self.ff(output)))
        return output
    
class BERT(torch.nn.Module):
    def __init__(self, n_layers, n_heads, d_model, vocab_size, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.embedding = BERTEmbedding(vocab_size, d_model)
        self.encoder_blocks = torch.nn.ModuleList(
            [Encoder(n_heads, d_model, dropout) for _ in range(n_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, segment_label):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_label)
        for encoder in self.encoder_blocks:
            x = encoder(x, mask)
        output = self.dropout(x)

        return output


