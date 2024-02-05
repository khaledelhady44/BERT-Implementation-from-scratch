import math
import torch

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        self.embedding = torch.zeros(max_len, d_model, requires_grad=False).float()

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                self.embedding[pos, i] = math.sin(pos / (10000 ** ((2*i)/d_model)))
                self.embedding[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1)) / d_model)))

        self.embedding = self.embedding.unsqueeze(0)
    
    def forward(self):
        return self.embedding
    

class BERTEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len = 64, dropout = 0.1):
        super().__init__()
        self.embed_size = embed_size
        self.token_emb = torch.nn.Embedding(vocab_size, embed_size, padding_idx = 0)
        self.seg_emb = torch.nn.Embedding(3, embed_size, padding_idx= 0)
        self.pos_emb = PositionalEmbedding(seq_len, embed_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, segment):
        embedding_sum = self.token_emb(x) + self.seg_emb(segment) + self.pos_emb()
        return self.dropout(embedding_sum)
        