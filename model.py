import math
import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int , seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.seq_len = seq_len

        