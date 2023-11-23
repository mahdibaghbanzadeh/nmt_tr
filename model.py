import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 0::2 means 0, 2, 4, 6, ...
        pe[:, 1::2] = torch.cos(position * div_term) # 1::2 means 1, 3, 5, 7, ...

        # batch first
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # alpha multiply with std
        self.bias = nn.Parameter(torch.zeros(d_model)) # bias add to mean
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

