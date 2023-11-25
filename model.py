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
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x) # (batch_size, seq_len, d_ff)
        x = self.relu(x) # (batch_size, seq_len, d_ff)
        x = self.dropout(x) # (batch_size, seq_len, d_ff)
        x = self.linear2(x) # (batch_size, seq_len, d_model)
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h (number of heads)"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.dropout = dropout
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask=None, dropout: nn.Dropout=None):
        """Compute 'Scaled Dot Product Attention'
        query: (batch_size, seq_len, d_model)
        key: (batch_size, seq_len, d_model)
        value: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len, seq_len)
        dropout: nn.Dropout
        """
        attention_scores = query @ key.transpose(-2, -1) / torch.sqrt(torch.tensor(self.d_k))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value, attention_scores)
    

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_k(k)   # (batch_size, seq_len, d_model)
        value = self.w_v(v)

        # split d_model into h heads
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)
        key = key.view(key.size(0), key.size(1), self.h, self,d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query=query, # (batch_size, h, seq_len, d_k)
                                                                key=key, # (batch_size, h, seq_len, d_k)
                                                                value=value, # (batch_size, h, seq_len, d_k)
                                                                mask=mask, # (batch_size, seq_len, seq_len)
                                                                dropout=self.dropout)
        # concat heads and put through final linear layer
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model) # (batch_size, seq_len, d_model)
        x = self.w_o(x) # (batch_size, seq_len, d_model)
        return x
    

class ResidualConnection(nn.Module):
    def __init(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float = 0.1):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        # x: (batch_size, seq_len, d_model) 
        # src_mask: (batch_size, seq_len, seq_len)
        # comment: src_mask is used to mask out the padding tokens
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x # (batch_size, seq_len, d_model)

class Encoder(nn.Module):
    def __init(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, src_mask):
        # x: (batch_size, seq_len, d_model) 
        # src_mask: (batch_size, seq_len, seq_len)
        # comment: src_mask is used to mask out the padding tokens
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
