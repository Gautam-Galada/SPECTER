import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        scores = scores / self.scale 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(attn_probs), value) 
        return output, attn_probs

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "`d_model` should be a multiple of `n_heads`"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads  # head_dim
        self.dropout_rate = dropout_rate
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(np.sqrt(self.d_k), dropout_rate)
    

    def split_heads(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        return x


    def group_heads(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return x


    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query)) 
        K = self.split_heads(self.W_k(key))  
        V = self.split_heads(self.W_v(value))  
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(Q, K, V, mask)
        x = self.group_heads(x)
        x = self.W_o(x)
        #x is  context vector (CoVe) of all the attention heads 
        return x, attn        