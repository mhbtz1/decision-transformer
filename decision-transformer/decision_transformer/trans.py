import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_size = embed_size
        self.Q = torch.rand(self.embed_size, self.embed_size)
        self.K = torch.rand(self.embed_size, self.embed_size)
        self.V = torch.rand(self.embed_size, self.embed_size)
    def forward(self, tkns: torch.Tensor, mask=None): # tkns dim: ( embed_size, num_tokens)
        Q = torch.matmul(self.Q, tkns)
        K = torch.matmul(self.K, tkns)
        V = torch.matmul(self.V, tkns)
        scores = torch.matmul(self.Q, self.K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask==0, 1e-9)
        
        attn = F.softmax(scores, dim=-1)
        sftmax = torch.matmul(attn, self.V)
        return attn, sftmax

class MultiheadedAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiheadedAttention, self).__init__()
        self.matrix_dim = embed_size // num_heads
        self.dot_attn = [ScaledDotProductAttention(self.matrix_dim) for _ in range(num_heads)]
        self.W_O = torch.rand(self.matrix_dim, self.matrix_dim)
        
    def forward(self, tkns: torch.Tensor, mask=None):
        t = self.dot_attn[0].forward(tkns[0, :, :])
        for i, mod in enumerate(self.dot_attn[1:]):
            t = torch.cat( (t, mod.forward(tkns[i, :, :])) )
        return torch.matmul(self.W_O, t)

            
if __name__== '__main__':
    model = ScaledDotProductAttention(64)
    multihead = MultiheadedAttention(64, 10)
    resp = multihead(torch.rand(5, 64, 10))
    print(resp)

