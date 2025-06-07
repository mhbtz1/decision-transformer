import torch
import torch.nn.functional as F
import math

D_K = 16

def self_attn(Q, K, V, mask=None):
    raw_mul = torch.matmul(F.softmax(torch.div(torch.matmul(Q, K.transpose(-2, -1)), math.sqrt(D_K))), V)
    if mask:
        raw_mul[mask == 0] = -1e9
    logits = raw_mul.softmax(dim=1)

    print(raw_mul)
    print('-' * 10)
    print(logits)


Q = torch.randn(D_K, D_K)
K = torch.randn(D_K, D_K)
V = torch.randn(D_K, D_K)
self_attn(Q, K, V)
