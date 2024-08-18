import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# functions relevant for the implementation of the transformer are at the top here

def attention(query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1) ) / math.sqrt(value.size(0))
    if mask is not None: 
        scores[mask == 0] = -1e9
    logits = dropout(scores.softmax(dim=-1))
    return torch.matmul(logits, value), logits

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

# utility classes

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, output_size, dropout_prob):
        super(PositionWiseFFN, self).__init__()
        self.w_1 = nn.Linear(d_model, output_size)
        self.w_2 = nn.Linear(output_size, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class MultiheadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.25):
        super(MultiheadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batches = query.size(0)
        # generate query, key, value per attn layer
        query, key, value = [lin(x).view(n_batches, -1, self.h, self.d_k) for lin, x in zip(self.linears, (query, key, value))]
        attn_outputs, attn_probs = attention(query, key, value, mask, dropout=self.dropout)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(n_batches, -1, self.d_k)
        return self.linears[-1](attn_outputs)



class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        pass

    def step(self):
        pass

    def zero_grad(self, set_to_zero=False):
        pass

# final layer of the decoder transformer
# takes the output from several decoder layers and outputs
class Generator:
    def __init__(self, d_model, vocab_size):
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return log_softmax(self.proj(x), dim=1)

class LayerNorm(nn.Module):
    # Facilitate LayerNorm within the transformer model
    # Paper related to Layer Normalization here: https://arxiv.org/pdf/1607.06450
    def __init__(self, features: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features)) # using nn.Parameter makes this tensor trainable in the model (i.e. gradients update when performing backprop)
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return self.a_2 * ( (x - mean) / (torch.sqrt(std + self.eps)) ) + self.b_2

class SublayerConnection(nn.Module):
    # Residual connection followed by LayerNorm for an arbitrary layer sublayer
    def __init__(self, size, dropout_prob=0.25):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x, sublayer: nn.Module):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn: nn.Module, feed_forward: nn.Module, dropout_prob):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.dropout = nn.Dropout(dropout_prob)
        self.sublayer = clones(SublayerConnection(size, dropout_prob), 2)
        self.feed_forward = feed_forward
        self.size = size
    def forward(self, x, mask):
        # this implements the unmasked multiheaded attention + LayerNorm + residual connection, and then feedforward
        # i.e. this is a single encoder block (as per the original Transformer paper)
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.N = N
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # every forward pass, normalize the output for more efficient training (LayerNorm paper)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N) #takes a series of DecoderLayers and applies them to input
        self.norm = LayerNorm(layer.size)
        self.N = N
    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_embed, dropout_prob=0.25, ctx_size=5000):
        super(PositionalEmbedding, self).__init__()
        self.position_embed = torch.zeros(ctx_size, d_embed)
        pos = torch.arange(0, d_embed).unsqueeze(1)
        pe = torch.zeros(d_embed, ctx_size)
        self.dropout = nn.Dropout(dropout_prob)
        div_term = torch.exp( 2 * torch.arange(0, ctx_size / 2) * -math.log(10000) / d_embed)
        print()
        pe[ : , 0::2] = torch.sin(pos *  div_term)
        pe[ :,  1::2] = torch.cos(pos * div_term)
        self.pe = pe

    def forward(self, x): # x is a batch / minibatch tensor of shape (n_batch, ctx_size, d_embed)
        x = x + (self.pe[:, 0::2] + self.pe[:, 1::2]).requires_grad(False) # we don'[t ]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# standard interface for an encoder-decoder model
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.generator = generator
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, mem, target, src_mask, target_mask):
        return self.decode(self.target_embed(target), mem, src_mask, target_mask)
    

