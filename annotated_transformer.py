import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        pass

    def step(self):
        pass

    def zero_grad(self, set_to_zero=False):
        pass

class Generator:
    def __init__(self, d_model, vocab_size):
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return log_softmax(self.proj(x), dim=1)

class LayerNorm(nn.Module):
    # Facilitate LayerNorm within the transformer model
    def __init__(self, features, eps=1e-6):
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return self.a_2 * ( (x - mean) / (torch.sqrt(std + self.eps)) ) + self.b_2

class SublayerConnection(nn.Module):
    # Residual connection followed by LayerNorm:
    def __init__(self, size, dropout_prob=0.25):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn: nn.Module, feed_forward: nn.Module, dropout_prob):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.dropout = nn.Dropout(dropout_prob)
        self.sublayer = clones(SublayerConnection(size, dropout_prob), 2)
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer: nn.Module, N):
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
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self):
        pass


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, target_embed, generator):
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.generator = generator
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, mem, target, src_mask, target_mask):
        return self.decode(self.target_embed(target), mem, src_mask, target_mask)
    

