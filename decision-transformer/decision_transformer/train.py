import torch
import copy
from .annotated_transformer import (EncoderDecoder, Encoder, EncoderLayer, Decoder, 
                                    MultiheadedAttention, Generator, PositionWiseFFN,
                                    DecoderLayer, PositionalEmbedding, SublayerConnection)
from torch.nn import Sequential, Embedding

def instantiate_model(src_vocab, tgt_vocab, attn_heads=6, ctx_size=2048, d_embed=512, d_feedforward=512, dropout=0.1):
    
    positional_encoding = PositionalEmbedding(d_embed, dropout)
    attn = MultiheadedAttention(attn_heads, d_embed)
    feedforward_nn = PositionWiseFFN(d_embed, dropout)
    return EncoderDecoder(
        Encoder(EncoderLayer(copy.deepcopy(attn), copy.deepcopy(feedforward_nn), copy.deepcopy(dropout)), attn_heads),
        Decoder(DecoderLayer(copy.deepcopy(attn), copy.deepcopy(feedforward_nn), copy.deepcopy(dropout)), attn_heads),
        Sequential(Embedding(d_embed, src_vocab), copy.deepcopy(positional_encoding)),
        Sequential(Embedding(d_embed, tgt_vocab), copy.deepcopy(positional_encoding)),
        Generator(d_embed, tgt_vocab)
    )

if __name__ == '__main__':
    instantiate_model()
