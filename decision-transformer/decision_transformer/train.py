import torch
from datasets import load_dataset
import copy
from annotated_transformer import (EncoderDecoder, Encoder, EncoderLayer, Decoder, 
                                    MultiheadedAttention, Generator, PositionWiseFFN,
                                    Embeddings, DecoderLayer, PositionalEmbedding)
from torch.nn import Sequential


books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(test_size=0.2)
print(books['train'][0])

def train():
    pass


def instantiate_model(src_vocab, tgt_vocab, attn_heads=6, d_embed=512, d_feedforward=2048, h=8, dropout=0.1):
    # h is the number of attention heads in one multiheaded attention module
    # d_feedforward is the intermediate output dimension of the feedforward network
    # d_embed is the dimension of the embedding vectors for the tokens in this model
    # src_vocab is the number of tokens in our source vocabulary (i.e. in encoder input)
    # tgt_vocab is the number of tokens in our target vocabulary (i.e. in decoder output)
    # THOROUGLY SCRUTINIZE TENSOR SIZES CAUSE WTF
    
    positional_encoding = PositionalEmbedding(d_embed, dropout)
    attn = MultiheadedAttention(h, d_embed)
    feedforward_nn = PositionWiseFFN(d_embed, d_feedforward, dropout)
    return EncoderDecoder(
        Encoder(EncoderLayer(d_embed, copy.deepcopy(attn), copy.deepcopy(feedforward_nn), copy.deepcopy(dropout)), attn_heads),
        Decoder(DecoderLayer(d_embed, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(feedforward_nn), copy.deepcopy(dropout)), attn_heads),
        Sequential(Embeddings(d_embed, src_vocab), copy.deepcopy(positional_encoding)),
        Sequential(Embeddings(d_embed, tgt_vocab), copy.deepcopy(positional_encoding)),
        Generator(d_embed, tgt_vocab)
    )

if __name__ == '__main__':
    instantiate_model(1024, 1024)
