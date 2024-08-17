from .annotated_transformer import (EncoderDecoder, Encoder, EncoderLayer, Decoder, 
                                    MultiheadedAttention,
                                    DecoderLayer, PositionalEmbedding, SublayerConnection)


def instantiate_model(src_vocab, tgt_vocab, attn_heads=6, ctx_size=2048, d_embed=512, d_feedforward=512, dropout=0.1):
    multihead_attn = Multi
    
    return EncoderDecoder(
        Encoder(),
        Decoder(),
        # a seeded input embedding model
        # a seeded output embedding model

    )
