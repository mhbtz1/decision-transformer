import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import copy
from vanilla_transformer import (EncoderDecoder, Encoder, EncoderLayer, Decoder, 
                                    MultiheadedAttention, Generator, PositionWiseFFN,
                                    Embeddings, DecoderLayer, PositionalEmbedding)
from torch.nn import Sequential
from tqdm import tqdm

books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(test_size=0.2)
print(books['train'][0:5])

tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-3.5-turbo")


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class Batch:
    """Object for holding a batch of data with mask during training."""

    # note: src's dim is (batch_size, sentence_length) and each row contains a tensor with numbers that are the index of the token for use by nn.Embedding
    # tgt's dim is (batch_size, sentence_length) also
    def __init__(self, src, target=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.target_mask = self.make_std_mask(self.target, pad)
            self.ntokens = (self.target_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


def train(model: EncoderDecoder, max_sentence_length: 5000, loss_fn):
    train_inputs = torch.tensor(len(books['train'], max_sentence_length))
    train_outputs = torch.tensor(len(books['train']), max_sentence_length)
    for i, data in tqdm(enumerate(books['train'])):
        input_sentence, result = tokenizer(data['translation']['en'], return_tensors='pt'), tokenizer(data['translation']['fr'], return_tensor='pt')
        input_token_ids, output_token_ids = input_sentence['input_ids'], result['input_ids']
        train_inputs[i, :] = input_token_ids
        train_outputs[i, :] =  output_token_ids

    b = Batch(train_inputs, train_outputs)
    for epoch in tqdm(range(1, 5)):
        result = model.forward(b.src, b.target, b.src_mask, b.target_mask)
        loss = loss_fn(result)
        loss.backward()

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


def mini_inference():
    test_model = instantiate_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        mini_inference()


if __name__ == '__main__':
    instantiate_model(1024, 1024)