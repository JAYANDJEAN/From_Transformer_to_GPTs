import numpy as np
import matplotlib.pyplot as plt
import torch
from modelsummary import summary
from models import (PositionalEncoding, ScaleDotProductAttention,
                    MultiHeadAttention, EncoderLayer, DecoderLayer,
                    TransformerScratch, TransformerTorch)
from utils import generate_mask, prepare_dataset, SPECIAL_IDS, src_lang, tgt_lang, generate
import torch.nn as nn
from tokenizers import Tokenizer
import sentencepiece as spm

import warnings


# warnings.filterwarnings("ignore")


# ----------------------transformer scratch----------------------
def check_positional_encoding():
    def positional_encoding_loop(max_len, d_model):
        pos_enc = np.zeros((max_len, d_model))
        for k in range(max_len):
            for i in range(0, d_model, 2):
                pos_enc[k, i] = np.sin(k / (100 ** ((2 * i) / d_model)))
                pos_enc[k, i + 1] = np.cos(k / (100 ** ((2 * i) / d_model)))
        return pos_enc

    pe1 = PositionalEncoding(MAX_LEN, D_MODEL, 0.1, batch_first=False)
    pe2 = PositionalEncoding(MAX_LEN, D_MODEL, 0.1, batch_first=True)
    print('check trainable: ', pe1.pos_embedding.requires_grad)
    pos_enc_list = [positional_encoding_loop(MAX_LEN, D_MODEL),
                    pe1.pos_embedding.squeeze().numpy(),
                    pe2.pos_embedding.squeeze().numpy()]
    title_list = ['Explicit Loop Positional Encoding',
                  'PyTorch Positional Encoding SRC_SEQ_LEN',
                  'PyTorch Positional Encoding batch_first']

    input_tensor = torch.randn(SRC_SEQ_LEN, BATCH_SIZE, D_MODEL)
    output_tensor = pe1(input_tensor)
    assert output_tensor.shape == input_tensor.shape

    input_tensor = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)
    output_tensor = pe2(input_tensor)
    assert output_tensor.shape == input_tensor.shape

    plt.figure(figsize=(10, 8))
    for i in range(len(pos_enc_list)):
        plt.subplot(2, 2, 1 + i)
        plt.imshow(pos_enc_list[i].T, cmap='viridis', aspect='auto', origin='lower')
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title(title_list[i])
        plt.colorbar()

    plt.tight_layout()
    plt.savefig('../00_assets/image/PE.png')


def check_scale_dot_product_attention():
    attention = ScaleDotProductAttention()
    SRC_SEQ_LEN = 5

    q = torch.randn(BATCH_SIZE, N_HEAD, SRC_SEQ_LEN, D_MODEL)
    k = torch.randn(BATCH_SIZE, N_HEAD, SRC_SEQ_LEN, D_MODEL)
    v = torch.randn(BATCH_SIZE, N_HEAD, SRC_SEQ_LEN, D_MODEL)
    attended_values, attention_weights = attention(q, k, v)
    assert attended_values.shape == (BATCH_SIZE, N_HEAD, SRC_SEQ_LEN, D_MODEL)
    assert attention_weights.shape == (BATCH_SIZE, N_HEAD, SRC_SEQ_LEN, SRC_SEQ_LEN)
    print('weights before mask:')
    print(attention_weights[0, 0, :, :])

    mask = generate_mask(SRC_SEQ_LEN)
    print('mask tensor is: ')
    print(mask)

    attended_values_masked, attention_weights_masked = attention(q, k, v, mask)

    assert attended_values_masked.shape == (BATCH_SIZE, N_HEAD, SRC_SEQ_LEN, D_MODEL)
    assert attention_weights_masked.shape == (BATCH_SIZE, N_HEAD, SRC_SEQ_LEN, SRC_SEQ_LEN)
    print('weights after mask:')
    print(attention_weights_masked[0, 0, :, :])

    from torch.nn.functional import scaled_dot_product_attention
    print('check result')
    print(scaled_dot_product_attention(q, k, v, attn_mask=mask)[0, 0, :, :5])
    print(attended_values_masked[0, 0, :, :5])


def check_multi_head_attention():
    model = MultiHeadAttention(D_MODEL, N_HEAD)
    # Generate random input tensors
    q = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)
    k = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)
    v = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)
    out = model(q, k, v)
    assert out.shape == (BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)


def check_encoder_layer():
    model = EncoderLayer(D_MODEL, DIM_FF, N_HEAD, DROPOUT)
    src_emb = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)
    src_mask = torch.ones(SRC_SEQ_LEN, SRC_SEQ_LEN)
    out = model(src_emb, src_mask)
    assert out.shape == (BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)


def check_decoder_layer():
    model = DecoderLayer(D_MODEL, DIM_FF, N_HEAD, DROPOUT)
    src_emb = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)
    tgt_emb = torch.randn(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)
    tgt_mask = torch.ones(TGT_SEQ_LEN, TGT_SEQ_LEN)

    out = model(tgt_emb, src_emb, tgt_mask)
    assert out.shape == (BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)


def check_transformer():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    src_vocab_size = 1000
    tgt_vocab_size = 1200
    model_scratch = TransformerScratch(3, 3, D_MODEL, 8, src_vocab_size, tgt_vocab_size)
    model_torch = TransformerTorch(3, 3, D_MODEL, 8, src_vocab_size, tgt_vocab_size)
    print("Total trainable parameters:", count_parameters(model_scratch))
    print("Total trainable parameters:", count_parameters(model_torch))

    src = torch.randint(low=0, high=100, size=(BATCH_SIZE, SRC_SEQ_LEN), dtype=torch.int)
    tgt = torch.randint(low=0, high=100, size=(BATCH_SIZE, TGT_SEQ_LEN), dtype=torch.int)
    src_mask = torch.randn(SRC_SEQ_LEN, SRC_SEQ_LEN)
    tgt_mask = torch.randn(TGT_SEQ_LEN, TGT_SEQ_LEN)
    src_padding_mask = torch.randn(BATCH_SIZE, SRC_SEQ_LEN)
    tgt_padding_mask = torch.randn(BATCH_SIZE, TGT_SEQ_LEN)

    print('-------model_scratch---------')
    summary(model_scratch, src, tgt, src_mask, tgt_mask, show_input=True)

    out1 = model_scratch(src, tgt, src_mask, tgt_mask)
    out2 = model_torch(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
    assert out1.shape == (BATCH_SIZE, TGT_SEQ_LEN, tgt_vocab_size)
    assert out2.shape == (BATCH_SIZE, TGT_SEQ_LEN, tgt_vocab_size)


# ----------------------transformer torch----------------------

def check_padding_mask():
    encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, batch_first=True)
    text_to_indices, vocabs, train_loader, eval_loader = prepare_dataset(BATCH_SIZE)

    _, (src, tgt) = next(enumerate(train_loader))
    seq_len = src.shape[1]
    src_emb = torch.randn(BATCH_SIZE, seq_len, D_MODEL)
    src_mask = torch.zeros((seq_len, seq_len))
    src_padding_mask = (src == SPECIAL_IDS['<pad>'])

    print(src.shape)
    print(src_padding_mask.shape)
    # 很明显，加与不加src_padding_mask是有区别的，但对最终的结果影响大吗？
    out = encoder_layer(src_emb, src_mask, src_padding_mask)
    print(out[0, 0, 0:5])
    out = encoder_layer(src_emb, src_mask, None)
    print(out[0, 0, 0:5])


def check_data():
    print('--------------------------data------------------------------------')
    '''
    text_to_indices: 将文本转成编号序列
    vocabs: 字典 src_size: 19214 tgt_size: 10837
    pip install -U spacy
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    '''
    text_to_indices, vocabs, train_loader, eval_loader = prepare_dataset(BATCH_SIZE)
    src_size, tgt_size = len(vocabs[src_lang]), len(vocabs[tgt_lang])
    print(src_size, tgt_size)
    _, (src, tgt) = next(enumerate(train_loader))
    print('src size: ', src.shape)
    print('tgt size: ', tgt.shape)
    print('src first sentence:')
    print(src[0, :])
    print(text_to_indices['de']('Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'))
    print('tgt first sentence:')
    print(tgt[0, :])
    print(text_to_indices['en']('Two young, White males are outside near many bushes.'))
    print('english vocab:')
    for i in range(5):
        print(i, vocabs['en'].lookup_token(i))

    '''
    去尾、掐头
    在这段代码中，tgt_input是目标序列的输入，tgt_out是目标序列的输出。
    这样做的目的是将模型的输入和输出对齐起来，使得模型在生成下一个符号时可以根据已生成的符号来预测下一个符号。
    假设目标序列是[<start>, a, b, c, <end>]，其中<start>表示序列的开始，<end>表示序列的结束。
    那么tgt_input就是[<start>, a, b, c]，而tgt_out就是[a, b, c, <end>]。
    这样一来，模型在预测第一个符号时可以根据[<start>]来生成a，在预测第二个符号时可以根据[<start>, a]来生成b，以此类推。
    这种处理方式在训练时可以更好地利用模型的输出来指导模型的训练，提高模型的性能。
    '''
    tgt_input, tgt_out = tgt[:, :-1], tgt[:, 1:]

    print('-------------------------mask-------------------------------')
    '''
        Shape:
            S is the source sequence length, 
            T is the target sequence length, 
            N is the batch size, 
            E is the feature number

            src: (S, E) for unbatched input, (S, N, E) if batch_first=False or (N, S, E) if batch_first=True.
            tgt: (T, E) for unbatched input, (T, N, E) if batch_first=False or (N, T, E) if batch_first=True.
            src_mask: (S, S) or (N⋅num_heads, S, S).
            tgt_mask: (T, T) or (N⋅num_heads, T, T).
            memory_mask: (T, S).
            src_key_padding_mask: (S) for unbatched input otherwise (N, S).
            tgt_key_padding_mask: (T) for unbatched input otherwise (N, T).
            memory_key_padding_mask: (S) for unbatched input otherwise (N, S).
            output: (T, E) for unbatched input, (T, N, E) if batch_first=False or (N, T, E) if batch_first=True.
        '''
    src_mask = torch.zeros((src.shape[1], src.shape[1]))
    tgt_mask = generate_mask(tgt_input.shape[1])
    src_padding_mask = (src == SPECIAL_IDS['<pad>'])
    tgt_padding_mask = (tgt_input == SPECIAL_IDS['<pad>'])
    print(src_mask.shape)  # (S, S)
    print(tgt_mask.shape)  # (T, T)
    print(src_padding_mask.shape)  # (N, S)
    print(tgt_padding_mask.shape)  # (N, T)

    print('-------------------------transformer-------------------------------')
    '''
    原始的句子的shape是(S, N)，经过embedding，是(S, N, E)，加上pos_embedding，依然是(S, N, E)
    '''

    # 模型定义，没什么好讲的
    transformer = TransformerTorch(num_encoder_layers=3,
                                   num_decoder_layers=3,
                                   d_model=D_MODEL,
                                   n_head=8,
                                   src_vocab_size=src_size,
                                   tgt_vocab_size=tgt_size
                                   )
    logits_predict = transformer(src, tgt_input, src_mask, tgt_mask,
                                 src_padding_mask, tgt_padding_mask,
                                 src_padding_mask)
    print('预测单例展示：')
    print('logits_predict size: ', logits_predict.shape)
    print(logits_predict[0, :, :].shape)
    token_predict = torch.argmax(logits_predict[0, :, :], dim=1)
    print(token_predict)

    print('----------------------------eval-------------------------------')
    memory = transformer.encode(src, src_mask)
    print(memory.shape)  # (N, S, E)


# 对比后选择tokenizers
def check_tokenizers():
    tokenizer_de = Tokenizer.from_file("./bpe_tokenizer/token-de.json")
    tokenizer_en = Tokenizer.from_file("./bpe_tokenizer/token-en.json")
    text_en = "A man in a blue shirt is standing on a ladder cleaning a window."

    print('de vocab size: ', tokenizer_de.get_vocab_size())
    output = tokenizer_de.encode("Ein kleines Mädchen klettert in ein Spielhaus aus Holz.")
    print([tokenizer_de.decode([i]) for i in output.ids])

    print('en vocab size: ', tokenizer_en.get_vocab_size())
    output = tokenizer_en.encode(text_en)
    print([tokenizer_en.decode([i]) for i in output.ids])

    print(tokenizer_de.id_to_token(0))
    print(tokenizer_de.id_to_token(1))
    print(tokenizer_de.id_to_token(2))
    print(tokenizer_de.id_to_token(3))
    print(tokenizer_de.token_to_id("<eos>"))


def check_sentencepiece():
    sp_de = spm.SentencePieceProcessor()
    sp_en = spm.SentencePieceProcessor()

    sp_de.load('./bpe_tokenizer/sp-de.model')
    sp_en.load('./bpe_tokenizer/sp-en.model')

    print('de vocab size: ', sp_de.vocab_size())
    output = sp_de.encode_as_ids("Ein kleines Mädchen klettert in ein Spielhaus aus Holz.")
    print([sp_de.decode_ids(i) for i in output])

    print('en vocab size: ', sp_en.vocab_size())
    output = sp_en.encode_as_ids("A man in a blue shirt is standing on a ladder cleaning a window.")
    print([sp_en.decode_ids(i) for i in output])

    print(sp_de.bos_id())
    print(sp_de.eos_id())
    print(sp_de.pad_id())
    print(sp_de.unk_id())


if __name__ == '__main__':
    MAX_LEN = 100
    D_MODEL = 512
    BATCH_SIZE = 64
    SRC_SEQ_LEN = 23
    TGT_SEQ_LEN = 17
    N_HEAD = 8
    DIM_FF = 256
    DROPOUT = 0.1

    # check_positional_encoding()
    # check_scale_dot_product_attention()
    # check_multi_head_attention()
    # check_encoder_layer()
    # check_decoder_layer()
    # check_transformer()
    # check_data()
    # check_translate()

    check_sentencepiece()
