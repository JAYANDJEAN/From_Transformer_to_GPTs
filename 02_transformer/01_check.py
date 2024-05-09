import numpy as np
import matplotlib.pyplot as plt
import torch
from modelsummary import summary
from models import (PositionalEncoding, ScaleDotProductAttention,
                    MultiHeadAttention, EncoderLayer, DecoderLayer,
                    TransformerScratch, TransformerTorch)
from utils import generate_mask, prepare_dataset, SPECIAL_IDS, src_lang, tgt_lang, translate
import warnings

warnings.filterwarnings("ignore")


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
    print(pe1.pos_embedding.requires_grad)
    pos_enc_list = [positional_encoding_loop(MAX_LEN, D_MODEL),
                    pe1.pos_embedding.squeeze().numpy(),
                    pe2.pos_embedding.squeeze().numpy()]
    title_list = ['Explicit Loop Positional Encoding',
                  'PyTorch Positional Encoding seq_len',
                  'PyTorch Positional Encoding batch_first']

    input_tensor = torch.randn(SEQ_LEN, BATCH_SIZE, D_MODEL)
    output_tensor = pe1(input_tensor)
    assert output_tensor.shape == input_tensor.shape

    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
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
    SEQ_LEN = 5

    q = torch.randn(BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    k = torch.randn(BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    v = torch.randn(BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    attended_values, attention_weights = attention(q, k, v)
    assert attended_values.shape == (BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    assert attention_weights.shape == (BATCH_SIZE, N_HEAD, SEQ_LEN, SEQ_LEN)
    print('weights before mask:')
    print(attention_weights[0, 0, :, :])

    mask = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), float('-inf')), diagonal=1)
    print('mask tensor is: ')
    print(mask)

    attended_values_masked, attention_weights_masked = attention(q, k, v, mask)

    assert attended_values_masked.shape == (BATCH_SIZE, N_HEAD, SEQ_LEN, D_MODEL)
    assert attention_weights_masked.shape == (BATCH_SIZE, N_HEAD, SEQ_LEN, SEQ_LEN)
    print('weights after mask:')
    print(attention_weights_masked[0, 0, :, :])

    from torch.nn.functional import scaled_dot_product_attention
    print('check result')
    print(scaled_dot_product_attention(q, k, v, attn_mask=mask)[0, 0, :, :5])
    print(attended_values_masked[0, 0, :, :5])


def check_multi_head_attention():
    model = MultiHeadAttention(D_MODEL, N_HEAD)
    # Generate random input tensors
    q = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    k = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    v = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    out = model(q, k, v)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def check_encoder_layer():
    model = EncoderLayer(D_MODEL, DIM_FF, N_HEAD, DROPOUT)
    src_emb = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    src_mask = torch.ones(SEQ_LEN, SEQ_LEN)
    out = model(src_emb, src_mask)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def check_decoder_layer():
    model = DecoderLayer(D_MODEL, DIM_FF, N_HEAD, DROPOUT)
    src_emb = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    tgt_emb = torch.randn(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)
    tgt_mask = torch.ones(TGT_SEQ_LEN, TGT_SEQ_LEN)

    out = model(tgt_emb, src_emb, tgt_mask)
    print(out.shape)


def check_transformer():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    src_vocab_size = 1000
    tgt_vocab_size = 1200
    model_scratch = TransformerScratch(3, 3, D_MODEL, 8, src_vocab_size, tgt_vocab_size)
    model_torch = TransformerTorch(3, 3, D_MODEL, 8, src_vocab_size, tgt_vocab_size)
    print("Total trainable parameters:", count_parameters(model_scratch))
    print("Total trainable parameters:", count_parameters(model_torch))

    src = torch.randint(low=0, high=100, size=(BATCH_SIZE, SEQ_LEN), dtype=torch.int)
    tgt = torch.randint(low=0, high=100, size=(BATCH_SIZE, TGT_SEQ_LEN), dtype=torch.int)
    src_mask = torch.randn(SEQ_LEN, SEQ_LEN)
    tgt_mask = torch.randn(TGT_SEQ_LEN, TGT_SEQ_LEN)
    src_padding_mask = torch.randn(BATCH_SIZE, SEQ_LEN)
    tgt_padding_mask = torch.randn(BATCH_SIZE, TGT_SEQ_LEN)

    print('-------model_scratch---------')
    summary(model_scratch, src, tgt, src_mask, tgt_mask, show_input=True)

    out1 = model_scratch(src, tgt, src_mask, tgt_mask)
    out2 = model_torch(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
    assert out1.shape == (BATCH_SIZE, TGT_SEQ_LEN, tgt_vocab_size)
    assert out2.shape == (BATCH_SIZE, TGT_SEQ_LEN, tgt_vocab_size)


def check_data():
    print('--------------------------data------------------------------------')
    '''
    text_to_indices: 将文本转成编号序列
    vocabs: 字典
    pip install -U spacy
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    '''
    text_to_indices, vocabs, train_loader, eval_loader = prepare_dataset(BATCH_SIZE)
    src_size, tgt_size = len(vocabs[src_lang]), len(vocabs[tgt_lang])
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


def check_translate():
    src_ = "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche."
    t2i, voc, train_loader, eval_loader = prepare_dataset(128)
    transformer = TransformerTorch(num_encoder_layers=3,
                                   num_decoder_layers=3,
                                   d_model=512,
                                   n_head=8,
                                   src_vocab_size=len(voc[src_lang]),
                                   tgt_vocab_size=len(voc[tgt_lang])
                                   ).to('cpu')

    print("Translated sentence:", translate(transformer, src_, t2i, voc, 'cpu'))


if __name__ == '__main__':
    MAX_LEN = 100
    D_MODEL = 512
    BATCH_SIZE = 64
    SEQ_LEN = 23
    N_HEAD = 8
    DIM_FF = 256
    DROPOUT = 0.1
    TGT_SEQ_LEN = 17

    check_positional_encoding()
    check_scale_dot_product_attention()
    check_multi_head_attention()
    check_encoder_layer()
    check_decoder_layer()
    check_transformer()
    check_data()
    check_translate()
