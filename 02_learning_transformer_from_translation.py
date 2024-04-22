from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import math
from timeit import default_timer as timer
from typing import Iterable, List, Dict

SPECIAL_IDS = {'UNK_IDX': 0, 'PAD_IDX': 1, 'BOS_IDX': 2, 'EOS_IDX': 3}
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: Tensor, tgt: Tensor):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    src_padding_mask = (src == SPECIAL_IDS['PAD_IDX']).transpose(0, 1)
    tgt_padding_mask = (tgt == SPECIAL_IDS['PAD_IDX']).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def get_data(batch_size: int):
    tokens = {}
    vocabs = {}
    text_indices = {}
    src_lang = 'de'
    tgt_lang = 'en'

    url = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/"
    multi30k.URL["train"] = url + "training.tar.gz"
    multi30k.URL["valid"] = url + "validation.tar.gz"
    tokens[src_lang] = get_tokenizer('spacy', language='de_core_news_sm')
    tokens[tgt_lang] = get_tokenizer('spacy', language='en_core_web_sm')

    # helper function to yield list of tokens
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
        language_index = {src_lang: 0, tgt_lang: 1}
        for data_sample in data_iter:
            yield tokens[language](data_sample[language_index[language]])

    # helper function to club together sequential operations
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(token_ids: List[int]):
        return torch.cat((torch.tensor([SPECIAL_IDS['BOS_IDX']]),
                          torch.tensor(token_ids),
                          torch.tensor([SPECIAL_IDS['EOS_IDX']])))

    for lang in [src_lang, tgt_lang]:
        train_iter = Multi30k(split='train', language_pair=(src_lang, tgt_lang))
        vocabs[lang] = build_vocab_from_iterator(yield_tokens(train_iter, lang),
                                                 min_freq=1,
                                                 specials=SPECIAL_SYMBOLS,
                                                 special_first=True)
        # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
        vocabs[lang].set_default_index(SPECIAL_IDS['UNK_IDX'])
        text_indices[lang] = sequential_transforms(tokens[lang], vocabs[lang], tensor_transform)

    # function to collate data samples into batch tensors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_indices[src_lang](src_sample.rstrip("\n")))
            tgt_batch.append(text_indices[tgt_lang](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=SPECIAL_IDS['PAD_IDX'])
        tgt_batch = pad_sequence(tgt_batch, padding_value=SPECIAL_IDS['PAD_IDX'])
        return src_batch, tgt_batch

    train_iter = Multi30k(split='train', language_pair=(src_lang, tgt_lang))
    train_ = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
    val_iter = Multi30k(split='valid', language_pair=(src_lang, tgt_lang))
    val_ = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)

    return text_indices, vocabs, train_, val_, len(vocabs[src_lang]), len(vocabs[tgt_lang])


def translate(model: torch.nn.Module, src_sentence: str, text_tensor: Dict, vocabs: Dict):
    model.eval()
    src_lang = 'de'
    tgt_lang = 'en'
    src_tensor = text_tensor[src_lang](src_sentence).view(-1, 1).to(DEVICE)
    num_tokens = src_tensor.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(DEVICE)
    max_len = num_tokens + 5
    start_symbol = SPECIAL_IDS['BOS_IDX']

    memory = model.encode(src_tensor, src_mask).to(DEVICE)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=0)
        if next_word == SPECIAL_IDS['EOS_IDX']:
            break

    tgt_tokens = ys.flatten()
    return " ".join(vocabs[tgt_lang].lookup_tokens(
        list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 n_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=n_head,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.initialize()

    def initialize(self):
        for para in self.parameters():
            if para.dim() > 1:
                nn.init.xavier_uniform_(para)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


def train_ops():
    def _epoch(model, dataloader, tp):
        losses = 0
        if tp == 'train':
            model.train()
        elif tp == 'eval':
            model.eval()

        for src, tgt in dataloader:
            tgt_input = tgt[:-1, :].to(DEVICE)
            tgt_out = tgt[1:, :].to(DEVICE)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits_pred = model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
            if tp == 'train':
                optimizer.zero_grad()
                loss = loss_fn(logits_pred.reshape(-1, logits_pred.shape[-1]), tgt_out.reshape(-1))
                loss.backward()
                optimizer.step()
                losses += loss.item()
            elif tp == 'eval':
                loss = loss_fn(logits_pred.reshape(-1, logits_pred.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()
        return losses / len(list(dataloader))

    BATCH_SIZE = 128
    NUM_EPOCHS = 18

    text_tensor, vocab_lang, train_loader, \
        eval_loader, src_size, tgt_size = get_data(BATCH_SIZE)

    transformer = Seq2SeqTransformer(num_encoder_layers=3,
                                     num_decoder_layers=3,
                                     emb_size=512,
                                     n_head=8,
                                     src_vocab_size=src_size,
                                     tgt_vocab_size=tgt_size).to(DEVICE)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_IDS['PAD_IDX'])

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = _epoch(transformer, train_loader, 'train')
        end_time = timer()
        val_loss = _epoch(transformer, eval_loader, 'eval')
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
               f"Val loss: {val_loss:.3f}, "
               f"Epoch time = {(end_time - start_time):.3f}s"))

    src_sentence = "Eine Gruppe von Menschen steht vor einem Iglu ."
    print(translate(transformer, src_sentence, text_tensor, vocab_lang))


def show_parameters():
    BATCH_SIZE = 128
    text_tensor, vocab_lang, train_loader, eval_loader, src_size, tgt_size = get_data(BATCH_SIZE)
    print('德语字典大小(src_size): ', src_size)  # 字典大小
    print('德语字典大小(tgt_size): ', tgt_size)
    print('----------train--------------')
    _, (src, tgt) = next(enumerate(train_loader))
    print('输入单例展示：')
    print('src size: ', src.shape)  # torch.Size([27, 128]) 最长句子是包含27个token
    print('tgt size: ', tgt.shape)  # torch.Size([24, 128])
    print(src[:, 0])
    print(tgt[:, 0])

    # 去尾、掐头
    tgt_input = tgt[:-1, :].to(DEVICE)
    tgt_out = tgt[1:, :].to(DEVICE)

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
    transformer = Seq2SeqTransformer(num_encoder_layers=3,
                                     num_decoder_layers=3,
                                     emb_size=512,
                                     n_head=8,
                                     src_vocab_size=src_size,
                                     tgt_vocab_size=tgt_size).to(DEVICE)
    logits_pred = transformer(src, tgt_input, src_mask, tgt_mask,
                              src_padding_mask, tgt_padding_mask, src_padding_mask)
    print('预测单例展示：')
    print('logits_pred size: ', logits_pred.shape)
    print(logits_pred[:, 0, :])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_IDS['PAD_IDX'])
    tgt_out_single = tgt_out[:, 0].reshape(-1)
    logits_pred_single = logits_pred[:, 0, :].reshape(-1, logits_pred.shape[-1])
    loss = loss_fn(logits_pred_single, tgt_out_single)
    print('logits_pred_single size:', logits_pred_single.shape)
    print('tgt_out_single size:', tgt_out_single.shape)
    print(logits_pred_single)
    print(tgt_out_single)
    print(loss)

    print('--------------eval-----------------')
    memory = transformer.encode(src, src_mask).to(DEVICE)
    print(memory.shape)


# train_ops()

show_parameters()
