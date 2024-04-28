from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Iterable, List, Dict

SPECIAL_IDS = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}


# batch_first = False

def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# 需要加入batch_first参数
def create_mask(src: Tensor, tgt: Tensor):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    src_padding_mask = (src == SPECIAL_IDS['<pad>'])
    tgt_padding_mask = (tgt == SPECIAL_IDS['<pad>'])
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def translate(model: torch.nn.Module, src_sentence: str, text_to_indices: Dict, vocabs: Dict, device):
    model.eval()
    src_lang = 'de'
    tgt_lang = 'en'
    src_tensor = text_to_indices[src_lang](src_sentence).view(-1, 1).to(device)
    num_tokens = src_tensor.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
    max_len = num_tokens + 5
    start_symbol = SPECIAL_IDS['<bos>']

    memory = model.encode(src_tensor, src_mask).to(device)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=0)
        if next_word == SPECIAL_IDS['<eos>']:
            break

    tgt_tokens = ys.flatten()
    return " ".join(vocabs[tgt_lang].lookup_tokens(
        list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


def get_data(batch_size: int):
    func_token = {}
    func_vocabs = {}
    func_t2i = {}
    src_l = 'de'
    tgt_l = 'en'
    url = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/"
    multi30k.URL["train"] = url + "training.tar.gz"
    multi30k.URL["valid"] = url + "validation.tar.gz"
    func_token[src_l] = get_tokenizer('spacy', language='de_core_news_sm')
    func_token[tgt_l] = get_tokenizer('spacy', language='en_core_web_sm')

    # helper function to club together sequential operations
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    # helper function to yield list of tokens
    def yield_tokens(data: Iterable, language: str) -> List[str]:
        i = 0 if language == src_l else 1
        for da in data:
            yield func_token[language](da[i])

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(token_ids: List[int]):
        return torch.cat((torch.tensor([SPECIAL_IDS['<bos>']]),
                          torch.tensor(token_ids),
                          torch.tensor([SPECIAL_IDS['<eos>']])))

    # function to collate data samples into batch tensors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(func_t2i[src_l](src_sample.rstrip("\n")))
            tgt_batch.append(func_t2i[tgt_l](tgt_sample.rstrip("\n")))
        src_batch = pad_sequence(src_batch, padding_value=SPECIAL_IDS['<pad>']).transpose(0, 1)
        tgt_batch = pad_sequence(tgt_batch, padding_value=SPECIAL_IDS['<pad>']).transpose(0, 1)
        return src_batch, tgt_batch

    for lang in [src_l, tgt_l]:
        train_iter = Multi30k(split='train', language_pair=(src_l, tgt_l))
        func_vocabs[lang] = build_vocab_from_iterator(yield_tokens(train_iter, lang),
                                                      min_freq=1,
                                                      specials=list(SPECIAL_IDS.keys()),
                                                      special_first=True
                                                      )
        func_vocabs[lang].set_default_index(SPECIAL_IDS['<unk>'])
        func_t2i[lang] = sequential_transforms(func_token[lang], func_vocabs[lang], tensor_transform)

    train = Multi30k(split='train', language_pair=(src_l, tgt_l))
    train_dataloader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn)
    val = Multi30k(split='valid', language_pair=(src_l, tgt_l))
    val_dataloader = DataLoader(val, batch_size=batch_size, collate_fn=collate_fn)

    return func_t2i, func_vocabs, train_dataloader, val_dataloader
