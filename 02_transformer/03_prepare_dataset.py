from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Iterable, List

SPECIAL_IDS = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
src_lang = 'de'
tgt_lang = 'en'


def prepare_dataset(batch_size: int):
    func_token = {}
    func_vocabs = {}
    func_t2i = {}

    url = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/"
    multi30k.URL["train"] = url + "training.tar.gz"
    multi30k.URL["valid"] = url + "validation.tar.gz"

    # 1.定义 tokenizer
    func_token[src_lang] = get_tokenizer('spacy', language='de_core_news_sm')
    func_token[tgt_lang] = get_tokenizer('spacy', language='en_core_web_sm')

    # 将所有数据处理方法整合到一起
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    # 2.挨个抛出token，方便做成字典。
    def yield_tokens(data: Iterable, language: str) -> List[str]:
        i = 0 if language == src_lang else 1
        for da in data:
            yield func_token[language](da[i])

    # 3.首尾拼接特殊字符
    def tensor_transform(token_ids: List[int]):
        return torch.cat((torch.tensor([SPECIAL_IDS['<bos>']]),
                          torch.tensor(token_ids),
                          torch.tensor([SPECIAL_IDS['<eos>']])))

    # 4.填充<pad>
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(func_t2i[src_lang](src_sample.rstrip("\n")))
            tgt_batch.append(func_t2i[tgt_lang](tgt_sample.rstrip("\n")))
        # batch_first 要 true
        src_batch = pad_sequence(src_batch, padding_value=SPECIAL_IDS['<pad>'], batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=SPECIAL_IDS['<pad>'], batch_first=True)
        return src_batch, tgt_batch

    # 从训练集里抽取字典和文本转int的方法。
    for lang in [src_lang, tgt_lang]:
        train_iter = Multi30k(split='train', language_pair=(src_lang, tgt_lang))
        func_vocabs[lang] = build_vocab_from_iterator(yield_tokens(train_iter, lang),
                                                      min_freq=1,
                                                      specials=list(SPECIAL_IDS.keys()),
                                                      special_first=True
                                                      )
        func_vocabs[lang].set_default_index(SPECIAL_IDS['<unk>'])
        func_t2i[lang] = sequential_transforms(func_token[lang], func_vocabs[lang], tensor_transform)

    train = Multi30k(split='train', language_pair=(src_lang, tgt_lang))
    train_dataloader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn)
    val = Multi30k(split='valid', language_pair=(src_lang, tgt_lang))
    val_dataloader = DataLoader(val, batch_size=batch_size, collate_fn=collate_fn)

    return func_t2i, func_vocabs, train_dataloader, val_dataloader
