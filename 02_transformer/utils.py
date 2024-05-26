import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Iterable, List, Dict
import torch
from torch.utils.data import Dataset

SPECIAL_IDS = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
src_lang = 'de'
tgt_lang = 'en'


# def prepare_dataset(batch_size: int):
#     func_token = {}
#     func_vocabs = {}
#     func_t2i = {}
#
#     # url = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/"
#     # multi30k.URL["train"] = url + "training.tar.gz"
#     # multi30k.URL["valid"] = url + "validation.tar.gz"
#
#     # 1.定义 tokenizer
#     func_token[src_lang] = get_tokenizer('spacy', language='de_core_news_sm')
#     func_token[tgt_lang] = get_tokenizer('spacy', language='en_core_web_sm')
#
#     # 将所有数据处理方法整合到一起
#     def sequential_transforms(*transforms):
#         def func(txt_input):
#             for transform in transforms:
#                 txt_input = transform(txt_input)
#             return txt_input
#
#         return func
#
#     # 2.挨个抛出token，方便做成字典。
#     def yield_tokens(data: Iterable, language: str) -> List[str]:
#         i = 0 if language == src_lang else 1
#         for da in data:
#             yield func_token[language](da[i])
#
#     # 3.首尾拼接特殊字符
#     def tensor_transform(token_ids: List[int]):
#         return torch.cat((torch.tensor([SPECIAL_IDS['<bos>']]),
#                           torch.tensor(token_ids),
#                           torch.tensor([SPECIAL_IDS['<eos>']])))
#
#     # 4.填充<pad>
#     def collate_fn(batch):
#         src_batch, tgt_batch = [], []
#         for src_sample, tgt_sample in batch:
#             src_batch.append(func_t2i[src_lang](src_sample.rstrip("\n")))
#             tgt_batch.append(func_t2i[tgt_lang](tgt_sample.rstrip("\n")))
#         # batch_first 要 true
#         src_batch = pad_sequence(src_batch, padding_value=SPECIAL_IDS['<pad>'], batch_first=True)
#         tgt_batch = pad_sequence(tgt_batch, padding_value=SPECIAL_IDS['<pad>'], batch_first=True)
#         return src_batch, tgt_batch
#
#     # 从训练集里抽取字典和文本转int的方法。
#     for lang in [src_lang, tgt_lang]:
#         train_iter = Multi30k(split='train', language_pair=(src_lang, tgt_lang))
#         func_vocabs[lang] = build_vocab_from_iterator(yield_tokens(train_iter, lang),
#                                                       min_freq=1,
#                                                       specials=list(SPECIAL_IDS.keys()),
#                                                       special_first=True
#                                                       )
#         func_vocabs[lang].set_default_index(SPECIAL_IDS['<unk>'])
#         func_t2i[lang] = sequential_transforms(func_token[lang], func_vocabs[lang], tensor_transform)
#
#     train = Multi30k(split='train', language_pair=(src_lang, tgt_lang))
#     train_dataloader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn)
#     val = Multi30k(split='valid', language_pair=(src_lang, tgt_lang))
#     val_dataloader = DataLoader(val, batch_size=batch_size, collate_fn=collate_fn)
#
#     return func_t2i, func_vocabs, train_dataloader, val_dataloader
#

def generate_mask(sz: int):
    # 生成一个下三角矩阵
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def generate(model: torch.nn.Module, src_sentences: List[str], text_to_indices: Dict, vocabs: Dict, device):
    model.eval()
    # text_to_indices 相当于 tokenizer的encode
    # vocabs 相当于 tokenizer的decode
    src_tensor = [text_to_indices[src_lang](src) for src in src_sentences]
    src_tensor = pad_sequence(src_tensor, padding_value=SPECIAL_IDS['<pad>'], batch_first=True)
    batch_size, seq_len = src_tensor.shape
    src_mask = (torch.zeros(seq_len, seq_len)).to(device)
    src_padding_mask = (src_tensor == SPECIAL_IDS['<pad>']).to(device)
    # memory: torch.Size([batch_size, seq_len, d_model])
    memory = model.encode(src_tensor, src_mask, src_padding_mask).to(device)
    generate_words = torch.ones(batch_size, 1).fill_(SPECIAL_IDS['<bos>']).long().to(device)
    eos_id = SPECIAL_IDS['<eos>']
    for i in range(seq_len + 5):
        # 这里的 tgt_mask 必须要加上！
        tgt_mask = (generate_mask(generate_words.size(1))).to(device)
        # out: torch.Size([batch_size, seq_len, d_model])
        out = model.decode(generate_words, memory, tgt_mask, None)
        # 取最后一个词的embedding，然后计算这个词的概率分布
        prob = model.generator(out[:, -1, :])
        # next_word: torch.Size([batch_size, 1])
        next_word = torch.argmax(prob, dim=1).unsqueeze(1)
        generate_words = torch.cat([generate_words, next_word], dim=1)
        if torch.all(next_word == eos_id):
            break
    out_text = []
    for _, tokens in enumerate(generate_words.tolist()):
        if SPECIAL_IDS['<eos>'] in tokens:
            eos_idx = tokens.index(eos_id)
            tokens = tokens[:eos_idx]
        tgt_words = vocabs[tgt_lang].lookup_tokens(tokens)
        out_text.append(" ".join([i for i in tgt_words if i not in SPECIAL_IDS.keys()]))
    return out_text


class TextDataset(Dataset):
    def __init__(self, dt):
        assert dt in ('train', 'val')
        german_file = f"../00_assets/de_en/{dt}.de"
        english_file = f"../00_assets/de_en/{dt}.en"
        with open(german_file, 'r', encoding='utf-8') as f_de, \
                open(english_file, 'r', encoding='utf-8') as f_en:
            self.de_texts = f_de.readlines()
            self.en_texts = f_en.readlines()

        assert len(self.de_texts) == len(
            self.en_texts), "German and English files should have the same number of lines."

    def __len__(self):
        return len(self.de_texts)

    def __getitem__(self, idx):
        de_text = self.de_texts[idx].strip()
        en_text = self.en_texts[idx].strip()
        return de_text, en_text
