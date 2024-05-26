import torch
from torch.nn.utils.rnn import pad_sequence

from typing import List, Dict
from torch.utils.data import Dataset

SPECIAL_IDS = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
src_lang = 'de'
tgt_lang = 'en'


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
