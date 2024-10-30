import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import List
from tokenizers import Tokenizer

src_lang = 'de'
tgt_lang = 'en'
tokenizers = {src_lang: Tokenizer.from_file("../00_assets/tokenizers/en_de_tokenizers/token-de.json"),
              tgt_lang: Tokenizer.from_file("../00_assets/tokenizers/en_de_tokenizers/token-en.json")}


def generate_mask(sz: int):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def generate(model: torch.nn.Module, src_sentences: List[str], device):
    model.eval()
    src_batch = [torch.as_tensor([tokenizers[src_lang].token_to_id("<bos>")] +
                                 tokenizers[src_lang].encode(src).ids +
                                 [tokenizers[src_lang].token_to_id("<eos>")])
                 for src in src_sentences]
    src_tensor = pad_sequence(src_batch, padding_value=tokenizers[src_lang].token_to_id("<pad>"), batch_first=True)

    batch_size, seq_len = src_tensor.shape
    src_mask = (torch.zeros(seq_len, seq_len)).to(device)
    src_padding_mask = (src_tensor == tokenizers[src_lang].token_to_id("<pad>")).to(device)
    # memory: torch.Size([batch_size, seq_len, d_model])
    memory = model.encode(src_tensor, src_mask, src_padding_mask).to(device)
    generate_words = torch.ones(batch_size, 1).fill_(tokenizers[tgt_lang].token_to_id("<bos>")).long().to(device)
    eos_id = tokenizers[tgt_lang].token_to_id("<eos>")
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
    for _, ids in enumerate(generate_words.tolist()):
        if eos_id in ids:
            eos_idx = ids.index(eos_id)
            ids = ids[:eos_idx]
        tgt_words = tokenizers[tgt_lang].decode(ids)
        out_text.append(tgt_words)
    return out_text


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_text, tgt_text in batch:
        output_de = tokenizers[src_lang].encode(src_text)
        output_en = tokenizers[tgt_lang].encode(tgt_text)
        src_batch.append(torch.as_tensor([tokenizers[src_lang].token_to_id("<bos>")] +
                                         output_de.ids +
                                         [tokenizers[src_lang].token_to_id("<eos>")]))
        tgt_batch.append(torch.as_tensor([tokenizers[tgt_lang].token_to_id("<bos>")] +
                                         output_en.ids +
                                         [tokenizers[tgt_lang].token_to_id("<eos>")]))
    src_batch = pad_sequence(src_batch, padding_value=tokenizers[src_lang].token_to_id("<pad>"), batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tokenizers[tgt_lang].token_to_id("<pad>"), batch_first=True)
    return src_batch, tgt_batch


class TextDataset(Dataset):
    def __init__(self, dt):
        assert dt in ('train', 'val')
        german_file = f"../00_assets/data/de_en/{dt}.de"
        english_file = f"../00_assets/data/de_en/{dt}.en"
        with open(german_file, 'r', encoding='utf-8') as f_de, \
                open(english_file, 'r', encoding='utf-8') as f_en:
            self.src_texts = f_de.readlines()
            self.tgt_texts = f_en.readlines()

        assert len(self.src_texts) == len(
            self.tgt_texts), "German and English files should have the same number of lines."

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx].strip()
        tgt_text = self.tgt_texts[idx].strip()
        return src_text, tgt_text
