import sentencepiece as spm
import os


data_types = ['en', 'de']
save_dir = "../00_assets/tokenizers/en_de_sentencepiece/"
os.makedirs(save_dir, exist_ok=True)

for dt in data_types:
    files = [f"../00_assets/data/de_en/{split}.{dt}" for split in ["train", "val"]]
    vocab_size = 10000 if dt == 'en' else 19000
    special_tokens = ["<pad>", "<bos>", "<eos>"]  # <unk> 已添加
    spm.SentencePieceTrainer.train(
        input=files,
        model_prefix=f"{save_dir}token-{dt}",
        vocab_size=vocab_size,
        user_defined_symbols=special_tokens,
        model_type='bpe'
    )
