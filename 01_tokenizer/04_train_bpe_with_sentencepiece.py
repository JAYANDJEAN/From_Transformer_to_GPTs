import sentencepiece as spm

data_types = ['en', 'de']
for dt in data_types:
    files = [f"../00_assets/de_en/{split}.{dt}" for split in ["train", "val"]]
    model_prefix = f"../02_transformer/bpe_tokenizer/sp-{dt}"
    vocab_size = 10000 if dt == 'en' else 19000
    special_tokens = ["<pad>", "<bos>", "<eos>"]  # <unk> 已添加
    spm.SentencePieceTrainer.train(
        input=files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        user_defined_symbols=special_tokens,
        model_type='bpe'
    )
