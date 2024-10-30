import os
import time
from minbpe.basic import BasicTokenizer
from minbpe.regex import RegexTokenizer

data_types = ['en', 'de']
save_dir = "../00_assets/tokenizers/en_de_minbpe/"
os.makedirs(save_dir, exist_ok=True)

for dt in data_types:
    # open some text and train a vocab of 512 tokens
    text = open(f"../00_assets/data/de_en/train.{dt}", "r", encoding="utf-8").read()
    text += open(f"../00_assets/data/de_en/val.{dt}", "r", encoding="utf-8").read()
    # create a directory for models, so we don't pollute the current directory
    tokenizer = BasicTokenizer()
    tokenizer.train(text, 1000, verbose=True)
    tokenizer.save(f"{save_dir}token-{dt}")
