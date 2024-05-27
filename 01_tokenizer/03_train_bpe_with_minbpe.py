"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe.basic import BasicTokenizer
from minbpe.regex import RegexTokenizer

# open some text and train a vocab of 512 tokens
text = open("../00_assets/de_en/train.en", "r", encoding="utf-8").read()
text += open("../00_assets/de_en/val.en", "r", encoding="utf-8").read()
# create a directory for models, so we don't pollute the current directory
save_dir = "../02_transformer/bpe_tokenizer"
os.makedirs(save_dir, exist_ok=True)

tokenizer = BasicTokenizer()
tokenizer.train(text, 1000, verbose=True)
prefix = os.path.join(save_dir, 'minbpe-en')
tokenizer.save(prefix)
