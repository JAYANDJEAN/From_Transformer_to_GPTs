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

# create a directory for models, so we don't pollute the current directory
save_dir = "../02_transformer/bpe_tokenizer"
os.makedirs(save_dir, exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join(save_dir, name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")