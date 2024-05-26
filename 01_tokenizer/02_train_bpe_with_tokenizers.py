from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

data_types = ['en', 'de']
for dt in data_types:
    vocab_size = 10000 if dt == 'en' else 19000
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
    tokenizer.pre_tokenizer = Whitespace()
    files = [f"../00_assets/de_en/{split}.{dt}" for split in ["train", "val"]]
    tokenizer.train(files, trainer)
    tokenizer.save(f"../02_transformer/bpe_tokenizer/token-{dt}.json")

