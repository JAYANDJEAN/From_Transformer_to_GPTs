from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer()
special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
tokenizer.add_special_tokens(special_tokens)

tokenizer.train(files=["../00_data/train.de", "../00_data/train.en"], trainer=trainer)
corpus = [
    "Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.",
    "A man in green holds a guitar while the other man observes his shirt."
]

print(tokenizer.get_vocab_size())

for sentence in corpus:
    print(tokenizer.encode(sentence).tokens)
