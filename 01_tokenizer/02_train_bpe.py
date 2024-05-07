from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# 应该有bug，但还没检查出来
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["../00_assets/train.de", "../00_assets/train.en"], trainer=trainer)

corpus = [
    "Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.",
    "A man in green holds a guitar 中 😁 while the other man observes his shirt."
]

print(tokenizer.get_vocab_size())
print(tokenizer.id_to_token(0))
print(tokenizer.id_to_token(1))
print(tokenizer.id_to_token(2))
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)

for sentence in corpus:
    res = tokenizer.encode(sentence)
    print(res.ids)
    print(res.tokens)
