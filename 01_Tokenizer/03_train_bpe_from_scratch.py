from tokenizers import decoders, models, normalizers, \
    pre_tokenizers, processors, trainers, Tokenizer
from datasets import load_dataset

model = models.BPE()
tokenizer = Tokenizer(model)

################# GPT-2 Skip Normalization ##################

################# Step1: Pre-tokenization ###################
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

print(tokenizer.pre_tokenizer.pre_tokenize_str("This's me  ."))
# [('This', (0, 4)), ("'s", (4, 6)), ('Ġme', (6, 9)), ('Ġ', (9, 10)), ('Ġ.', (10, 12))]

################# Step2: Trainer ###################
special_tokens = ["<|endoftext|>"]  # end-of-text token
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=special_tokens)

################# Step3: dataset ###################


dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]  # batch size = 1000


################# Step4: train ####################
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
# tokenizer.train(["wikitext-2.txt"], trainer=trainer) # 也可以从文本文件来训练

## 测试训练好的 BPE
encoding = tokenizer.encode("This's me  .")
print(encoding)
# Encoding(num_tokens=8, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
print(encoding.ids)
# [52, 72, 215, 7, 83, 701, 159, 209]
print(encoding.type_ids)
# [0, 0, 0, 0, 0, 0, 0, 0]
print(encoding.tokens)
# ['T', 'h', 'is', "'", 's', 'Ġme', 'Ġ', 'Ġ.']
print(encoding.offsets)
# [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 9), (9, 10), (10, 12)]
print(encoding.attention_mask)
# [1, 1, 1, 1, 1, 1, 1, 1]
print(encoding.special_tokens_mask)
# [0, 0, 0, 0, 0, 0, 0, 0]
print(encoding.overflowing)
# []

################# Step5: Post-Processing ####################
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)  # 保留 ‘Ġ’ 代表的空格

## 测试训练好的 BPE (单个句子)
encoding = tokenizer.encode("This's me  .")
print(encoding)
# Encoding(num_tokens=8, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
print(encoding.ids)
# [52, 72, 215, 7, 83, 701, 159, 209]
print(encoding.type_ids)
# [0, 0, 0, 0, 0, 0, 0, 0]
print(encoding.tokens)
# ['T', 'h', 'is', "'", 's', 'Ġme', 'Ġ', 'Ġ.']
print(encoding.offsets)
# [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 9), (9, 10), (10, 12)]
print(encoding.attention_mask)
# [1, 1, 1, 1, 1, 1, 1, 1]
print(encoding.special_tokens_mask)
# [0, 0, 0, 0, 0, 0, 0, 0]
print(encoding.overflowing)
# []

## 测试训练好的 BPE (多个句子)
encoding = tokenizer.encode("This's me  .", "That's is fine-tuning.")
print(encoding)
# Encoding(num_tokens=19, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
print(encoding.ids)
# [52, 72, 215, 7, 83, 701, 159, 209, 52, 6312, 7, 83, 301, 7620, 13, 84, 302, 223, 14]
print(encoding.type_ids)
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(encoding.tokens)
# ['T', 'h', 'is', "'", 's', 'Ġme', 'Ġ', 'Ġ.', 'T', 'hat', "'", 's', 'Ġis', 'Ġfine', '-', 't', 'un', 'ing', '.']
print(encoding.offsets)
# [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 9), (9, 10), (10, 12), (0, 1), (1, 4), (4, 5), (5, 6), (6, 9), (9, 14), (14, 15), (15, 16), (16, 18), (18, 21), (21, 22)]
print(encoding.attention_mask)
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(encoding.special_tokens_mask)
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(encoding.overflowing)
# []

################# Step6: Decode ####################
tokenizer.decoder = decoders.ByteLevel()
tokenizer.decode(encoding.ids)  # 注意：空格能够被还原
# "This's me  .That's is fine-tuning."

################# Step7: Save ####################
tokenizer.save("tokenizer.json")
new_tokenizer = Tokenizer.from_file("tokenizer.json")
print(new_tokenizer.decode(encoding.ids))
# This's me  .That's is fine-tuning.
