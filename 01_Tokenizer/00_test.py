from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torchtext.data.utils import get_tokenizer

sentence = "A man in 中 a blue windbreaker is playing with his yellow remote controlled airplane."
s1 = "主导的序列转换模型基于复杂的循环或卷积神经网络，包括一个编码器和一个解码器。"

# tokenizer = Tokenizer(BPE())
# file1 = '/Users/yuan.feng/PycharmProjects/From_Transformer_to_GPTs/00_data/train.de'
# file2 = '/Users/yuan.feng/PycharmProjects/From_Transformer_to_GPTs/00_data/train.en'
# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# tokenizer.train(files=[file1, file2], trainer=trainer)
# output = tokenizer.encode(sentence)
# print('--')
# print(output.tokens)

tokenizer2 = get_tokenizer('spacy', language='zh_core_web_sm')
print(tokenizer2(s1))


