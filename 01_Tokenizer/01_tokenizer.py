from transformers import AutoTokenizer
from torchtext.data.utils import get_tokenizer

model_list = ['openai-community/gpt2', 'google-bert/bert-base-uncased']
corpus = ["Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.",
          "A man in green holds a guitar while the other man observes his shirt."]
tokenizer0 = get_tokenizer('spacy', language='de_core_news_sm')
tokenizer1 = get_tokenizer('spacy', language='en_core_web_sm')

'''
spacy 基本是按空格分割
gpt 和 bert 都是按subword分。
'''
print(tokenizer0(corpus[0]))
# ['Ein', 'Mann', 'in', 'grün', 'hält', 'eine', 'Gitarre', ',', 'während', 'der',
# 'andere', 'Mann', 'sein', 'Hemd', 'ansieht', '.']
print(tokenizer1(corpus[1]))
# ['A', 'man', 'in', 'green', 'holds', 'a', 'guitar', 'while', 'the', 'other', 'man', 'observes', 'his', 'shirt', '.']

for model_id in model_list:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(model_id, len(tokenizer))
    for sentence in corpus:
        print(tokenizer(sentence).tokens())
