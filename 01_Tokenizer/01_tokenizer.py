from transformers import AutoTokenizer, MBart50TokenizerFast
from torchtext.data.utils import get_tokenizer

model_list = ['openai-community/gpt2', 'google-bert/bert-base-uncased', 'facebook/mbart-large-cc25']

corpus = [
    "Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.",
    "A man in green holds a guitar while the other man observes his shirt."
]
tokenizer0 = get_tokenizer('spacy', language='de_core_news_sm')
tokenizer1 = get_tokenizer('spacy', language='en_core_web_sm')

print(tokenizer0(corpus[0]))
print(tokenizer1(corpus[1]))
for model_id in model_list:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(model_id, len(tokenizer))

    for sentence in corpus:
        print(tokenizer(sentence).tokens())
