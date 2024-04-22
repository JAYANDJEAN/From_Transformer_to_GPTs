from transformers import AutoTokenizer

# model_id = 'openai-community/gpt2'
# model_id= 'google-bert/bert-base-cased'
# 'facebook/mbart-large-cc25'
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token='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')

# The first sentences from the abstract of "<Attention Is All You Need>"
corpus = [
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.",
    "主导的序列转换模型基于复杂的循环或卷积神经网络，包括一个编码器和一个解码器。"]

result = tokenizer(corpus)
print(result['input_ids'][1])
for i in result['input_ids'][1]:
    print(tokenizer.decode(i).encode('utf-8'))

print(tokenizer.all_special_tokens)
print(tokenizer.__len__())
