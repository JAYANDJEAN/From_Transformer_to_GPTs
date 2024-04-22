from transformers import AutoTokenizer

# model_id = 'openai-community/gpt2'
# model_id= 'google-bert/bert-base-cased'
# 'facebook/mbart-large-cc25'

model_path = '/Users/yuan.feng/PycharmProjects/From_Transformer_to_GPTs/01_Tokenizer/llama-2'
tokenizer = AutoTokenizer.from_pretrained(model_path)

corpus = [
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.",
    "主导的序列转换模型基于复杂的循环或卷积神经网络，包括一个编码器和一个解码器。"]

result = tokenizer(corpus)
print(result['input_ids'][0])
for i in result['input_ids'][0]:
    print(tokenizer.decode(i))

print(tokenizer.all_special_tokens)
print(tokenizer.__len__())
