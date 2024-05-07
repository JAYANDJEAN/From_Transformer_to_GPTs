from llama2_scratch import LlamaForCompletion
import torch
import json

with open('../00_assets/prompts.json', 'r') as file:
    data = json.load(file)

prompts = data['prompts']
torch.manual_seed(0)

model_path = '/Users/fengyuan/Documents/models/'
# 目前m3 max上的cpu可跑，还未测试cuda，mps报错，未细看
device = "cpu"

model = LlamaForCompletion.build(
    checkpoints_dir=model_path + 'Llama-2-7b/',
    tokenizer_path=model_path + 'Llama-2-7b/tokenizer.model',
    max_batch_size=len(prompts),
    max_seq_len=1024,
    device=device
)

out_tokens, out_texts = (model.completion(prompts, max_gen_len=64))
assert len(out_texts) == len(prompts)
for i in range(len(out_texts)):
    print(f'{out_texts[i]}')
    print('-' * 50)
