from llama2_scratch import LlamaForCausal
import torch
import json

with open('prompts.json', 'r') as file:
    data = json.load(file)

prompts = data['prompts']
torch.manual_seed(0)
device = "cpu"
model_path = '/Users/fengyuan/Documents/models/'

model = LlamaForCausal.build(
    checkpoints_dir=model_path + 'Llama-2-7b/',
    tokenizer_path=model_path + 'Llama-2-7b/tokenizer.model',
    max_batch_size=len(prompts),
    max_seq_len=1024,
    device=device
)

out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
assert len(out_texts) == len(prompts)
for i in range(len(out_texts)):
    print(f'{out_texts[i]}')
    print('-' * 50)
