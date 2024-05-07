from models import LlamaForCompletion
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import json

with open('../00_assets/prompts.json', 'r') as file:
    data = json.load(file)
prompts = data['prompts']
torch.manual_seed(0)
model_path = '/Users/fengyuan/Documents/models/'
device = "cpu"  # 目前m3 max上的cpu可跑，还未测试cuda，mps报错，未细看
use_transformers = False

# 两种生成出来的还是有些不一样的，我也没查具体原因。
if use_transformers:
    model = LlamaForCausalLM.from_pretrained(model_path + "Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_path + "Llama-2-7b-hf")
    print('transformers:')
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        generate_ids = model.generate(inputs.input_ids, max_length=64)
        print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        print('-' * 40)
else:
    print('LlamaForCompletion:')
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
        print(out_texts[i])
        print('-' * 40)


