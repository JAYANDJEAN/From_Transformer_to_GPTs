from utils import LlamaForCausal
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import json
import yaml

with open('../00_assets/json/prompts.json', 'r') as file:
    data = json.load(file)
prompts = data['prompts']
torch.manual_seed(0)
with open('../00_assets/yml/local_settings.yml', 'r') as file:
    config = yaml.safe_load(file)
device = "cpu"  # 目前m3 max上的cpu可跑，还未测试cuda，mps报错，未细看
model_type = '7b'  # '7b', 'hf'

# 两种生成出来的还是有些不一样的，我也没查具体原因。
if model_type == 'hf':
    model = LlamaForCausalLM.from_pretrained(config['model_path'] + "Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'] + "Llama-2-7b-hf")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        generate_ids = model.generate(inputs.input_ids, max_length=64)
        print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        print('-' * 40)
elif model_type == '7b':
    model = LlamaForCausal(
        checkpoints_dir=config['model_path'] + 'Llama-2-7b/',
        tokenizer_path=config['model_path'] + 'Llama-2-7b/tokenizer.model',
        tokenizer_tp='SPP',
        max_batch_size=len(prompts),
        max_seq_len=1024,
        device=device
    )
    out_tokens, out_texts = (model.generate(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(out_texts[i])
        print('-' * 40)
elif model_type == 'tiny':
    model = LlamaForCausal(
        checkpoints_dir='../00_assets/pretrain/',
        tokenizer_path='./chatglm_tokenizer/tokenizer.model',
        tokenizer_tp='GLM',
        max_batch_size=3,
        max_seq_len=256,
        device=device
    )
    sentence = ["今天天气不错啊！"]
    # ，门牌编号为香港黄大仙区竹园大成街8号，为一个公园中的公园，入口设于东头村道及大成街。
    out_tokens, out_texts = (model.generate(sentence, max_gen_len=64))
    assert len(out_texts) == len(sentence)
    for i in range(len(out_texts)):
        print(out_texts[i])
        print('-' * 40)
else:
    pass
