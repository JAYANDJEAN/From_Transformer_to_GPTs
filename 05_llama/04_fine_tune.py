from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from timeit import default_timer as timer
from models import LlamaModel, ModelArgs
import torch
import os
import yaml

# https://github.com/DLLXW/baby-llama2-chinese/tree/main

sentnece = "黄大仙文化公园（Wong Tai Sin Culture Park）是香港一个公园，位于九龙黄大仙摩士公园，门牌编号为香港黄大仙区竹园大成街8号，为一个公园中的公园，入口设于东头村道及大成街。"

with open('../00_assets/tiny_chinese_llama.yml', 'r') as file:
    config = yaml.safe_load(file)

model_args: ModelArgs = ModelArgs(
    dim=config['dim'],
    n_layers=config['n_layers'],
    n_heads=config['n_heads'],
    n_kv_heads=config['n_heads'],
    vocab_size=config['vocab_size'],
    multiple_of=config['multiple_of'],
    max_seq_len=config['max_seq_len'],
    kv_cache=False,
    device=config['device']
)
model = LlamaModel(model_args)
checkpoint = torch.load(os.path.join(config['out_dir'], "pretrain/best.pth"), map_location=config['device'])
model.load_state_dict(checkpoint)

tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
print(tokenizer.vocab_size)
print(tokenizer.pad_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.eos_token)