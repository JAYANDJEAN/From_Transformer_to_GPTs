import os
import json

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer


# Instruction Tuning
# https://github.com/DLLXW/baby-llama2-chinese/tree/main

# https://huggingface.co/datasets/shibing624/alpaca-zh

def sft_process():
    with open('./sft_data/alpaca_gpt4_data_zh.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    #
    q_lst = []
    a_lst = []
    for per in data:
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = q + i
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q) > 256 or len(a) > 256:
            continue
        q_lst.append(q)
        a_lst.append(a)

    f = open('./sft_data/Belle_open_source_1M.json', 'r', encoding='utf-8')

    # s
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = q + i
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q) > 256 or len(a) > 256:
            continue
        q_lst.append(q)
        a_lst.append(a)
    df = pd.DataFrame(columns=['prompt', 'answer'])
    df['prompt'] = q_lst
    df['answer'] = a_lst
    df.to_csv('sft_data/sft_data.csv', index=False)
    print(df)


save_dir = './sft_data'
if not os.path.exists(save_dir): os.makedirs(save_dir)
sft_process()


class SFTDataset(Dataset):
    def __init__(self, df, tokenizer
                 , max_length=256
                 , prompt_max_len=128
                 , answer_max_len=128):
        super().__init__()
        self.df = df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        #
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.special_tokens['<bos>']
        self.eos = self.tokenizer.special_tokens['<eos>']
        self.pad = 0  # self.tokenizer.special_tokens['<pad>']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        prompt = self.tokenizer.encode(sample['prompt'], add_special_tokens=False)
        answer = self.tokenizer.encode(sample['answer'], add_special_tokens=False)
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len - 2]
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len - 2]
        #
        input_id = prompt + [self.bos] + answer + [self.eos]
        context_length = input_id.index(self.bos)
        mask_position = context_length - 1
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [self.pad] * pad_len
        if pad_len == 0:
            loss_mask = [0] * context_length + [1] * (len(input_id[mask_position + 1:])) + [0] * pad_len
        else:
            loss_mask = [0] * context_length + [1] * (len(input_id[mask_position + 1:-pad_len])) + [0] * pad_len
        #
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[:-1])
        #
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


#
if __name__ == "__main__":
    df = pd.read_csv('./data/sft_data.csv')
    tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    train_ds = SFTDataset(df, tokenizer, max_length=256)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    for i, (X, Y, loss_mask) in enumerate(train_loader):
        print(X.shape, Y.shape)
        print(X[0])
        print(Y[0])
        print(loss_mask[0])
        break
