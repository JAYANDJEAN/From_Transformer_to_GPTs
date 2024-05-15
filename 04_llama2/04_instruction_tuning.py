import pandas as pd
from torch.utils.data import Dataset
import json
import numpy as np
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from timeit import default_timer as timer
from utils import init_model
import torch
import os
import yaml


# Instruction Tuning


class InstructionDataset(Dataset):
    def __init__(self, dp, tokenizer
                 , max_length=256
                 , prompt_max_len=128
                 , answer_max_len=128):
        super().__init__()
        # download data: https://huggingface.co/datasets/shibing624/alpaca-zh
        with open(dp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        self.df = df.loc[(df['instruction'].str.len() >= 10) & (df['instruction'].str.len() <= 256) &
                         (df['output'].str.len() >= 5) & (df['output'].str.len() <= 256), ['instruction', 'output']]
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.special_tokens['<bos>']  # 1
        self.eos = self.tokenizer.special_tokens['<eos>']  # 2
        self.pad = self.tokenizer.special_tokens['<pad>']  # 0

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        prompt = self.tokenizer.encode(sample['instruction'], add_special_tokens=False)
        answer = self.tokenizer.encode(sample['output'], add_special_tokens=False)
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len - 2]
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len - 2]

        input_id = prompt + [self.bos] + answer + [self.eos]
        context_length = input_id.index(self.bos)
        mask_position = context_length - 1
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [self.pad] * pad_len
        if pad_len == 0:
            loss_mask = [0] * context_length + [1] * (len(input_id[mask_position + 1:])) + [0] * pad_len
        else:
            loss_mask = [0] * context_length + [1] * (len(input_id[mask_position + 1:-pad_len])) + [0] * pad_len
        input_id = np.array(input_id)
        src = np.array(input_id[:-1]).astype(np.int64)
        tgt = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[:-1])

        return torch.from_numpy(src), torch.from_numpy(tgt), torch.from_numpy(loss_mask)


def train_epoch(model, dataloader):
    model.train()
    losses = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 betas=(config['beta1'], config['beta2']))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    for i, (src, tgt, loss_mask) in enumerate(dataloader):
        src = src.to(config['device'])
        tgt = tgt.to(config['device'])
        loss_mask = loss_mask.to(config['device'])
        tgt_predict = model(src, 0)

        optimizer.zero_grad()
        loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt.reshape(-1))
        loss_mask = loss_mask.view(-1)
        loss = torch.sum(loss * loss_mask) / loss_mask.sum()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        pbar.update(1)

    return losses / len(list(dataloader))


if __name__ == "__main__":
    with open('../00_assets/yml/tiny_chinese_llama.yml', 'r') as file:
        config = yaml.safe_load(file)
    config['init_from'] = 'resume'
    with open('../00_assets/yml/local_settings.yml', 'r') as file:
        setting = yaml.safe_load(file)
    datapath = setting['model_path'] + 'alpaca_gpt4_data_zh.json'

    tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    train_ds = InstructionDataset(datapath, tokenizer, max_length=256)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    llama_model = init_model(config)

    print("Total trainable parameters:", sum(p.numel() for p in llama_model.parameters() if p.requires_grad))
    min_train_loss = float('inf')
    save_dir = os.path.join(config['out_dir'], 'pretrain')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, config['num_epochs'] + 1):
        start_time = timer()
        with tqdm(total=len(list(train_loader)), desc=f'Epoch {epoch}', unit='batch') as pbar:
            train_loss = train_epoch(llama_model, train_loader)
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save(llama_model.state_dict(), '{}/best.pth'.format(save_dir))
            end_time = timer()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
                   f"Epoch time = {(end_time - start_time):.3f}s"))
