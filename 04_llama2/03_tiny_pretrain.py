import os
from timeit import default_timer as timer

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from utils import init_model, PretrainDataset, ChatGLMTokenizer

'''
https://github.com/DLLXW/baby-llama2-chinese
'''


def train_and_translate(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = ChatGLMTokenizer(vocab_file='../00_assets/tokenizers/chatglm_sentencepiece/tokenizer.model')
    train_loader = torch.utils.data.DataLoader(
        PretrainDataset(["../00_assets/data/wiki.bin"],
                        max_length=config['max_seq_len']),
        batch_size=config['batch_size'])
    llama_model = init_model(config)
    gradient_clipping_value = 1.0
    optimizer = AdamW(llama_model.parameters(),
                      lr=config['lr'], betas=(config['beta1'], config['beta2']),
                      eps=config['eps'], weight_decay=config['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=config['warmup_steps'],
                                                num_training_steps=len(train_loader) * config['num_epochs'])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    print("Total trainable parameters:", sum(p.numel() for p in llama_model.parameters() if p.requires_grad))

    min_train_loss = float('inf')
    for epoch in range(1, config['num_epochs'] + 1):
        start_time = timer()
        with tqdm(total=len(list(train_loader)), desc=f'Epoch {epoch}', unit='batch') as pbar:
            llama_model.train()
            losses = 0
            for i, train_data in enumerate(train_loader):
                src = train_data[:, :-1].to(device)
                tgt = train_data[:, 1:].to(device)
                tgt_predict = llama_model(src, 0).to(device)
                optimizer.zero_grad()
                loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(llama_model.parameters(), gradient_clipping_value)
                optimizer.step()
                scheduler.step()
                losses += loss.item()
                pbar.update(1)
            train_loss = losses / len(list(train_loader))

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save(llama_model.state_dict(), f"{config['save_dir']}/tiny_chinese_llama.pth")
            end_time = timer()
            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")


if __name__ == "__main__":
    # 模型训练不能用kv_cache，因为...
    train_config = {
        'init_from': 'scratch',
        'save_dir': '../00_assets/models',
        # 'dtype': float16,
        'batch_size': 32,
        'num_epochs': 10,
        'dim': 256,
        'n_layers': 2,
        'n_heads': 4,
        'n_kv_heads': 4,
        'vocab_size': 64793,
        'max_seq_len': 256,
        'hidden_dim': 512,
        'dropout': 0.0,
        'lr': 2e-5,
        'beta1': 0.9,
        'beta2': 0.98,
        'eps': 1e-5,
        'weight_decay': 0.1,
        'warmup_steps': 200
    }

    os.makedirs(train_config['save_dir'], exist_ok=True)
