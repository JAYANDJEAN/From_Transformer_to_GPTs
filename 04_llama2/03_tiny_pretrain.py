import json
import numpy as np
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from timeit import default_timer as timer
from utils import init_model

from torch.utils.data import Dataset
import torch
import os
import yaml
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

'''
https://github.com/DLLXW/baby-llama2-chinese
'''


class PretrainDataset(Dataset):
    def __init__(self, data_path_lst, max_length=256):
        super().__init__()
        data_lst = []
        for data_path in data_path_lst:
            with open(data_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.uint16)
                data_lst.append(data)
        data = np.concatenate(data_lst)
        data = data[:max_length * int(len(data) / max_length)]
        self.data = data.reshape(-1, max_length)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        sample = self.data[index].astype(np.int64)
        return torch.from_numpy(sample)


def process_wiki_clean(path_in, path_out):
    # download data: https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered
    token = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    with open(path_in, 'r', encoding='utf-8') as f:
        data = json.load(f)
    doc_ids = []
    for line in tqdm(data):
        text = line['completion']
        text_id = token.encode(text, add_special_tokens=False)
        text_id.append(token.special_tokens['<eos>'])
        if len(text_id) > 5:
            doc_ids += text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    with open(path_out, 'wb') as f:
        f.write(arr.tobytes())


if __name__ == "__main__":
    # 模型训练不能用kv_cache，因为...
    with open('../00_assets/yml/tiny_chinese_llama.yml', 'r') as file:
        config = yaml.safe_load(file)
    with open('../00_assets/yml/local_settings.yml', 'r') as file:
        setting = yaml.safe_load(file)

    data_input = setting['model_path'] + 'wikipedia-cn-20230720-filtered.json'
    data_output = setting['model_path'] + 'wiki.bin'
    save_dir = os.path.join(setting['model_path'], 'tiny_llama')
    config['save_dir'] = save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(data_output):
        print('-------data process--------')
        process_wiki_clean(data_input, data_output)

    train_loader = torch.utils.data.DataLoader(PretrainDataset([data_output], max_length=config['max_seq_len']),
                                               batch_size=config['batch_size'])
    llama_model = init_model(config)

    # 设置优化器和学习率调度器的参数
    learning_rate = 2e-5
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    eps = 1e-5
    warmup_steps = 200
    gradient_clipping_value = 1.0
    total_training_steps = len(train_loader) * config['num_epochs']
    optimizer = AdamW(llama_model.parameters(),
                      lr=learning_rate, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_training_steps)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    print("Total trainable parameters:", sum(p.numel() for p in llama_model.parameters() if p.requires_grad))

    min_train_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        start_time = timer()
        with tqdm(total=len(list(train_loader)), desc=f'Epoch {epoch}', unit='batch') as pbar:
            llama_model.train()
            losses = 0
            for i, data in enumerate(train_loader):
                src = data[:, :-1].to(config['device'])
                tgt = data[:, 1:].to(config['device'])
                tgt_predict = llama_model(src, 0).to(config['device'])

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
                torch.save(llama_model.state_dict(), '{}/best.pth'.format(save_dir))
            end_time = timer()
            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
