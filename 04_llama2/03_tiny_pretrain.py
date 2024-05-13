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

'''
https://github.com/DLLXW/baby-llama2-chinese
# 未完成！
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


def train_epoch(model, dataloader):
    model.train()
    losses = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 betas=(config['beta1'], config['beta2']))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    for i, d in enumerate(dataloader):
        src = d[:, :-1].to(config['device'])
        tgt = d[:, 1:].to(config['device'])
        tgt_predict = model(src, 0)

        optimizer.zero_grad()
        loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
        pbar.update(1)

    return losses / len(list(dataloader))


if __name__ == "__main__":
    # 模型训练不能用kv_cache，因为...
    with open('../00_assets/yml/tiny_chinese_llama.yml', 'r') as file:
        config = yaml.safe_load(file)
    with open('../00_assets/yml/local_settings.yml', 'r') as file:
        setting = yaml.safe_load(file)
    dp = setting['model_path'] + 'wikipedia-cn-20230720-filtered.json'

    data_file = config['out_dir'] + 'wiki.bin'
    if not os.path.exists(data_file):
        print('-------data process--------')
        process_wiki_clean(dp, data_file)

    train_ds = PretrainDataset([data_file], max_length=config['max_seq_len'])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'])

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
