import json
import numpy as np
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from timeit import default_timer as timer
from models import LlamaModel, ModelArgs

from torch.utils.data import Dataset
import torch
import os
import yaml
from modelsummary import summary

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


def train_epoch(model, dataloader, tp):
    if tp == 'train':
        model.train()
    elif tp == 'eval':
        model.eval()

    losses = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 betas=(config['beta1'], config['beta2']))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    for i, d in enumerate(dataloader):
        src = d[:, :-1].to(config['device'])
        tgt = d[:, 1:].to(config['device'])
        tgt_predict = model(src, 0)

        if tp == 'train':
            optimizer.zero_grad()
            loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            losses += loss.item()
        elif tp == 'eval':
            loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt.reshape(-1))
            losses += loss.item()
        pbar.update(1)

    return losses / len(list(dataloader))


if __name__ == "__main__":
    # 模型训练不能用kv_cache，因为...
    with open('../00_assets/tiny_chinese_llama.yml', 'r') as file:
        config = yaml.safe_load(file)
    dp = '/Users/yuan.feng/Downloads/wikipedia-cn-20230720-filtered.json'

    data_file = config['out_dir'] + 'wiki.bin'
    if not os.path.exists(data_file):
        print('-------data process--------')
        process_wiki_clean(dp, data_file)

    train_ds = PretrainDataset([data_file], max_length=config['max_seq_len'])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'])

    # 模型定义
    if config['init_from'] == "scratch":
        print("Initializing a new model from scratch")
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
    elif config['init_from'] == "resume":
        print(f"Resuming training from {config['out_dir']}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(config['out_dir'], "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config['device'])
        checkpoint_model_args = checkpoint["model_args"]
        model_args: ModelArgs = ModelArgs(
            dim=checkpoint_model_args['dim'],
            n_layers=checkpoint_model_args['n_layers'],
            n_heads=checkpoint_model_args['n_heads'],
            n_kv_heads=checkpoint_model_args['n_heads'],
            vocab_size=checkpoint_model_args['vocab_size'],
            multiple_of=checkpoint_model_args['multiple_of'],
            max_seq_len=checkpoint_model_args['max_seq_len'],
            kv_cache=False,
            device=config['device']
        )
        model = LlamaModel(model_args)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        # iter_num = checkpoint["iter_num"]
        # best_val_loss = checkpoint["best_val_loss"]
    else:
        model = None
    model = model.to(config['device'])

    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    min_train_loss = float('inf')
    save_dir = os.path.join(config['out_dir'], 'pretrain')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, config['num_epochs'] + 1):
        start_time = timer()
        with tqdm(total=len(list(train_loader)), desc=f'Epoch {epoch}', unit='batch') as pbar:
            train_loss = train_epoch(model, train_loader, 'train')
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save(model.state_dict(), '{}/best.pth'.format(save_dir))
            end_time = timer()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
                   f"Epoch time = {(end_time - start_time):.3f}s"))
