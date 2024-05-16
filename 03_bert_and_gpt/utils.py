from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

eli5 = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)
eli5 = eli5.train_test_split(test_size=0.2)
eli5 = eli5.flatten()


def generate_mask(sz: int):
    # 生成一个下三角矩阵
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def prepare_dataset(block_size: int, tokenizer):
    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()  # 不同！！！！！！！gpt
        return result

    tokenized_eli5 = eli5.map(preprocess_function, batched=True, num_proc=4, remove_columns=eli5["train"].column_names)
    dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
    return dataset


def prepare_loader_from_set(batch_size: int, tokenizer):
    # 对于已有数据集，把处理逻辑放在collate_fn
    def collate_fn(batch):
        res = []
        for sample in batch:
            encoded_input = tokenizer(sample['title'], return_tensors='pt')
            res.append(encoded_input['input_ids'].squeeze())
        res_batch = pad_sequence(res, padding_value=tokenizer.pad_token_id, batch_first=True)
        mask = (res_batch != tokenizer.pad_token_id).int()
        return res_batch, mask

    train_dataloader = DataLoader(eli5['train'], batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(eli5['test'], batch_size=batch_size, collate_fn=collate_fn)
    return train_dataloader, test_dataloader


def prepare_loader_from_file(batch_size, src_tokenizer, tgt_vocabs, csv_file, columns):
    # 对于大型文件，使用Dataset，避免一次性加载到内存，处理逻辑依然放在collate_fn，因为需要拿到batch内所有数据
    class CustomDataset(Dataset):
        def __init__(self, file_path, tp, columns_use):
            self.data = pd.read_csv(file_path)
            assert tp in ('train', 'test', 'val')
            assert len(columns_use) == 2
            self.data = self.data[self.data['tp'] == tp]
            self.data = self.data[columns_use]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            addr = self.data.iloc[idx, 0]
            geo_code = self.data.iloc[idx, 1]
            return addr, geo_code

    def collate_fn(batch):
        addr_batch, geo_batch = [], []
        for addr_sample, geo_sample in batch:
            encoded_input = src_tokenizer(addr_sample, return_tensors='pt')
            addr_batch.append(encoded_input['input_ids'].squeeze())
            results = ([src_tokenizer.bos_token_id] +
                       [tgt_vocabs[char] for char in geo_sample] +
                       [src_tokenizer.eos_token_id])
            geo_batch.append(torch.as_tensor(results))
        src_batch = pad_sequence(addr_batch, padding_value=src_tokenizer.pad_token_id, batch_first=True)
        src_mask = (src_batch != src_tokenizer.pad_token_id).int()
        tgt_batch = pad_sequence(geo_batch, padding_value=src_tokenizer.pad_token_id, batch_first=True)
        return src_batch, src_mask, tgt_batch

    train_ = DataLoader(CustomDataset(csv_file, 'train', columns),
                        batch_size=batch_size,
                        collate_fn=collate_fn,
                        shuffle=True)
    test_ = DataLoader(CustomDataset(csv_file, 'test', columns),
                       batch_size=batch_size,
                       collate_fn=collate_fn,
                       shuffle=False)
    val_ = DataLoader(CustomDataset(csv_file, 'val', columns),
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      shuffle=False)
    return train_, val_, test_


def generate(model, tokenizer, dataloader, tgt_vocab, config):
    def int_to_string(int_array):
        return ''.join([tgt_vocab.get(val, '-') for val in int_array])

    model.eval()
    res_pred = []
    res_true = []
    device = config['device']
    for src, src_mask, tgt in dataloader:
        btz, _ = src.shape
        src = src.to(device)
        src_mask = src_mask.to(device)
        memory = model.encode(src, src_mask).to(device)
        ys = torch.ones(btz, 1).fill_(tokenizer.bos_token_id).type(torch.long).to(device)
        max_len = 13 if config['geo_code_type'] == 'geohash' else 17
        for i in range(max_len):
            tgt_mask = (generate_mask(ys.size(1))).to(device)
            out = model.decode(ys, memory, tgt_mask)
            prob = model.generator(out[:, -1, :])
            next_word = torch.argmax(prob, dim=1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)

        predict = ys[:, 1:-1].cpu().numpy()
        tgt = tgt[:, 1:-1].cpu().numpy()
        res_pred.append(np.apply_along_axis(int_to_string, axis=1, arr=predict))
        res_true.append(np.apply_along_axis(int_to_string, axis=1, arr=tgt))

    concat_pred = np.concatenate(res_pred)
    concat_true = np.concatenate(res_true)
    df = pd.DataFrame({'geo_true': concat_true, 'geo_pred': concat_pred})
    df.to_csv(f"result_{config['model_version']}.csv", index=False)
