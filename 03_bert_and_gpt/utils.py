from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch


def generate_mask(sz: int):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def prepare_dataset_books(tokenizer):
    def pre_process(examples):
        prefix = "translate German to English: "
        src_lang = "de"
        tgt_lang = "en"
        inputs = [prefix + example[src_lang] for example in examples["translation"]]
        targets = [example[tgt_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    books = load_dataset("opus_books", "de-en")
    books = books["train"].train_test_split(test_size=0.1)
    dataset = books.map(pre_process, batched=True)
    return dataset


def prepare_dataset_eli5(tokenizer):
    block_size = 128
    eli5 = load_dataset("eli5_category", split="train[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2)
    eli5 = eli5.flatten()

    tokenized_eli5 = eli5.map(
        lambda x: tokenizer([" ".join(x) for x in x["answers.text"]]),
        batched=True,
        num_proc=4,
        remove_columns=eli5["train"].column_names,
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

    return lm_dataset


def prepare_dataset_geo(tokenizer):
    dataset = load_dataset("csv", data_files="../00_assets/data/addr_to_geo.csv")
    train_dataset = dataset['train'].filter(lambda example: example['dt'] == 'train')
    val_dataset = dataset['train'].filter(lambda example: example['dt'] == 'val')
    test_dataset = dataset['train'].filter(lambda example: example['dt'] == 'test')

    def preprocess_function(examples):
        inputs = [example for example in examples["address"]]
        targets = [''.join([f"<%{i}>" for i in example]) for example in examples["geohash"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    return tokenized_train, tokenized_val, tokenized_test


def prepare_torch_dataset(batch_size, src_tokenizer, tgt_vocabs, csv_file, columns):
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
