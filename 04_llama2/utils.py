import time
import json
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sentencepiece import SentencePieceProcessor
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from llama import ModelArgs, LlamaModel, sample_top_p


def process_wiki_clean(path_in, path_out, tokenizer):
    # download data: https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered
    with open(path_in, 'r', encoding='utf-8') as f:
        data = json.load(f)
    doc_ids = []
    for line in tqdm(data):
        text = line['completion']
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id) > 5:
            doc_ids += text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    with open(path_out, 'wb') as f:
        f.write(arr.tobytes())


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


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256, prompt_max_len=128, answer_max_len=128):
        super().__init__()
        # download data: https://huggingface.co/datasets/shibing624/alpaca-zh
        with open(data_path, 'r', encoding='utf-8') as f:
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


def init_model(config):
    # 模型定义
    model_args: ModelArgs = ModelArgs(
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        use_cache=False,
        device=config['device']
    )

    if config['init_from'] == "scratch":
        print("Initializing a new model from scratch")
        model = LlamaModel(model_args)
    elif config['init_from'] == "resume":
        print(f"Resuming training from {config['save_dir']}")
        # resume training from a checkpoint.
        checkpoints = sorted(Path(config['save_dir']).glob("*.pth"))
        checkpoint = torch.load(checkpoints[0], map_location="cpu")
        model = LlamaModel(model_args)
        model.load_state_dict(checkpoint, strict=True)
    else:
        model = None
    model = model.to(config['device'])
    return model


class LlamaForCausal:
    def __init__(self, checkpoints_dir: str, tokenizer_path: str, tokenizer_tp: str, args: ModelArgs):
        checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
        print(f'Loading checkpoint "{checkpoints[0]}"')
        prev_time = time.time()
        checkpoint = torch.load(checkpoints[0], map_location="cpu")
        print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")

        self.args = args

        # 兼容一下不同的tokenizer
        assert tokenizer_tp in ("GLM", "SPP")
        if tokenizer_tp == 'SPP':
            self.tokenizer = SentencePieceProcessor()
            self.tokenizer.load(tokenizer_path)
            self.args.vocab_size = self.tokenizer.vocab_size()
            del checkpoint['rope.freqs']
            self.pad_id = self.tokenizer.pad_id()
            self.eos_id = self.tokenizer.eos_id()
        elif tokenizer_tp == 'GLM':
            self.tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
            # 64793
            self.args.vocab_size = self.tokenizer.vocab_size - 1
            self.pad_id = self.tokenizer.pad_token_id
            self.eos_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer = None

        self.model = LlamaModel(self.args).to(self.args.device)
        prev_time = time.time()
        self.model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

    def generate(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9,
                 max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
                         for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, \
            f"batch size must be less than or equal to {self.args.max_batch_size}"
        min_prompt_len = min(len(prompt) for prompt in prompt_tokens)
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, \
            f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        tokens = torch.full((batch_size, total_len), self.pad_id, dtype=torch.long, device=self.args.device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = tokens != self.pad_id
        cur_iterator = tqdm(range(min_prompt_len, total_len), desc="Generating tokens")
        prev_pos = 0
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens, out_text = [], []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text


class LlamaForSequenceClassification:
    pass
