import time
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from llama import ModelArgs, LlamaModel, sample_top_p
import os
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding


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


class SPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def tokenize(self, s: str):
        return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens or index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        return self.sp_model.IdToPiece(index)


class ChatGLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, **kwargs):
        self.name = "GLMTokenizer"
        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }

        super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def unk_token(self) -> str:
        return "<unk>"

    @property
    def pad_token(self) -> str:
        return "<unk>"

    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    @property
    def eos_token(self) -> str:
        return "</s>"

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    def build_prompt(self, query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
                                         ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    def _pad(self,
             encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
             max_length: Optional[int] = None,
             padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
             pad_to_multiple_of: Optional[int] = None,
             return_attention_mask: Optional[bool] = None,
             ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs
