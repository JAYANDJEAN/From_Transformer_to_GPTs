import matplotlib.pyplot as plt
from llama2_scratch import *
from torch import Tensor
import torch
import json
from modelsummary import summary


def check_rope():
    class PositionalEncoding(nn.Module):
        def __init__(self, max_len: int, d_model: int, dropout: float, batch_first=False):
            super(PositionalEncoding, self).__init__()
            self.batch_first = batch_first

            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(- torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
            pos_embedding = torch.zeros(max_len, d_model)
            pos_embedding[:, 0::2] = torch.sin(position * div_term)
            pos_embedding[:, 1::2] = torch.cos(position * div_term)
            index = 0 if self.batch_first else 1
            pos_embedding = pos_embedding.unsqueeze(index)
            self.dropout = nn.Dropout(dropout)
            # pos_embedding被通过register_buffer方法注册为一个缓冲，而不是模型的可学习参数。
            self.register_buffer('pos_embedding', pos_embedding)

        def forward(self, x: Tensor):
            if self.batch_first:
                return self.dropout(x + self.pos_embedding[:, :x.size(1), :])
            else:
                return self.dropout(x + self.pos_embedding[:x.size(0), :, :])

    freqs_complex = precompute_theta_pos_frequencies(D_MODEL, MAX_LEN, 'cpu')
    x = torch.ones((1, MAX_LEN, 1, D_MODEL))

    pe = PositionalEncoding(MAX_LEN, D_MODEL, 0.1, batch_first=True)
    data_title = [(pe.pos_embedding.squeeze().numpy(), 'PyTorch Positional Encoding'),
                  (apply_rotary_embeddings(x, freqs_complex, 'cpu').squeeze().numpy(),
                   'PyTorch Rotary Positional Encoding')
                  ]

    plt.figure(figsize=(14, 6))
    for i in range(len(data_title)):
        plt.subplot(1, 2, 1 + i)
        plt.imshow(data_title[i][0].T, cmap='viridis', aspect='auto', origin='lower')
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title(data_title[i][1])
        plt.colorbar()

    plt.tight_layout()
    plt.savefig('PE and RoPE.png')


def check_rms_norm():
    norm = RMSNorm(D_MODEL)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    assert norm(x).shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def check_silu():
    def silu(x):
        return x * torch.sigmoid(x)

    # 生成输入数据
    x = torch.linspace(-5, 5, 100)
    y = silu(x)

    # 绘制函数图像
    plt.plot(x.numpy(), y.numpy(), label='SiLU')
    plt.xlabel('x')
    plt.ylabel('SiLU(x)')
    plt.title('SiLU Function')
    plt.grid(True)
    plt.legend()
    plt.savefig('SiLU.png')


def check_transformer():
    model_args: ModelArgs = ModelArgs(
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        dim=D_MODEL,
        n_layers=N_LAYER,
        n_heads=N_HEAD,
        vocab_size=VOCAB_SIZE,
        multiple_of=16,
        norm_eps=1e-5,
        device=DEVICE
    )

    model = LlamaModel(model_args).to(DEVICE)
    tokens = torch.randint(low=0, high=100, size=(BATCH_SIZE, 1), dtype=torch.int)
    summary(model, tokens, 1, show_input=True)
    output = model(tokens, 1)
    assert output.shape == (BATCH_SIZE, 1, VOCAB_SIZE)


def check_train():
    pass


def check_inference():
    tokenizer = SentencePieceProcessor()
    with open('prompts.json', 'r') as file:
        data = json.load(file)
    prompts = data['prompts']
    max_gen_len = MAX_SEQ_LEN - 1
    tokenizer.load('/Users/yuan.feng/Downloads/tokenizer.model')

    print('==========================tokenizer==========================')
    prompt_tokens = [tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
                     for prompt in prompts]
    token_ids = [1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 344, 29874, 4932, 357, 1149, 301, 449, 276, 316, 2778,
                 13, 412, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 572, 1878, 330, 3055, 1725, 1149,
                 330, 3055, 1725, 4639, 28754, 13, 1173, 968, 1149]
    print(tokenizer.decode(token_ids))
    print([tokenizer.decode(i) for i in token_ids])

    batch_size = len(prompt_tokens)
    max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    total_len = min(MAX_SEQ_LEN, max_gen_len + max_prompt_len)
    print(total_len)

    pad_id = tokenizer.pad_id()
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=DEVICE)
    print('==========================inference==========================')
    # 把 prompt_tokens 写入
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=DEVICE)
    print(tokens)

    eos_reached = torch.tensor([False] * batch_size, device=DEVICE)
    prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
    cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
    temperature = 0.6
    top_p = 0.9
    for cur_pos in cur_iterator:
        with torch.no_grad():
            logits = torch.randn(batch_size, 1, VOCAB_SIZE)

        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        # shape: (batch_size)
        next_token = next_token.reshape(-1)
        # !!!!!! Only replace token if it is a padding token
        next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        # EOS is reached only if we found an EOS token for a padding position
        eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
        if all(eos_reached):
            break

    out_tokens = []
    out_text = []
    for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
        # Cut to the EOS token, if present
        if tokenizer.eos_id in current_prompt_tokens:
            eos_idx = current_prompt_tokens.index(tokenizer.eos_id)
            current_prompt_tokens = current_prompt_tokens[:eos_idx]
        out_tokens.append(current_prompt_tokens)
        out_text.append(tokenizer.decode(current_prompt_tokens))
    print(out_text)


if __name__ == '__main__':
    VOCAB_SIZE = 32000
    SEQ_LEN = 1
    MAX_SEQ_LEN = 2048
    BATCH_SIZE = 16
    MAX_BATCH_SIZE = 32
    MAX_LEN = 100
    D_MODEL = 512
    N_HEAD = 8
    N_LAYER = 6
    DIM_FF = 256
    DEVICE = 'cpu'

    check_silu()
