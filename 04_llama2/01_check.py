import matplotlib.pyplot as plt
from llama import *
from torch import Tensor
import torch
from modelsummary import summary
from sentencepiece import SentencePieceProcessor
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import yaml


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
    plt.savefig('../00_assets/image/PE and RoPE.png')


def check_rms_norm():
    norm = RMSNorm(D_MODEL)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    assert norm(x).shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def check_silu():
    def silu(x):
        return x * torch.sigmoid(x)

    x = torch.linspace(-5, 5, 100)
    y = silu(x)

    plt.plot(x.numpy(), y.numpy(), label='SiLU')
    plt.xlabel('x')
    plt.ylabel('SiLU(x)')
    plt.title('SiLU Function')
    plt.grid(True)
    plt.legend()
    plt.savefig('../00_assets/image/SiLU.png')


def check_kv_cache():
    dim = 4
    n_heads = 1
    max_seq_len = 10
    max_batch_size = 8
    seq_len = 3
    batch_size = 2
    show_cache = True
    freqs_complex = precompute_theta_pos_frequencies(dim, seq_len, DEVICE)
    model_args: ModelArgs = ModelArgs(
        n_heads=n_heads,
        dim=dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        device=DEVICE
    )
    # 1.每次读一个token，2.一次性读入所有token，这两种方法应该返回同样的结果！
    # 因为模型里的参数存在随机过程，所以通过复制，保持这部分参数相同。
    attention1 = Attention(model_args)
    attention2 = Attention(model_args)
    attention2.load_state_dict(attention1.state_dict())

    x = torch.randn(batch_size, seq_len, dim)
    # 可以观察到两种方法的结果是完全一样的，
    # 一次性读入token，必须加上mask，不然就存在当前token可观察后面token的情况。
    print('Method_1:')
    for i in range(seq_len):
        t = x[:, i, :].unsqueeze(1)
        f = freqs_complex[i:i + 1]
        output1 = attention1(t, i, f, None)
        print('-' * 30)
        if show_cache:
            print('cache_k_1:')
            print(attention1.cache_k[:batch_size, :i + 1].squeeze())
        else:
            print('output_1:')
            print(output1.squeeze())

    print('=' * 70)
    print('Method_2:')
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"),
                                 device=DEVICE),
                      diagonal=1)
    output2 = attention2(x, 0, freqs_complex, mask)
    if show_cache:
        print('cache_k_2:')
        print(attention2.cache_k[:batch_size, :seq_len].squeeze())
    else:
        print('output_2:')
        print(output2.squeeze())


def check_feed_forward():
    # todo
    pass


def check_tokenizer():
    tokenizer = SentencePieceProcessor()
    with open('../00_assets/yml/local_settings.yml', 'r') as file:
        config = yaml.safe_load(file)
    tokenizer.load(config['model_path'] + 'Llama-2-7b/tokenizer.model')

    print('==========================tokenizer==========================')
    # 中文支持的不好把，“气”都没有。
    prompt = "今天是个好天气！"
    print(tokenizer.encode(prompt))
    print([tokenizer.decode(i) for i in
           [29871, 31482, 30408, 30392, 30502, 31076, 30408, 233, 179, 151, 30584]])
    token_ids = [1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 344, 29874, 4932, 357, 1149, 301, 449, 276, 316, 2778,
                 13, 412, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 572, 1878, 330, 3055, 1725, 1149,
                 330, 3055, 1725, 4639, 28754, 13, 1173, 968, 1149]
    print(tokenizer.decode(token_ids))
    print([tokenizer.decode(i) for i in token_ids])


def check_glm_tokenizer():
    tokenizer_path = './chatglm_tokenizer/tokenizer.model'
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
    print(tokenizer.pad_token_id)
    print(tokenizer.bos_token_id)
    print(tokenizer.eos_token_id)
    print(tokenizer.special_tokens['<bos>'])
    print(tokenizer.special_tokens['<eos>'])
    print(tokenizer.special_tokens['<pad>'])


def check_model_and_loss():
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
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    model = LlamaModel(model_args).to(DEVICE)
    tokens = torch.randint(low=0, high=100, size=(BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    summary(model, tokens, 0, show_input=True)
    output = model(tokens, 0)

    assert output.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    loss = loss_fn(output.reshape(-1, output.shape[-1]), tokens.reshape(-1))
    print(loss)


if __name__ == '__main__':
    VOCAB_SIZE = 32000
    SEQ_LEN = 23
    MAX_SEQ_LEN = 2048
    BATCH_SIZE = 6
    MAX_BATCH_SIZE = 32
    MAX_LEN = 100
    D_MODEL = 512
    N_HEAD = 8
    N_LAYER = 6
    DIM_FF = 256
    DEVICE = 'cpu'

    check_rope()
    check_rms_norm()
    check_silu()
    check_kv_cache()
    check_feed_forward()
    check_tokenizer()
    check_model_and_loss()
    check_glm_tokenizer()
