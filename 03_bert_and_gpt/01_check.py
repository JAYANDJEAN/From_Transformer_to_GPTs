from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer
from modelsummary import summary
import torch
from utils import prepare_loader_from_set_eli5, prepare_loader_from_file, prepare_dataset_books

BATCH_SIZE = 64
SRC_SEQ_LEN = 17


def check_tokenizer():
    text = "Why there was a 'leap second' added to the end of 2016?"
    tokenizer_bert = RobertaTokenizer.from_pretrained("distilbert/distilroberta-base")
    tokenizer_gpt = AutoTokenizer.from_pretrained("openai-community/gpt2")
    encode_bert = tokenizer_bert(text, return_tensors='pt', add_special_tokens=False)
    print(tokenizer_bert.decode(encode_bert['input_ids'].squeeze()))
    encode_bert = tokenizer_bert(text, return_tensors='pt', add_special_tokens=True)
    print(tokenizer_bert.decode(encode_bert['input_ids'].squeeze()))

    encode_gpt = tokenizer_gpt(text, return_tensors='pt', add_special_tokens=False)
    print(tokenizer_gpt.decode(encode_gpt['input_ids'].squeeze()))
    encode_gpt = tokenizer_gpt(text, return_tensors='pt', add_special_tokens=True)
    print(tokenizer_gpt.decode(encode_gpt['input_ids'].squeeze()))


def check_mlm_model():
    model = RobertaModel.from_pretrained("distilbert/distilroberta-base")
    print('=================model=======================')
    src = torch.randint(low=0, high=100, size=(BATCH_SIZE, SRC_SEQ_LEN), dtype=torch.int)
    mask = torch.randn(size=(BATCH_SIZE, SRC_SEQ_LEN))
    summary(model, src, mask, show_input=True)
    # for name, param in model.named_parameters():
    #     print(f'{name}: requires_grad={param.requires_grad}')
    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print('-------------freeze---------------')
    for param in model.embeddings.parameters():
        param.requires_grad = False
    for param in model.encoder.layer[0:4].parameters():
        param.requires_grad = False
    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


def check_mlm_forward():
    tokenizer = RobertaTokenizer.from_pretrained("distilbert/distilroberta-base")
    model = RobertaModel.from_pretrained("distilbert/distilroberta-base")
    print('=================tokenizer=======================')
    # encode/decode one sentence
    text = "Why there was a 'leap second' added to the end of 2016?"
    encode = tokenizer(text, return_tensors='pt')
    print(encode['input_ids'])
    print(tokenizer.decode(encode['input_ids'].squeeze()))

    train_dataloader, _ = prepare_loader_from_set_eli5(BATCH_SIZE, tokenizer)
    _, (src, src_mask) = next(enumerate(train_dataloader))
    # 展示第一条结果，应该和上面的encode结果是一样的
    print(src[0, :])
    print(src_mask[0, :])
    print('=================forward=======================')
    output = model(input_ids=src, attention_mask=src_mask)
    memory = output.last_hidden_state
    pooler_output = output.pooler_output
    print(memory.shape)  # torch.Size([64, 56, 768])
    print(pooler_output.shape)  # torch.Size([64, 768])
    index = 0
    # 这里 pooler 和 last_hidden_state 的第一个为什么不相同呢？是因为 pooler 还经过一个 nn.Linear。
    # this returns the classification token after processing through a linear layer and a tanh activation function.
    print(memory[0, index, :10])
    print(pooler_output[0, :10])


def check_gpt_model():
    text = "Kontrakan Pak Haji Ewok, RT.1/RW.6, Kp. Jati Pilar, Kab. Bekasi, Jawa Barat 17530, Indonesia"
    model = AutoModel.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    encode = tokenizer(text, return_tensors='pt')
    print('vocab_size:', tokenizer.vocab_size)

    output = model(**encode)
    last_hidden_state = output.last_hidden_state
    past_key_values = output.past_key_values
    print(last_hidden_state.shape)  # torch.Size([1, 41, 768])


def check_custom_data():
    tokenizer = RobertaTokenizer.from_pretrained("distilbert/distilroberta-base")
    csv_file = '../00_assets/csv/addr_to_geo_min.csv'
    columns = ['address', 's2']
    special_tokens = {
        tokenizer.unk_token_id: tokenizer.unk_token,
        tokenizer.pad_token_id: tokenizer.pad_token,
        tokenizer.bos_token_id: tokenizer.bos_token,
        tokenizer.eos_token_id: tokenizer.eos_token
    }
    tgt_vocab = {4: '5', 5: 'a', 6: 'd', 7: '0', 8: '1', 9: '7', 10: '9', 11: '2', 12: 'f', 13: '3', 14: 'b',
                 15: 'e', 16: '8', 17: '6', 18: 'c', 19: '4'}
    tgt_vocab.update(special_tokens)
    reversed_vocab = {v: k for k, v in tgt_vocab.items()}
    train_dataloader, _, _ = prepare_loader_from_file(BATCH_SIZE, tokenizer, csv_file, columns, reversed_vocab)
    _, (src, src_mask, tgt) = next(enumerate(train_dataloader))
    # 展示第一条结果
    print(src[0, :])
    print(src_mask[0, :])
    print(tgt[0, :])


def check_opus_books():
    tokenizer = RobertaTokenizer.from_pretrained("distilbert/distilroberta-base")
    dataset = prepare_dataset_books(tokenizer)
    print(dataset)


if __name__ == '__main__':
    check_tokenizer()
