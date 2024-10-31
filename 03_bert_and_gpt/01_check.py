from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer
from modelsummary import summary
import torch
from utils import prepare_loader_from_file, prepare_dataset_books

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
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    print("All special tokens:", tokenizer.special_tokens_map)
    print("eos_token_id:", tokenizer.eos_token_id)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("unk_token_id:", tokenizer.unk_token_id)
    num_case = 3
    dataset = prepare_dataset_books(tokenizer)
    input_ids = dataset['train']['input_ids'][:num_case]
    translation = dataset['train']['translation'][:num_case]
    labels = dataset['train']['labels'][:num_case]
    print('=' * 70)
    for i in range(num_case):
        print("de:", translation[i]['de'])
        print("de ids: ", input_ids[i])
        print("en: ", translation[i]['en'])
        print("en ids: ", labels[i])


if __name__ == '__main__':
    check_opus_books()
