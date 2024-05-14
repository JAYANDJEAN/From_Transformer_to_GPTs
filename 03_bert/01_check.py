from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

model_name = 'distilroberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
eli5 = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)
batch_size = 64


def check_model():
    print('=================model=======================')
    print(model)
    for name, param in model.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')
    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print('-------------freeze---------------')
    for param in model.embeddings.parameters():
        param.requires_grad = False
    for param in model.encoder.layer[0:4].parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')
    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


def check_forward():
    def collate_fn(batch):
        res = []
        for sample in batch:
            encoded_input = tokenizer(sample['title'], return_tensors='pt')
            res.append(encoded_input['input_ids'].squeeze())
        res_batch = pad_sequence(res, padding_value=tokenizer.pad_token_id, batch_first=True)
        mask = (res_batch != tokenizer.pad_token_id).int()
        return res_batch, mask

    print('=================tokenizer=======================')
    # encode/decode one sentence
    text = "Why there was a 'leap second' added to the end of 2016?"
    encode = tokenizer(text, return_tensors='pt')
    print(encode['input_ids'])
    print(tokenizer.decode(encode['input_ids'].squeeze()))

    train_dataloader = DataLoader(eli5, batch_size=batch_size, collate_fn=collate_fn)
    _, (src, src_mask) = next(enumerate(train_dataloader))
    # 展示第一条结果，应该和上面的encode结果是一样的
    print(src[0, :])
    print(src_mask[0, :])
    print('=================forward=======================')
    output = model(input_ids=src, attention_mask=src_mask)
    memory = output.last_hidden_state
    pooler_output = output.pooler_output
    print(memory.shape)
    print(pooler_output.shape)
    index = 0
    # 这里 pooler 和 last_hidden_state 的第一个为什么不相同呢？是因为 pooler 还经过一个 nn.Linear。
    # this returns the classification token after processing through a linear layer and a tanh activation function.
    print(memory[0, index, :10])
    print(pooler_output[0, :10])


if __name__ == '__main__':
    check_forward()
