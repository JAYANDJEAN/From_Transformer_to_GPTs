import string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import prepare_dataset_books, prepare_dataset_geo


def check_tokenizer_and_geo():
    model_id = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("原始tokenizer的大小:", len(tokenizer))
    new_tokens = [f"<%{i}>" for i in string.ascii_lowercase]
    new_tokens += [f"<%{i}>" for i in range(10)]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    print("扩容tokenizer的大小:", len(tokenizer))
    t_train, t_val, t_test = prepare_dataset_geo(tokenizer)
    print("geohash编码后的结果:", t_train[:3]['labels'])

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print(f"原始模型的参数总量: {sum(p.numel() for p in model.parameters())}")
    model.resize_token_embeddings(len(tokenizer))
    print(f"扩容模型的参数总量: {sum(p.numel() for p in model.parameters())}")

    # 打印模型的每一层和参数量
    for name, layer in model.named_modules():
        if len(list(layer.parameters())) > 0:  # 只打印有参数的层
            layer_params = sum(p.numel() for p in layer.parameters())
            print(f"层名称: {name}, 层类型: {layer.__class__.__name__}, 参数量: {layer_params}")


def check_opus_books():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    num_case = 3
    dataset = prepare_dataset_books(tokenizer)
    print(dataset['train'])
    """
        Dataset({
            features: ['id', 'translation', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 46320
        })
    """
    input_ids = dataset['train']['input_ids'][:num_case]
    translation = dataset['train']['translation'][:num_case]
    labels = dataset['train']['labels'][:num_case]  # output_ids
    print('=' * 70)
    for i in range(num_case):
        print("de:", translation[i]['de'])
        print("de ids: ", input_ids[i])
        print("en: ", translation[i]['en'])
        print("en ids: ", labels[i])


if __name__ == '__main__':
    check_tokenizer_and_geo()
