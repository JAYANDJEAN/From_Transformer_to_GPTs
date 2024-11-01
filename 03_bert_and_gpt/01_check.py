from transformers import AutoTokenizer
from modelsummary import summary
from utils import prepare_dataset_books
import string


def check_tokenizer():
    gh = 'qqw7rh5u0r5q'
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    print("eos_token_id:", tokenizer.eos_token_id)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("unk_token_id:", tokenizer.unk_token_id)
    print("raw number of tokenizer:", len(tokenizer))
    new_tokens = [f"<%{i}>" for i in string.ascii_lowercase]
    new_tokens += [f"<%{i}>" for i in range(10)]
    print(new_tokens)
    gh_new = ''.join([f"<%{i}>" for i in gh])
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    print("new number of tokenizer:", len(tokenizer))
    print(tokenizer.encode(gh))
    print(tokenizer.encode(gh_new))


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
    check_opus_books()
