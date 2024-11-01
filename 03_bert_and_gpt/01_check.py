from transformers import AutoTokenizer
from modelsummary import summary
from utils import prepare_dataset_books, prepare_dataset_geo
import string


def check_tokenizer_and_geo():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    print("raw number of tokenizer:", len(tokenizer))
    new_tokens = [f"<%{i}>" for i in string.ascii_lowercase]
    new_tokens += [f"<%{i}>" for i in range(10)]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    print("new number of tokenizer:", len(tokenizer))
    t_train, t_val, t_test = prepare_dataset_geo(tokenizer)
    print(t_train[:3]['labels'])


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
