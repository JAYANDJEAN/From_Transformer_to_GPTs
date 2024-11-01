from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
import evaluate
from utils import prepare_dataset_geo
import string
import re
import os
import geohash2
from s2sphere import CellId
from geopy.distance import geodesic

# https://huggingface.co/docs/transformers/tasks/translation

model_id = "google-t5/t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_id)
new_tokens = [f"<%{i}>" for i in string.ascii_lowercase]
new_tokens += [f"<%{i}>" for i in range(10)]
tokenizer.add_tokens(new_tokens, special_tokens=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))

t_train, t_val, t_test = prepare_dataset_geo(tokenizer)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""
具体来说，data_collator 的作用包括：
1. 动态填充批数据：将不同长度的输入序列填充到相同长度，以满足批处理的形状要求，避免因长度不一致而导致的错误。
2. 创建注意力掩码：对于填充的部分生成相应的注意力掩码，确保模型在训练或推理时忽略填充部分。
3. 处理特殊任务需求：针对特定任务，data_collator 还可以负责其他数据处理步骤，比如在语言模型任务中，
   创建输入序列的遮蔽标签或随机遮蔽部分词汇（如用于 Masked Language Modeling）。
"""
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_id)
metric = evaluate.load("sacrebleu")


def cal_distance(gc1, gc2, tp):
    if tp == 'geohash':
        try:
            latlon1 = geohash2.decode(gc1)
            latlon2 = geohash2.decode(gc2)
            distance = geodesic(latlon1, latlon2).meters
            return distance
        except:
            return None
    elif tp == 's2':
        try:
            cell_id1 = CellId.from_token(hex(int(gc1, 16)))
            cell_id2 = CellId.from_token(hex(int(gc2, 16)))
            lat_lng1 = cell_id1.to_lat_lng()
            lat_lng2 = cell_id2.to_lat_lng()
            distance = lat_lng1.get_distance(lat_lng2).radians * 6371000
            return distance
        except:
            return None
    else:
        return None


def compute_metrics_geo(eval_outputs):
    predictions = eval_outputs.predictions
    labels = eval_outputs.label_ids
    predictions_decode = tokenizer.batch_decode(predictions)
    labels_decode = tokenizer.batch_decode(labels)
    predictions_decode = [''.join(re.findall(r'<%([a-z0-9])>', i)) for i in predictions_decode]
    labels_decode = [''.join(re.findall(r'<%([a-z0-9])>', i)) for i in labels_decode]
    results = [cal_distance(pred, label, 'geohash') for pred, label in zip(predictions_decode, labels_decode)]
    num = len(results)
    error_geo = sum(item is None for item in results) / num
    acc_100 = sum(isinstance(item, (int, float)) and item < 100 for item in results)
    acc_200 = sum(isinstance(item, (int, float)) and item < 200 for item in results)
    acc_500 = sum(isinstance(item, (int, float)) and item < 500 for item in results)
    error_5000 = sum(isinstance(item, (int, float)) and item > 5000 for item in results)
    error_10000 = sum(isinstance(item, (int, float)) and item > 10000 for item in results)
    result = {
        'acc_100': acc_100,
        'acc_200': acc_200,
        'acc_500': acc_500,
        'error_5000': error_5000,
        'error_10000': error_10000,
        'error_geo': error_geo
    }
    return result


training_args = Seq2SeqTrainingArguments(
    output_dir="../00_assets/models/t5-small-finetune-geo",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    num_train_epochs=50,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    push_to_hub=False,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=t_train,
    eval_dataset=t_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_geo,
)

trainer.train()
