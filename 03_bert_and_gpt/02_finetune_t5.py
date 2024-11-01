from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline)
import evaluate
import numpy as np
from utils import prepare_dataset_books
import os

# https://huggingface.co/docs/transformers/tasks/translation
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
dataset = prepare_dataset_books(tokenizer)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""
具体来说，data_collator 的作用包括：
1. 动态填充批数据：将不同长度的输入序列填充到相同长度，以满足批处理的形状要求，避免因长度不一致而导致的错误。
2. 创建注意力掩码：对于填充的部分生成相应的注意力掩码，确保模型在训练或推理时忽略填充部分。
3. 处理特殊任务需求：针对特定任务，data_collator 还可以负责其他数据处理步骤，比如在语言模型任务中，
   创建输入序列的遮蔽标签或随机遮蔽部分词汇（如用于 Masked Language Modeling）。
"""
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
metric = evaluate.load("sacrebleu")


def compute_metrics(eval_predicts):
    pred, label = eval_predicts
    if isinstance(pred, tuple):
        pred = pred[0]
    label = np.where(label != -100, label, tokenizer.pad_token_id)
    pred_decoded = tokenizer.batch_decode(pred, skip_special_tokens=True)
    label_decoded = tokenizer.batch_decode(label, skip_special_tokens=True)
    pred_decoded = [pred.strip() for pred in pred_decoded]
    label_decoded = [[label.strip()] for label in label_decoded]
    result = metric.compute(predictions=pred_decoded, references=label_decoded)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in pred]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


training_args = Seq2SeqTrainingArguments(
    output_dir="../00_assets/models/t5-small-finetune-opus-books",
    evaluation_strategy="steps",
    save_strategy="epoch",
    logging_steps=100,
    eval_steps=1000,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    push_to_hub=False,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

text = "translate German to English: Legumes share resources with nitrogen-fixing bacteria."
translator = pipeline("translation_xx_to_yy", model="../00_assets/models/t5-small-finetune-opus-books")
translator(text)
