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
