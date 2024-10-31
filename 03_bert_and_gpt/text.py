from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

# 加载数据集并划分训练和测试集
dataset = load_dataset("opus_books", "de-en")
dataset = dataset['train'].train_test_split(test_size=0.2)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModel.from_pretrained("google-t5/t5-small")

# 数据预处理函数
def preprocess_function(examples):
    inputs = examples['translation']['de']
    targets = examples['translation']['en']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 应用数据预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./t5-small-finetuned-opus-books",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
)

# 定义评价指标
metric = load_metric("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 将 -100 标签移除并解码为字符串
    labels = [[label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # 计算 BLEU 分数
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    return result

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print(f"BLEU score on validation set: {eval_results['eval_bleu']:.2f}")

# 保存模型
trainer.save_model("./t5-small-finetuned-opus-books")
