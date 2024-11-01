from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from utils import prepare_dataset_eli5


# https://huggingface.co/docs/transformers/tasks/language_modeling
# 这里没啥预测
model_id = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

lm_dataset = prepare_dataset_eli5(tokenizer)

training_args = TrainingArguments(
    output_dir="../00_assets/models/gpt2-finetune-eli5",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
