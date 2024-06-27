from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import wandb

wandb.login(key="f2abbefd773fd38b0442492f1bfe727799792595")

dataset = load_dataset("imdb", split="train")

sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=512,
    output_dir="/tmp",
)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=sft_config,
)
trainer.train()