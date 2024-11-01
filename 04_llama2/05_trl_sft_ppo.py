from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

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


import torch
from transformers import GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import warnings

warnings.filterwarnings("ignore")
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
ppo_trainer = PPOTrainer(
    PPOConfig(mini_batch_size=1, batch_size=1),
    model,
    model_ref,
    tokenizer
)

query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = ppo_trainer.generate(list(query_tensor), return_prompt=False, **generation_kwargs)
print(response_tensor)
response_txt = tokenizer.decode(response_tensor[0])
print(response_txt)

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)


import torch
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

'''
https://github.com/hkproj/rlhf-ppo
'''

# Load all helpfulness/harmless subsets (share the same schema)
dataset = load_dataset("Anthropic/hh-rlhf")
