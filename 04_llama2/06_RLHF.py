import torch
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

'''
https://github.com/hkproj/rlhf-ppo
'''

# Load all helpfulness/harmless subsets (share the same schema)
dataset = load_dataset("Anthropic/hh-rlhf")
