import torch

'''
torchrun --nproc_per_node 1 02_completion.py \
    --ckpt_dir /Users/fengyuan/Documents/models/CodeLlama-7b/ \
    --tokenizer_path /Users/fengyuan/Documents/models/CodeLlama-7b/tokenizer.model \
    --max_seq_len 512 --max_batch_size 4
'''
