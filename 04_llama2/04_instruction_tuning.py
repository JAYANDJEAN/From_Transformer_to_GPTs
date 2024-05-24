from torch.utils.data import Dataset
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from timeit import default_timer as timer
from utils import init_model, InstructionDataset
import torch
import os
import yaml

# Instruction Tuning


if __name__ == "__main__":
    with open('../00_assets/yml/tiny_chinese_llama.yml', 'r') as file:
        config = yaml.safe_load(file)
    with open('../00_assets/yml/local_settings.yml', 'r') as file:
        setting = yaml.safe_load(file)

    data_input = setting['model_path'] + 'alpaca_gpt4_data_zh.json'
    save_dir = os.path.join(setting['model_path'], 'tiny_llama')
    config['init_from'] = 'resume'
    config['save_dir'] = save_dir

    tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

    train_loader = torch.utils.data.DataLoader(InstructionDataset(data_input, tokenizer, max_length=256),
                                               batch_size=1, pin_memory=False, drop_last=False,
                                               shuffle=False, num_workers=0)
    llama_model = init_model(config)

    optimizer = torch.optim.Adam(llama_model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    print("Total trainable parameters:", sum(p.numel() for p in llama_model.parameters() if p.requires_grad))
    min_train_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        start_time = timer()
        with tqdm(total=len(list(train_loader)), desc=f'Epoch {epoch}', unit='batch') as pbar:
            llama_model.train()
            losses = 0
            for i, (src, tgt, loss_mask) in enumerate(train_loader):
                src = src.to(config['device'])
                tgt = tgt.to(config['device'])
                loss_mask = loss_mask.to(config['device'])
                tgt_predict = llama_model(src, 0).to(config['device'])
                optimizer.zero_grad()
                loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt.reshape(-1))
                loss_mask = loss_mask.view(-1)
                loss = torch.sum(loss * loss_mask) / loss_mask.sum()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                pbar.update(1)

            train_loss = losses / len(list(train_loader))
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save(llama_model.state_dict(), '{}/best_chat.pth'.format(save_dir))
            end_time = timer()
            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
