import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from timeit import default_timer as timer
import warnings
from tqdm import tqdm
from pathlib import Path
from models import TransformerTorch
import yaml
import os
from utils import generate_mask, TextDataset, src_lang, tgt_lang, collate_fn, tokenizers
import argparse


def train_model(config):
    def _epoch(model, dataloader, tp):
        if tp == 'train':
            model.train()
        elif tp == 'eval':
            model.eval()
        losses = 0

        for src, tgt in dataloader:
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_out = tgt[:, 1:].to(device)
            src_mask = torch.zeros((src.shape[1], src.shape[1])).to(device)
            tgt_mask = generate_mask(tgt_input.shape[1]).to(device)
            src_padding_mask = (src == tokenizers[src_lang].token_to_id("<pad>")).to(device)
            tgt_padding_mask = (tgt_input == tokenizers[tgt_lang].token_to_id("<pad>")).to(device)
            tgt_predict = model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)

            if tp == 'train':
                optimizer.zero_grad()
                loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt_out.reshape(-1))
                loss.backward()
                optimizer.step()
                losses += loss.item()
            elif tp == 'eval':
                loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()

        return losses / len(list(dataloader))

    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {config.local_rank} - Using device: {device}")

    with open('../00_assets/yml/local_settings.yml', 'r') as file:
        setting = yaml.safe_load(file)
    save_dir = os.path.join(setting['model_path'], 'translation')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"GPU {config.local_rank} - Loading dataset...")

    train_loader = DataLoader(TextDataset(dt='train'),
                              batch_size=config.batch_size,
                              collate_fn=collate_fn,
                              shuffle=False,
                              sampler=DistributedSampler(TextDataset(dt='train'), shuffle=True))
    val_loader = DataLoader(TextDataset(dt='val'),
                            batch_size=config.batch_size,
                            collate_fn=collate_fn,
                            shuffle=False)

    model = TransformerTorch(num_encoder_layers=3,
                             num_decoder_layers=3,
                             d_model=config.d_model,
                             n_head=config.n_head,
                             src_vocab_size=tokenizers[src_lang].get_vocab_size(),
                             tgt_vocab_size=tokenizers[tgt_lang].get_vocab_size(),
                             ).to(device)

    # Convert the model to DistributedDataParallel
    # Here we can also specify the bucket_cap_mb parameter to control the size of the buckets
    model = DistributedDataParallel(model, device_ids=[config.local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizers[tgt_lang].token_to_id("<pad>"),
                                        label_smoothing=0.1).to(device)

    for epoch in range(config.num_epochs):
        torch.cuda.empty_cache()
        start_time = timer()
        with tqdm(total=len(list(train_loader)) + len(list(val_loader)),
                  desc=f'Epoch {epoch}', unit='batch') as pbar:
            train_loss = _epoch(model, train_loader, 'train')
            val_loss = _epoch(model, val_loader, 'eval')

            end_time = timer()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
                   f"Val loss: {val_loss:.3f}, "
                   f"Epoch time = {(end_time - start_time):.3f}s"))

        # Only run validation and checkpoint saving on the rank 0 node
        if config.global_rank == 0:
            torch.save(model.state_dict(), f'{save_dir}/translation_{epoch}.pth')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Disable tokenizers parallelism (this is to avoid deadlocks when creating the tokenizers on multiple GPUs)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--model_folder', type=str, default='./model/')

    args = parser.parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.global_rank = int(os.environ['RANK'])

    assert args.local_rank != -1, "LOCAL_RANK environment variable not set"
    assert args.global_rank != -1, "RANK environment variable not set"

    # Print configuration (only once per server)
    if args.local_rank == 0:
        print("Configuration:")
        for key, value in args.__dict__.items():
            print(f"{key:>20}: {value}")

    # Setup distributed training
    init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # Train the model
    train_model(args)

    # Clean up distributed training
    destroy_process_group()
