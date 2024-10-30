import os
import warnings
from timeit import default_timer as timer

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models import TransformerTorch
from utils import one_epoch, TextDataset, src_lang, tgt_lang, collate_fn, tokenizers


def train_model(config):
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")

    train_loader = DataLoader(TextDataset(dt='train'),
                              batch_size=config['batch_size'],
                              collate_fn=collate_fn,
                              shuffle=False,
                              sampler=DistributedSampler(TextDataset(dt='train'), shuffle=True))
    val_loader = DataLoader(TextDataset(dt='val'),
                            batch_size=config['batch_size'],
                            collate_fn=collate_fn,
                            shuffle=False)

    transformer = TransformerTorch(num_encoder_layers=config['num_encode'],
                                   num_decoder_layers=config['num_decode'],
                                   d_model=config['d_model'],
                                   n_head=config['n_head'],
                                   src_vocab_size=tokenizers[src_lang].get_vocab_size(),
                                   tgt_vocab_size=tokenizers[tgt_lang].get_vocab_size(),
                                   ).to(device)
    transformer = DistributedDataParallel(transformer, device_ids=[config['local_rank']])

    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizers[tgt_lang].token_to_id("<pad>"),
                                        label_smoothing=0.1).to(device)

    for epoch in range(config['num_epochs']):
        torch.cuda.empty_cache()
        start_time = timer()

        train_loss = one_epoch(transformer, optimizer, loss_fn, train_loader, device, 'train', epoch)
        val_loss = one_epoch(transformer, optimizer, loss_fn, val_loader, device, 'eval', epoch)

        end_time = timer()
        print((f"Epoch {epoch}: GPU = {config['local_rank']}, Train loss = {train_loss:.3f}, "
               f"Val loss = {val_loss:.3f}, time = {(end_time - start_time):.3f}s"))

        # Only run validation and checkpoint saving on the rank 0 node
        if config['global_rank'] == 0:
            torch.save(transformer.state_dict(), f"{config['save_dir']}translation_de_to_en_ddp.pth")


if __name__ == '__main__':
    # torchrun --nnodes=1 --nproc_per_node=1 03_train_ddp_torchrun.py
    train_config = {
        'model_name': 'torch',
        'save_dir': '../00_assets/models/',
        'batch_size': 128,
        'num_epochs': 3,
        'num_encode': 3,
        'num_decode': 3,
        'd_model': 512,
        'n_head': 8,
        'lr': 0.0001,
        'beta_min': 0.9,
        'beta_max': 0.98,
        'eps': 1e-9,
        'local_rank': int(os.environ['LOCAL_RANK']),
        'global_rank': int(os.environ['RANK'])
    }
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(train_config['save_dir'], exist_ok=True)

    assert train_config['local_rank'] != -1, "LOCAL_RANK environment variable not set"
    assert train_config['global_rank'] != -1, "RANK environment variable not set"

    # Setup distributed training
    init_process_group(backend='nccl')
    torch.cuda.set_device(train_config['local_rank'])

    # Train the model
    train_model(train_config)

    # Clean up distributed training
    destroy_process_group()
