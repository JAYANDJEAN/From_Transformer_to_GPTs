import os
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from models import TransformerTorch
from utils import generate, TextDataset, src_lang, tgt_lang, collate_fn, tokenizers, one_epoch


def train_and_translate(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_loader = DataLoader(TextDataset(dt='train'),
                              batch_size=config['batch_size'],
                              collate_fn=collate_fn,
                              shuffle=True)
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
    optimizer = torch.optim.Adam(transformer.parameters(),
                                 lr=config['lr'],
                                 betas=(config['beta_min'], config['beta_max']),
                                 eps=config['eps'])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizers[tgt_lang].token_to_id("<pad>"))
    min_train_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        start_time = timer()
        train_loss = one_epoch(transformer, optimizer, loss_fn, train_loader, device, 'train', epoch)
        val_loss = one_epoch(transformer, optimizer, loss_fn, val_loader, device, 'eval', epoch)
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(transformer.state_dict(), f"{config['save_dir']}translation_de_to_en.pth")
        end_time = timer()
        print((f"Epoch {epoch}: Train loss = {train_loss:.3f}, Val loss = {val_loss:.3f}, "
               f"time = {(end_time - start_time):.3f}s"))

    src_sentences = [
        "Ein schickes Mädchen spricht mit dem Handy während sie langsam die Straße entlangschwebt.",
        # A trendy girl talking on her cellphone while gliding slowly down the street.
        "Eine Frau mit einer großen Geldbörse geht an einem Tor vorbei."
        # A woman with a large purse is walking by a gate.
    ]
    transformer.to('cpu')
    transformer.load_state_dict(
        torch.load(
            f"{config['save_dir']}translation_de_to_en.pth",
            map_location="cpu",
            weights_only=True)
    )
    print("Translated sentence:", generate(transformer, src_sentences, 'cpu'))


if __name__ == '__main__':
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
        'eps': 1e-9
    }
    os.makedirs(train_config['save_dir'], exist_ok=True)
    train_and_translate(train_config)
