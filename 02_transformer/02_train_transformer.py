import torch
from timeit import default_timer as timer
from models import TransformerTorch
from utils import generate_mask, generate, TextDataset, src_lang, tgt_lang, collate_fn, tokenizers
from tqdm import tqdm
import yaml
import os
from torch.utils.data import DataLoader


def train_and_translate():
    def _epoch(model, dataloader, tp):
        if tp == 'train':
            model.train()
        elif tp == 'eval':
            model.eval()
        epoch_loss = 0

        with tqdm(total=len(list(dataloader)), desc=f'{tp}: Epoch {epoch}', unit='batch') as pbar:
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
                    epoch_loss += loss.item()
                elif tp == 'eval':
                    loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt_out.reshape(-1))
                    epoch_loss += loss.item()
                pbar.update(1)
        return epoch_loss / len(list(dataloader))

    with open('../00_assets/yml/translation.yml', 'r') as file:
        config = yaml.safe_load(file)
    with open('../00_assets/yml/local_settings.yml', 'r') as file:
        setting = yaml.safe_load(file)
    save_dir = os.path.join(setting['model_path'], 'translation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = config['device']
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
        train_loss = _epoch(transformer, train_loader, 'train')
        val_loss = _epoch(transformer, val_loader, 'eval')
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(transformer.state_dict(), f'{save_dir}/translation_de_to_en.pth')
        end_time = timer()
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
               f"Val loss: {val_loss:.3f}, "
               f"Epoch time = {(end_time - start_time):.3f}s"))
    '''
    A trendy girl talking on her cellphone while gliding slowly down the street.
    A woman with a large purse is walking by a gate.
    '''
    src_sentences = [
        "Ein schickes Mädchen spricht mit dem Handy während sie langsam die Straße entlangschwebt.",
        "Eine Frau mit einer großen Geldbörse geht an einem Tor vorbei."
    ]
    transformer.load_state_dict(torch.load(f'{save_dir}/translation_de_to_en.pth', map_location="cpu"))
    print("Translated sentence:", generate(transformer, src_sentences, device))


if __name__ == '__main__':
    train_and_translate()
