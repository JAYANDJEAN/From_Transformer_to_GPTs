import torch
from timeit import default_timer as timer
from models import TransformerTorch
from utils import generate_mask, prepare_dataset, generate, SPECIAL_IDS, src_lang, tgt_lang
from tqdm import tqdm
import yaml
import os

# if torch.cuda.is_available():
#     device = 'cuda'
# elif torch.backends.mps.is_available():
#     device = 'mps'
# else:
#     device = 'cpu'
# DEVICE = torch.device(device)
DEVICE = 'cpu'


# TransformerTorch 不能在mps设备上跑，
def train_and_translate():
    def _epoch(model, dataloader, tp):
        if tp == 'train':
            model.train()
        elif tp == 'eval':
            model.eval()
        losses = 0
        optimizer = torch.optim.Adam(transformer.parameters(),
                                     lr=config['lr'],
                                     betas=eval(config['betas']),
                                     eps=eval(config['eps']))
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_IDS['<pad>'])
        for src, tgt in dataloader:
            src = src.to(DEVICE)
            tgt_input = tgt[:, :-1].to(DEVICE)
            tgt_out = tgt[:, 1:].to(DEVICE)
            src_mask = torch.zeros((src.shape[1], src.shape[1])).to(DEVICE)
            tgt_mask = generate_mask(tgt_input.shape[1]).to(DEVICE)
            src_padding_mask = (src == SPECIAL_IDS['<pad>']).to(DEVICE)
            tgt_padding_mask = (tgt_input == SPECIAL_IDS['<pad>']).to(DEVICE)

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
            pbar.update(1)

        return losses / len(list(dataloader))

    with open('../00_assets/yml/translation.yml', 'r') as file:
        config = yaml.safe_load(file)
    with open('../00_assets/yml/local_settings.yml', 'r') as file:
        setting = yaml.safe_load(file)
    save_dir = os.path.join(setting['model_path'], 'translation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    text_to_indices, vocabs, train_loader, eval_loader = prepare_dataset(config['batch_size'])
    transformer = TransformerTorch(num_encoder_layers=config['num_encode'],
                                   num_decoder_layers=config['num_decode'],
                                   d_model=config['d_model'],
                                   n_head=config['n_head'],
                                   src_vocab_size=len(vocabs[src_lang]),
                                   tgt_vocab_size=len(vocabs[tgt_lang])
                                   ).to(DEVICE)
    min_train_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        start_time = timer()
        with tqdm(total=len(list(train_loader)) + len(list(eval_loader)), desc=f'Epoch {epoch}', unit='batch') as pbar:
            train_loss = _epoch(transformer, train_loader, 'train')
            val_loss = _epoch(transformer, eval_loader, 'eval')

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
    
    A bald girl talks on a cellphone while talking to her cellphone .
    A woman with a large purse is walking past a gate .
    '''
    src_sentences = [
        "Ein schickes Mädchen spricht mit dem Handy während sie langsam die Straße entlangschwebt.",
        "Eine Frau mit einer großen Geldbörse geht an einem Tor vorbei."
    ]
    transformer.load_state_dict(torch.load(f'{save_dir}/translation_de_to_en.pth', map_location="cpu"))
    print("Translated sentence:", generate(transformer, src_sentences, text_to_indices, vocabs, DEVICE))


train_and_translate()
