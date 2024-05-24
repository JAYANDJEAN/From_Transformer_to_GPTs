from timeit import default_timer as timer
import torch
from tqdm import tqdm
import yaml
from utils import prepare_loader_from_file, generate_mask, generate
from models import EncoderDecoderModel
from transformers import AutoTokenizer


def train_op(config):
    def _epoch(model, dataloader, tp):
        losses = 0
        if tp == 'train':
            model.train()
        elif tp == 'val':
            model.eval()

        for src, src_mask, tgt in dataloader:
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_out = tgt[:, 1:].to(device)
            tgt_mask = generate_mask(tgt_input.shape[1]).to(device)

            src_mask = src_mask.to(device)
            tgt_predict = model(src, src_mask, tgt_input, tgt_mask)

            if tp == 'train':
                optimizer.zero_grad()
                loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt_out.reshape(-1))
                loss.backward()
                optimizer.step()
                losses += loss.item()
            elif tp == 'val':
                loss = loss_fn(tgt_predict.reshape(-1, tgt_predict.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()
            if pbar:
                pbar.update(1)
        return losses / len(list(dataloader))

    geo_code_type = config['geo_code_type']
    device = config['device']
    batch_size = config['batch_size']
    csv_file = config['csv_file']
    model_id = "distilbert/distilroberta-base"
    save_path = f"../00_assets/best_{config['model_version']}.pth"

    assert geo_code_type in ('s2', 'geohash')
    columns = ['address', geo_code_type]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    special_tokens = {
        tokenizer.unk_token_id: tokenizer.unk_token,
        tokenizer.pad_token_id: tokenizer.pad_token,
        tokenizer.bos_token_id: tokenizer.bos_token,
        tokenizer.eos_token_id: tokenizer.eos_token
    }
    tgt_gh_vocab = {4: 'q', 5: 'w', 6: 'u', 7: 'g', 8: '8', 9: 'x', 10: '0', 11: 'r', 12: 'h', 13: 'v', 14: 'z',
                    15: '2', 16: 'p', 17: 'y', 18: '6', 19: 'b', 20: 't', 21: '4', 22: 'n', 23: 's', 24: '5', 25: 'm',
                    26: 'f', 27: '1', 28: 'j', 29: '3', 30: 'c', 31: '7', 32: '9', 33: 'd', 34: 'k', 35: 'e'}
    tgt_s2_vocab = {4: '5', 5: 'a', 6: 'd', 7: '0', 8: '1', 9: '7', 10: '9', 11: '2', 12: 'f', 13: '3', 14: 'b',
                    15: 'e', 16: '8', 17: '6', 18: 'c', 19: '4'}
    tgt_vocab = tgt_s2_vocab if geo_code_type == 's2' else tgt_gh_vocab
    tgt_vocab.update(special_tokens)
    reversed_vocab = {v: k for k, v in tgt_vocab.items()}

    # =========================================================== #

    transformer = EncoderDecoderModel(model_id, len(tgt_vocab)).to(device)
    train_loader, val_loader, test_loader = prepare_loader_from_file(
        batch_size, tokenizer, reversed_vocab, csv_file, columns
    )
    optimizer = torch.optim.Adam(transformer.parameters(),
                                 lr=config['lr'],
                                 betas=(config['beta_min'], config['beta_max']),
                                 eps=config['eps'])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    min_train_loss = float('inf')

    print("Total parameters:", sum(p.numel() for p in transformer.parameters() if p.requires_grad))
    print('-------------------train----------------------------')
    for epoch in range(1, config['num_epochs'] + 1):
        start_time = timer()
        with tqdm(total=len(train_loader) + len(val_loader),
                  desc=f'Epoch {epoch}',
                  unit='batch') as pbar:
            train_loss = _epoch(transformer, train_loader, 'train')
            val_loss = _epoch(transformer, val_loader, 'val')

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save(transformer.state_dict(), save_path)
            end_time = timer()
            print((f"\nEpoch: {epoch}, Train loss: {train_loss:.3f}, "
                   f"Val loss: {val_loss:.3f}, "
                   f"Epoch time = {(end_time - start_time):.3f}s"))

    print('-------------------inference----------------------------')
    transformer.load_state_dict(torch.load(save_path))
    generate(transformer, tokenizer, test_loader, tgt_vocab, config)


with open('../00_assets/yml/ft_encode_decode.yml', 'r') as file:
    configs = yaml.safe_load(file)
    print(configs)

train_op(configs)
