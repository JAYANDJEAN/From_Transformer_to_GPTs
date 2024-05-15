from timeit import default_timer as timer
import torch
from tqdm import tqdm
import yaml
import sys
from utils import prepare_loader_from_file, generate_mask, generate
from models import Seq2Seq_Pre_Encode, TransformerTorch
from transformers import AutoTokenizer
import warnings

args = sys.argv


def train_op(params):
    def _epoch(model, dataloader, tp):
        losses = 0
        if tp == 'train':
            model.train()
        elif tp == 'val':
            model.eval()

        for src, src_mask, tgt in dataloader:
            src = src.to(DEVICE)
            tgt_input = tgt[:, :-1].to(DEVICE)
            tgt_out = tgt[:, 1:].to(DEVICE)
            tgt_mask = generate_mask(tgt_input.shape[1]).to(DEVICE)
            src_padding_mask = (src == tokenizer.pad_token_id).to(DEVICE)
            tgt_padding_mask = (tgt_input == tokenizer.pad_token_id).to(DEVICE)

            if params['model_name'] == 'scratch':
                src_mask = torch.zeros((src.shape[1], src.shape[1])).to(DEVICE)
                tgt_predict = model(src, tgt_input, src_mask, tgt_mask,
                                    src_padding_mask, tgt_padding_mask, src_padding_mask)
            else:
                src_mask = src_mask.to(DEVICE)
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

    # model_name = "distilbert/distilroberta-base"
    geo_code_type = params['geo_code_type']
    DEVICE = params['device']
    BATCH_SIZE = params['batch_size']
    model_name = params['model_name']

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

    assert geo_code_type in ('s2', 'geohash')
    columns = ['address', geo_code_type]
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

    tgt_size = len(tgt_vocab)
    src_size = tokenizer.vocab_size

    if model_name == 'scratch':
        transformer = TransformerTorch(num_encoder_layers=params['num_encode'],
                                       num_decoder_layers=params['num_decode'],
                                       d_model=params['d_model'],
                                       n_head=params['n_head'],
                                       src_vocab_size=src_size,
                                       tgt_vocab_size=tgt_size
                                       ).to(DEVICE)
    else:
        transformer = Seq2Seq_Pre_Encode(model_path=model_name, tgt_vocab_size=tgt_size).to(DEVICE)

    csv_file = '../00_assets/csv/addr_to_geo_min.csv'
    train_loader, val_loader, test_loader = prepare_loader_from_file(
        BATCH_SIZE, tokenizer, csv_file, columns, reversed_vocab
    )

    optimizer = torch.optim.Adam(transformer.parameters(),
                                 lr=params['lr'],
                                 betas=eval(params['betas']),
                                 eps=eval(params['eps']))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    min_train_loss = float('inf')
    save_path = f"../00_assets/best_{params['model_version']}.pth"

    print("Total parameters:", sum(p.numel() for p in transformer.parameters() if p.requires_grad))
    print('-------------------train----------------------------')
    for epoch in range(1, params['num_epochs'] + 1):
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
    generate(transformer, tokenizer, test_loader, tgt_vocab, params)


version = 'v1'
with open('../00_assets/yml/' + version + '.yml', 'r') as file:
    config = yaml.safe_load(file)
    print(config)

train_op(config)
