import torch
from timeit import default_timer as timer
from models import TransformerTorch, TransformerScratch
from utils import generate_mask, prepare_dataset, translate, SPECIAL_IDS, src_lang, tgt_lang
from tqdm import tqdm
import warnings
import yaml

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
DEVICE = torch.device(device)
print(device)


# TransformerScratch 可在三种设备上跑
# TransformerTorch 不能在mps设备上跑，
# The operator 'aten::_nested_tensor_from_mask_left_aligned' is not currently implemented for the MPS device.

def train_and_translate(parameters):
    def _epoch(model, dataloader, tp):
        if tp == 'train':
            model.train()
        elif tp == 'eval':
            model.eval()

        losses = 0
        optimizer = torch.optim.Adam(transformer.parameters(),
                                     lr=parameters['lr'],
                                     betas=eval(parameters['betas']),
                                     eps=eval(parameters['eps']))
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_IDS['<pad>'])

        for src, tgt in dataloader:
            src = src.to(DEVICE)
            tgt_input = tgt[:, :-1].to(DEVICE)
            tgt_out = tgt[:, 1:].to(DEVICE)
            src_mask = torch.zeros((src.shape[1], src.shape[1])).to(DEVICE)
            tgt_mask = generate_mask(tgt_input.shape[1]).to(DEVICE)
            src_padding_mask = (src == SPECIAL_IDS['<pad>']).to(DEVICE)
            tgt_padding_mask = (tgt_input == SPECIAL_IDS['<pad>']).to(DEVICE)

            if parameters['model_name'] == 'torch':
                tgt_predict = model(src, tgt_input, src_mask, tgt_mask,
                                    src_padding_mask, tgt_padding_mask, src_padding_mask)
            elif parameters['model_name'] == 'scratch':
                tgt_predict = model(src, tgt_input, src_mask, tgt_mask)
            else:
                tgt_predict = None

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

    text_to_indices, vocabs, train_loader, eval_loader = prepare_dataset(parameters['batch_size'])
    if parameters['model_name'] == 'torch':
        transformer = TransformerTorch(num_encoder_layers=parameters['num_encode'],
                                       num_decoder_layers=parameters['num_decode'],
                                       d_model=parameters['d_model'],
                                       n_head=parameters['n_head'],
                                       src_vocab_size=len(vocabs[src_lang]),
                                       tgt_vocab_size=len(vocabs[tgt_lang])
                                       ).to(DEVICE)
    elif parameters['model_name'] == 'scratch':
        transformer = TransformerScratch(num_encoder_layers=parameters['num_encode'],
                                         num_decoder_layers=parameters['num_decode'],
                                         d_model=parameters['d_model'],
                                         n_head=parameters['n_head'],
                                         src_vocab_size=len(vocabs[src_lang]),
                                         tgt_vocab_size=len(vocabs[tgt_lang])
                                         ).to(DEVICE)
    else:
        transformer = None

    min_train_loss = float('inf')
    model_path = '../00_assets/best_model_' + parameters['model_name'] + '_' + parameters['model_version'] + '.pth'

    for epoch in range(1, parameters['num_epochs'] + 1):
        start_time = timer()
        with tqdm(total=len(list(train_loader)) + len(list(eval_loader)),
                  desc=f'Epoch {epoch}',
                  unit='batch') as pbar:
            train_loss = _epoch(transformer, train_loader, 'train')
            val_loss = _epoch(transformer, eval_loader, 'eval')

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save(transformer.state_dict(), model_path)
            end_time = timer()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
                   f"Val loss: {val_loss:.3f}, "
                   f"Epoch time = {(end_time - start_time):.3f}s"))

    src_sentence = "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche."
    transformer.load_state_dict(torch.load(model_path))
    print("Translated sentence:", translate(transformer, src_sentence, text_to_indices, vocabs, DEVICE))


with open('../00_assets/yml/translation.yml', 'r') as file:
    config = yaml.safe_load(file)

train_and_translate(config)
