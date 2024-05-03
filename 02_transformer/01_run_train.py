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


def show_parameters():
    BATCH_SIZE = 128
    D_MODEL = 512
    print('--------------------------data------------------------------------')
    '''
    text_to_indices: 将文本转成编号序列
    vocabs: 字典
    pip install -U spacy
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    '''
    text_to_indices, vocabs, train_loader, eval_loader = prepare_dataset(BATCH_SIZE)
    src_size, tgt_size = len(vocabs[src_lang]), len(vocabs[tgt_lang])
    _, (src, tgt) = next(enumerate(train_loader))
    print('src size: ', src.shape)
    print('tgt size: ', tgt.shape)
    print(src[0, :])
    print(tgt[0, :])

    print(text_to_indices['de']('Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'))
    print(text_to_indices['en']('Two young, White males are outside near many bushes.'))
    for i in range(5):
        print(i, vocabs['en'].lookup_token(i))

    '''
    去尾、掐头
    在这段代码中，tgt_input是目标序列的输入，tgt_out是目标序列的输出。
    这样做的目的是将模型的输入和输出对齐起来，使得模型在生成下一个符号时可以根据已生成的符号来预测下一个符号。
    假设目标序列是[<start>, a, b, c, <end>]，其中<start>表示序列的开始，<end>表示序列的结束。
    那么tgt_input就是[<start>, a, b, c]，而tgt_out就是[a, b, c, <end>]。
    这样一来，模型在预测第一个符号时可以根据[<start>]来生成a，在预测第二个符号时可以根据[<start>, a]来生成b，以此类推。
    这种处理方式在训练时可以更好地利用模型的输出来指导模型的训练，提高模型的性能。
    '''
    tgt_input, tgt_out = tgt[:, :-1].to(DEVICE), tgt[:, 1:].to(DEVICE)

    print('-------------------------mask-------------------------------')
    '''
        Shape:
            S is the source sequence length, 
            T is the target sequence length, 
            N is the batch size, 
            E is the feature number
            
            src: (S, E) for unbatched input, (S, N, E) if batch_first=False or (N, S, E) if batch_first=True.
            tgt: (T, E) for unbatched input, (T, N, E) if batch_first=False or (N, T, E) if batch_first=True.
            src_mask: (S, S) or (N⋅num_heads, S, S).
            tgt_mask: (T, T) or (N⋅num_heads, T, T).
            memory_mask: (T, S).
            src_key_padding_mask: (S) for unbatched input otherwise (N, S).
            tgt_key_padding_mask: (T) for unbatched input otherwise (N, T).
            memory_key_padding_mask: (S) for unbatched input otherwise (N, S).
            output: (T, E) for unbatched input, (T, N, E) if batch_first=False or (N, T, E) if batch_first=True.
        '''
    src_mask = torch.zeros((src.shape[1], src.shape[1])).to(DEVICE)
    tgt_mask = generate_mask(tgt_input.shape[1]).to(DEVICE)
    src_padding_mask = (src == SPECIAL_IDS['<pad>']).to(DEVICE)
    tgt_padding_mask = (tgt_input == SPECIAL_IDS['<pad>']).to(DEVICE)
    print(src_mask.shape)  # torch.Size([27, 27])
    print(tgt_mask.shape)  # torch.Size([23, 23])
    print(src_padding_mask.shape)  # torch.Size([128, 27])
    print(tgt_padding_mask.shape)  # torch.Size([128, 23])

    print('-------------------------transformer-------------------------------')
    '''
    原始的句子的shape是(S, N)，经过embedding，是(S, N, E)，加上pos_embedding，依然是(S, N, E)
    '''

    # 模型定义，没什么好讲的
    transformer = TransformerTorch(num_encoder_layers=3,
                                   num_decoder_layers=3,
                                   d_model=D_MODEL,
                                   n_head=8,
                                   src_vocab_size=src_size,
                                   tgt_vocab_size=tgt_size
                                   ).to(DEVICE)
    logits_pred = transformer(src, tgt_input, src_mask, tgt_mask,
                              src_padding_mask, tgt_padding_mask,
                              src_padding_mask)
    print('预测单例展示：')
    print('logits_pred size: ', logits_pred.shape)
    print(logits_pred[:, 0, :])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_IDS['<pad>'])
    loss = loss_fn(logits_pred.reshape(-1, logits_pred.shape[-1]), tgt_out.reshape(-1))
    print(loss)

    print('----------------------------eval-------------------------------')
    memory = transformer.encode(src, src_mask).to(DEVICE)
    print(memory.shape)


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
    model_path = 'best_model_' + parameters['model_version'] + '.pth'

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


with open('scratch_v1.yml', 'r') as file:
    config = yaml.safe_load(file)

train_and_translate(config)
