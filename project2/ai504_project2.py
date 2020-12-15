from easydict import EasyDict

from google.colab import files
import os

import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import torch.nn as nn
import torch.optim as optim

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm
import json




config = EasyDict({
    "emb_dim":64,
    "ffn_dim":128,
    "attention_heads":8,
    "dropout":0.2518,
    "encoder_layers":3,
    "decoder_layers":2,
    "lr":0.0007894,
    "batch_size":461,
    "nepochs":48,
})

#####      Do not modify      #####
config.max_position_embeddings=512




os.environ['STUDENT_ID']="20195220"

if os.path.isdir('result'):
    !rm -rf result

%mkdir result
%mv config.json model.pt result

!zip $STUDENT_ID.zip result/*
files.download('{}.zip'.format(os.environ['STUDENT_ID']))



!pip install --upgrade torchtext
!python -m spacy download de
!python -m spacy download en
!pip install -Iv --upgrade nltk==3.5


torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 3)
TRG.build_vocab(train_data, min_freq = 3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = config.batch_size,
    device = device,
    shuffle=False)

PAD_IDX = TRG.vocab.stoi['<pad>']




class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer,self).__init__()
        self.encoder_embedding = nn.Embedding(len(SRC.vocab),config.emb_dim)
        self.decoder_embedding = nn.Embedding(len(TRG.vocab),config.emb_dim)
        self.transformer = nn.Transformer(d_model=config.emb_dim, nhead=config.attention_heads, 
                       num_encoder_layers=config.encoder_layers, num_decoder_layers=config.decoder_layers,
                       dim_feedforward=config.ffn_dim, dropout=config.dropout, activation='gelu')
        self.prediction_head = nn.Linear(config.emb_dim,len(TRG.vocab))
        
    def forward(self, src, trg):
        src_emb = self.encoder_embedding(src)
        trg_emb = self.decoder_embedding(trg)
        output = self.transformer(src_emb, trg_emb,
                       tgt_mask=self.transformer.generate_square_subsequent_mask(trg.size(0)).to(device),
                       src_key_padding_mask=src.eq(PAD_IDX).permute(1,0).to(device),
                       memory_key_padding_mask=src.eq(PAD_IDX).permute(1,0).to(device),
                       tgt_key_padding_mask=trg.eq(PAD_IDX).permute(1,0).to(device))
        prediction = self.prediction_head(output)
        return prediction

CLIP = 1
    
model = Transformer(config)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)





best_valid_loss = float('inf')

def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    model.train()

    epoch_loss = 0

    for idx, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[:-1].reshape(-1, output.shape[-1])
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg)            
            
            output = output[:-1].reshape(-1, output.shape[-1])
            
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def measure_BLEU(model: nn.Module,
             iterator: BucketIterator
                ):
    model.eval()
    iterator.batch_size = 1
    BLEU_scores = list()
    
    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg)           
            predicted = [TRG.vocab.itos[token] for token in output[:-1].argmax(dim=2).squeeze().tolist() if token!=PAD_IDX]
            GT = [TRG.vocab.itos[token] for token in trg[1:].squeeze().tolist() if token!=PAD_IDX]
            BLEU_scores.append(sentence_bleu([GT], predicted))
    return sum(BLEU_scores)/len(BLEU_scores)
                         
queue=0
for epoch in tqdm(range(config.nepochs), total=config.nepochs):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    test_bleu = measure_BLEU(model, test_iterator)
    print("Test BLEU score : {}".format(test_bleu))
    print("Epoch : {} / Training loss : {} / Validation loss : {}".format(epoch+1, train_loss, valid_loss))

    if best_valid_loss < valid_loss:
        queue+=1
        if queue>1:
            break
    else:
        best_valid_loss = valid_loss
        queue = 0

test_bleu = measure_BLEU(model, test_iterator)
print("Test BLEU score : {}".format(test_bleu))
        
with open('config.json','w') as f:
    json.dump(vars(config),f)
torch.save(model.state_dict(),'model.pt')