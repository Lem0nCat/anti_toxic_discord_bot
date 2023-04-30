import re

import numpy as np
import torch
from textblob import TextBlob
from transformers import (AutoModel, BertTokenizerFast, DistilBertModel,
                          DistilBertTokenizer, RobertaModel, RobertaTokenizer)

from model import BERT_Arch

id2label = {0: 'Not Toxic', 1: 'Toxic'}

# Указываем устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir = f'models/{input("Model name: ")}/model.pth'
# Загружаем модель из файла
data_model = torch.load(dir)

if 'used_model_name' in data_model:
    user_input = data_model['used_model_name']
else:
    user_input = input(
        'Choose a model [0 - BERT Model] [1 - Roberta Model] [2 - DistilBert Model]: ')
if user_input == 'Roberta' or user_input == '1':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    bert = RobertaModel.from_pretrained('roberta-base')
elif user_input == 'DistilBert' or user_input == '2':
    # Загружаем токенизатор DistilBert
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    # Импортируем предварительно обученную модель DistilBert
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
else:
    # Загружаем токенизатор BERT
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # Импортируем предварительно обученную модель на основе BERT
    bert = AutoModel.from_pretrained('bert-base-uncased')


def get_prediction(str):
    str = re.sub(r'[^a-zA-Z ]+', '', str)
    test_text = [str]
    model.eval()

    tokens_test_data = tokenizer(
        test_text,
        max_length=data_model['max_seq_len'],
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])

    preds = None

    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))

    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)

    predicted_class = id2label[preds[0]]
    return predicted_class


model = BERT_Arch(bert).to(device)
model.load_state_dict(data_model['model_state'])
model.eval()

messages = [
    "You're so stupid, you can't even tie your own shoes.",
    "Nobody likes you, why do you even bother trying?",
    "You're a waste of space.",
    "You're such a loser.",
    "You're not good enough for anything.",
    "I hope you fail at everything you do.",
    "You're ugly and nobody will ever love you.",
    "You're a disgrace to your family.",
    "You're a terrible person.",
    "You don't deserve to be happy.",
    "You're a burden to everyone around you.",
    "You're so annoying, nobody wants to be around you.",
    "You're a pathetic excuse for a human being.",
    "You're weak and can't handle anything.",
    "You'll never amount to anything.",
    "Nobody cares about you.",
    "You're a waste of time.",
    "You're a joke.",
    "You're a failure and always will be.",
    "You're a disappointment to everyone who knows you.",
    'Fuck your mom',
    'niggers',
    'Fuck'
]

count = 0
# Исправление ошибок в сообщении
# Making our first textblob
# textBlb = TextBlob(message.content)
# textCorrected = textBlb.correct()
# print(textCorrected)

for message in messages:
    intent = get_prediction(message)
    if intent == 'Toxic':
        count += 1
        # print(f'Toxic message removed: {message}')
    else:
        print(f'Toxic message not removed: {message}')


print(f'Model correct: {(count * 100) / len(messages):.1f}%')
