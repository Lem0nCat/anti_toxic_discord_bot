import re

import numpy as np
import pandas as pd
import torch
import transformers
from model import BERT_Arch

id2label = {0: 'Not Toxic', 1: 'Toxic'}

# Указываем устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dir = f'models/{input("Model name: ")}/model.pth'


def get_bert_model(data_model, filename):
    if 'used_model_name' in data_model:
        user_input = data_model['used_model_name']
    else:
        user_input = input(
            f'Choose a model for file {filename} \n[0 - BERT Model] [1 - Roberta Model] [2 - DistilBert Model] [3 - RU Model]: ')
    if user_input == 'Roberta' or user_input == '1':
        tokenizer = transformers.RobertaTokenizer.from_pretrained(
            'roberta-base')
        bert = transformers.RobertaModel.from_pretrained('roberta-base')
    elif user_input == 'DistilBert' or user_input == '2':
        tokenizer = transformers.DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")
        bert = transformers.DistilBertModel.from_pretrained(
            "distilbert-base-uncased")
    elif user_input == 'RU' or user_input == '3':
        tokenizer = transformers.BertTokenizer.from_pretrained(
            "sberbank-ai/ruBert-base")
        bert = transformers.AutoModel.from_pretrained(
            "sberbank-ai/ruBert-base")
    else:
        tokenizer = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased')
        bert = transformers.AutoModel.from_pretrained('bert-base-uncased')
    return tokenizer, bert


def get_prediction(str):
    str = re.sub(r'[^a-zA-Zа-яА-Я ]+', '', str)
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


dataset_name = input('Dataset name: ')
df = pd.read_csv(f"datasets/{dataset_name}.csv")
df = df[['comment_text', 'toxic']]
num_rows, num_cols = df.shape

files = [
    'ru1000-Mult-e200-lr0.001-bs32-msl70',
    'ru1000-RU-e200-lr0.001-bs32-msl140',
    'ru1250-RU-e200-lr0.001-bs32-msl70',
    'ru1250-RU-e200-lr0.001-bs32-msl140'
]
corrects = []
for file in files:
    try:
        data_model = torch.load(f'models/{file}/model.pth')
        tokenizer, bert = get_bert_model(data_model, file)

        model = BERT_Arch(bert).to(device)
        model.load_state_dict(data_model['model_state'])
        model.eval()

        count = 0
        for index, row in df.iterrows():
            intent = get_prediction(row['comment_text'])
            if intent == id2label[row['toxic']]:
                count += 1
            if index % 500 == 0:
                print(
                    f'Model correct: {(count * 100) / (index + 1):.1f}% {index}/{num_rows}')
        print(f'Final correct: {(count * 100) / num_rows:.1f}%')
        corrects.append((count * 100) / num_rows)
    except:
        corrects.append('error')


for index, file in enumerate(files):
    print(f'File {corrects[index]}% - "{file}"')
