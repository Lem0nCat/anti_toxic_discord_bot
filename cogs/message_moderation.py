import random
import re

import numpy as np
import torch
from disnake.ext import commands
from transformers import DistilBertModel, DistilBertTokenizer, AutoModel, BertTokenizerFast

from config import *
from nn.model import BERT_Arch

id2label = {0: 'Not Toxic', 1: 'Toxic'}
data = {"intents": [
    {"tag": "Toxic",
     "responses": [":("]},
    {"tag": "Not Toxic",
     "responses": [":)"]}
]}

# Указываем устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Загружаем токенизатор DistilBert
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# Импортируем предварительно обученную модель DistilBert
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

# # Загружаем токенизатор BERT
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# # Импортируем предварительно обученную модель на основе BERT
# bert = AutoModel.from_pretrained('bert-base-uncased')

# Загружаем модель из файла
data_model = torch.load(EN_MODEL_FILE)


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


def get_response(message):
    intent = get_prediction(message)
    for i in data['intents']:
        if i["tag"] == intent:
            result = random.choice(i["responses"])
            break
    return "Intent: " + intent + '\n' + "Response: " + result


model = BERT_Arch(bert).to(device)
model.load_state_dict(data_model['model_state'])
model.eval()


class Speaking(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot or message.content.startswith(PREFIX):
            return

        intent = get_prediction(message.content)

        if intent == 'Toxic':
            await message.delete()
            warning_str = "не ругайтесь!"
            await message.channel.send(f'{message.author.mention}, {warning_str}')
        print(f'Intent: {intent}')


def setup(bot):
    bot.add_cog(Speaking(bot))
