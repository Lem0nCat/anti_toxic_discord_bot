import re

import numpy as np
import torch
from disnake.ext import commands
from transformers import (AutoModel, BertTokenizerFast, DistilBertModel,
                          DistilBertTokenizer, RobertaModel, RobertaTokenizer)

from config import *
from nn.model import BERT_Arch
from utils.databases import UsersDataBase

id2label = {0: 'Not Toxic', 1: 'Toxic'}
data = {"intents": [
    {"tag": "Toxic",
     "responses": [":("]},
    {"tag": "Not Toxic",
     "responses": [":)"]}
]}

# Указываем устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем модель из файла
data_model = torch.load(EN_MODEL_FILE)


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


class MessageModeration(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = UsersDataBase()

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot or message.content.startswith(PREFIX):
            return

        intent = get_prediction(message.content)
        if intent == 'Toxic':
            await message.delete()
            warning_str = "не ругайтесь!"
            await message.channel.send(f'{message.author.mention}, {warning_str}')
            print(f'Toxic message removed: {message.content}')


            ctx = await self.bot.get_context(message)
            await ctx.invoke(self.bot.get_slash_command('warn'), user=message.author, reason='The manifestation of toxicity in the chat')


def setup(bot):
    bot.add_cog(MessageModeration(bot))
