import re

import numpy as np
import torch
import transformers
from disnake.ext import commands

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


def get_bert_model(data_model):
    if 'used_model_name' in data_model:
        user_input = data_model['used_model_name']
    else:
        user_input = input(
            'Choose a model [0 - BERT Model] [1 - Roberta Model] [2 - DistilBert Model] [3 - RU Model]: ')
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


async def get_prediction(str, model, tokenizer, data_model):
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

# Загружаем модель из файла
ru_data_model = torch.load(RU_MODEL_FILE, map_location=device)   
ru_tokenizer, ru_bert = get_bert_model(ru_data_model)
ru_model = BERT_Arch(ru_bert).to(device)
ru_model.load_state_dict(ru_data_model['model_state'])
ru_model.eval()

en_data_model = torch.load(EN_MODEL_FILE, map_location=device)
en_tokenizer, en_bert = get_bert_model(en_data_model)
en_model = BERT_Arch(en_bert).to(device)
en_model.load_state_dict(en_data_model['model_state'])
en_model.eval()


class MessageModeration(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot or message.content.startswith(PREFIX):
            return

        intent = await get_prediction(message.content, en_model, en_tokenizer, en_data_model)
        if intent == 'Not Toxic':
            intent = await get_prediction(message.content, ru_model, ru_tokenizer, ru_data_model)
        if intent == 'Toxic':
            await message.delete()
            print(f'Toxic message removed: {message.content}')
            ctx = await self.bot.get_context(message)
            await ctx.invoke(self.bot.get_slash_command('warn'), user=message.author, reason='The manifestation of toxicity in the chat')


def setup(bot):
    bot.add_cog(MessageModeration(bot))
