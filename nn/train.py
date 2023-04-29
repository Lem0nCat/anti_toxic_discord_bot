import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import BERT_Arch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW, DistilBertModel, DistilBertTokenizer

# Указываем GPU
device = torch.device('cuda')

# Подготовка данных для обучения
df = pd.read_csv("datasets/train_preprocessed.csv")
df = df[['comment_text', 'toxic']]

# переименование столбцов
df = df.rename(columns={'comment_text': 'text', 'toxic': 'label'})

# Заменяем значения
df['label'] = df['label'].replace({0: 'Not Toxic', 1: 'Toxic'})


# Преобразование меток в кодировки
le = LabelEncoder()
# Преобразуем в числа столбец 'label'
df['label'] = le.fit_transform(df['label'])

# Присваиваем столбцам переменные для удобства
train_text, train_labels = df['text'], df['label']

# Загружаем токенизатор DistilBert
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# Импортируем предварительно обученную модель DistilBert
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Получаем длину всех сообщений в наборе тестовых данных
seq_len = [len(i.split()) for i in train_text]

# Строим гистограмму с диапазоном значений = 10
pd.Series(seq_len).hist()
# s.hist(bins=10)
plt.ylabel('Count messages')
plt.xlabel('Length messages')
plt.show()

# Основываясь на гистограмме, мы выбираем максимальную длину (как 8)
max_seq_len = int(input('max_seq_len: '))
# Очищаем график
plt.clf()

# Токенизируем и кодируем последовательности в обучающем наборе
tokens_train = tokenizer(
    train_text.tolist(),
    max_length=max_seq_len,
    # pad_to_max_length=True,
    padding='max_length',
    truncation=True,
    return_token_type_ids=False
)

# Преобразуем целочисленные последовательности в тензоры
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())  # Массив с метками
# print(train_seq, train_mask, train_y)


# Размер пачки для обучения
batch_size = 16

# Преобразуем в тензоры
train_data = TensorDataset(train_seq, train_mask, train_y)

# Сэмплер для выборки данных во время обучения
train_sampler = RandomSampler(train_data)

# DataLoader для набора данных
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size)

# Заморозить все параметры
# Это предотвратит обновление весов модели во время настройки
for param in bert.parameters():
    param.requires_grad = False

model = BERT_Arch(bert)

# Выгружаем модель на GPU
model = model.to(device)


# Определяем оптимизатор
optimizer = AdamW(model.parameters(), lr=1e-3)
# Вычисляем веса классов

class_wts = compute_class_weight(class_weight="balanced",
                                 classes=np.unique(train_labels),
                                 y=train_labels)

# Балансировка весов при вычислении ошибки
# Преобразование весов классов в тензор
weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.to(device)

# Функция потерь
cross_entropy = nn.NLLLoss(weight=weights)

# Пустой список для хранения потерь обучения и проверки каждой эпохи
train_losses = []

# Количество тренировочных эпох
epochs = 200

# Мы также можем использовать планировщик скорости обучения
# для достижения лучших результатов
lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


# Функция для обучения модели
def train():
    model.train()
    total_loss = 0

    # Пустой список для сохранения прогнозов модели
    total_preds = []

    # Итерация по партиям
    for step, batch in enumerate(train_dataloader):

        # Обновление прогресса после каждых 50 партий
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(
                step, len(train_dataloader)))

        # Отправить пачку данных на GPU
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        # Получить прогнозы модели для текущей партии
        preds = model(sent_id, mask)

        # Вычислить потери между фактическими и прогнозируемыми значениями
        loss = cross_entropy(preds, labels)

        # Добавьте к общей сумме потерь
        total_loss = total_loss + loss.item()

        # Обратный проход для вычисления градиентов
        loss.backward()

        # Обрежьте градиенты до 1.0
        # Это помогает предотвратить проблему взрывающегося градиента
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Обновить параметры
        optimizer.step()

        # Очистить рассчитанные градиенты
        optimizer.zero_grad()

        # На данный момент мы не используем планировщик скорости обучения
        # lr_sch.step()

        # Предсказания модели хранятся на GPU
        # Мы переносим их на CPU
        preds = preds.detach().cpu().numpy()

        # Добавьте прогнозы модели к общему списку
        total_preds.append(preds)

    # Вычислите потери при обучении за эпоху
    avg_loss = total_loss / len(train_dataloader)

    # Прогнозы в форме (количество партий, размер партии, количество классов)
    # Измените прогнозы в виде (количество выборок, количество классов)
    total_preds = np.concatenate(total_preds, axis=0)

    # Возвращаем потерю и предсказание модели
    return avg_loss, total_preds


# Обучение модели
for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # Обучение модели
    train_loss, _ = train()

    # Добавляем полученную потерю ко всему списку
    train_losses.append(train_loss)

    # Это может сделать ваш эксперимент воспроизводимым,
    # подобно установке случайного начального числа для всех вариантов,
    # где требуется случайное начальное число.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f'\nTraining Loss: {train_loss:.3f}')


# Выводим результаты тренировки
plt.plot(train_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')


data = {
    "model_state": model.state_dict(),
    "max_seq_len": max_seq_len
}

# filename = input('Введите название файла: ')
filename = '160k'

# Сохраняем график в файл
plt.savefig(f'models/{filename}.png')

FILE = f'models/{filename}.pth'
torch.save(data, FILE)
print(f'File saved to "{FILE}"')
