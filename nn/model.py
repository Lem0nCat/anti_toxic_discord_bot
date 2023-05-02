import torch.nn as nn


class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        # Слой dropout
        self.dropout = nn.Dropout(0.2)

        # Активационная функция ReLU
        self.relu = nn.ReLU()

        # Dense layer
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        # Функция активации Softmax
        self.softmax = nn.LogSoftmax(dim=1)

    # Определяем функцию forward

    def forward(self, sent_id, mask):
        # Передаем входные данные модель
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Выходной слой
        x = self.fc3(x)

        # Применить активацию softmax
        x = self.softmax(x)
        return x
