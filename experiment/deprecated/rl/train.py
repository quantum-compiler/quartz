import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DEBUG = 0


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_f = nn.MSELoss()
        if DEBUG:
            print(f"logits:{logits}")
        if DEBUG:
            print(f"labels:{labels}")
        loss = loss_f(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        if DEBUG:
            print(f"loss: {loss}")
        label_diff = pairwise_matrix(labels)
        logits = torch.squeeze(logits)
        logits_diff = pairwise_matrix(logits)
        loss_pw = pairwise_loss(label_diff, logits_diff)
        if DEBUG:
            print(f"loss_pw: {loss_pw}")
        pairwise_weight = 0.3
        loss += loss_pw * pairwise_weight
        return (loss, outputs) if return_outputs else loss


def pairwise_matrix(a):
    n = len(a)
    A = a.repeat(n, 1)
    B = a.reshape([n, 1]).repeat(1, n)
    C = A - B
    return C


def pairwise_loss(A, B):
    relu = nn.ReLU()
    diff = -(A * B)
    r = relu(diff)
    loss = torch.mean(torch.mean(r))
    return loss


### Data
df = pd.read_csv(
    'searched_graphs.tsv',
    sep='\t',
    error_bad_lines=False,
    names=['text', 'label'],
    encoding='utf-8',
)
df = shuffle(df)

data = pd.DataFrame({'text': df['text'], 'label': df['label']})
data = data[:500]
data = shuffle(data)
data['label'] = data['label'].apply(lambda x: x / 100.0)
# data=data.loc[data.index.repeat(100)]
print(data[:10])

### pretrained model
model_name = "google/bert_uncased_L-2_H-128_A-2"  # "bert-base-uncased"
model = QBertForSequenceClassification.from_pretrained(model_name, num_labels=1)


# Preprocess data
X = list(data["text"])
y = list(data["label"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
# y_train = y
# X_train = X

word_dict = {'[CLS]': 0, '[PAD]': 1, 't': 2, 'tdg': 3, 'h': 4, ';': 5, 'x': 6, 'cx': 7}
for i in range(20):
    word_dict[str(i)] = i + 8


def QToken(data, padding=True, truncation=True, max_length=512):
    input_ids = []
    for line in data:
        words = line.split()
        cur = [word_dict[x] for x in words]
        if max_length > len(cur):
            if padding:
                pad_n = max_length - len(cur)
                cur.extend([0] * pad_n)
        else:
            if truncation:
                cur = cur[:max_length]
        input_ids.append(cur)
    # number_dict = {i: w for i, w in enumerate(word_dict)}
    print(len(input_ids[0]))
    return {'input_ids': input_ids}


def QTokenCat(data, padding=True, truncation=True, max_length=512):
    input_ids, qubit_ids, qubit2_ids = [], [], []
    for line in data:
        nodes = line.split(';')[:-1]

        gates = [x.strip().split() for x in nodes]
        # print([len(x) for x in gates])
        gate_tuple = [(x[0], x[1], x[2] if len(x) > 2 else -2) for x in gates]
        gate, qubit, qubit2 = list(zip(*gate_tuple))
        # print(gate)
        # print(qubit2)
        gate_id = [1] + [word_dict[x] for x in list(gate)]
        qubit_id = [1] + [int(x) + 2 for x in list(qubit)]
        qubit2_id = [1] + [int(x) + 2 for x in list(qubit2)]

        if max_length > len(gate_id):
            if padding:
                pad_n = max_length - len(gate_id)
                gate_id.extend([0] * pad_n)
                qubit_id.extend([0] * pad_n)
                qubit2_id.extend([0] * pad_n)
        else:
            if truncation:
                gate_id = gate_id[:max_length]
                qubit_id = qubit_id[:max_length]
                qubit2_id = qubit2_id[:max_length]

        input_ids.append(gate_id)
        qubit_ids.append(qubit_id)
        qubit2_ids.append(qubit2_id)

    # number_dict = {i: w for i, w in enumerate(word_dict)}
    print(len(input_ids[0]))
    return {
        'input_ids': input_ids,
        'token_type_ids': qubit_ids,
        'position_ids': qubit2_ids,
    }


def GetPair(data):
    data = data.sort_values(by='label')
    data['count'] = data['text'].apply(lambda x: len(x.split(';')))
    # data['count'].value_counts()
    k = pd.merge(data, data, on=['count'])
    k = k[k.label_x != k.label_y]
    k['label'] = k['label_x'] > k['label_y']
    return k


def QTokenPW(data, data2, padding=True, truncation=True, max_length=512):
    input_ids = []
    data = list(data)
    data2 = list(data2)
    print(len(data))
    print(len(data2))
    assert len(data) == len(data2)
    for i in range(len(data)):
        words = data[i].split()
        cur = [word_dict[x] for x in words]

        cur = [1] + cur
        if max_length > len(cur):
            if padding:
                pad_n = max_length - len(cur)
                cur.extend([0] * pad_n)
        else:
            if truncation:
                cur = cur[:max_length]
        line = cur
        print(line)
        words = data2[i].split()
        cur = [word_dict[x] for x in words]
        cur = [1] + cur
        if max_length > len(cur):
            if padding:
                pad_n = max_length - len(cur)
                cur.extend([0] * pad_n)
        else:
            if truncation:
                cur = cur[:max_length]
        line += cur
        print(cur)
        input_ids.append(line)
    # number_dict = {i: w for i, w in enumerate(word_dict)}
    print(len(input_ids[0]))
    return {'input_ids': input_ids}


X_train_tokenized = QTokenCat(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = QTokenCat(X_val, padding=True, truncation=True, max_length=512)

# X_train_tokenized = QToken(X_train, padding=True, truncation=True, max_length=512)
# X_val_tokenized = QToken(X_val, padding=True, truncation=True, max_length=512)
# print(X_train_tokenized)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        # print(f"encode: {encodings}")
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)


def PairwiseAccuracy(A, B):
    assert len(A) == len(B)
    n = len(A)
    c = 0.01
    p = 0
    equal = 0
    for i in range(n):
        for j in range(i):
            if (abs(A[i] - A[j]) < 0.0005) and (abs(B[i] - B[j]) < 0.0005):
                equal += 1
                continue
            c += 1
            if (A[i] > A[j]) ^ (B[i] <= B[j]):
                p += 1
    return p / c, equal / c


def PairwiseAccuracyGroup(A, B):
    assert len(A) == len(B)
    n = len(A)
    c = 0.01
    p = 0
    equal = 0
    for i in range(n):
        for j in range(i):
            if abs(A[i] - A[j]) < 0.01:

                if abs(A[i] - A[j]) < 0.0005:
                    equal += 1
                    continue
                c += 1
                if (A[i] > A[j]) ^ (B[i] <= B[j]):
                    p += 1
    return p / c, equal / c


def compute_metrics(p):
    pred, labels = p
    pred = np.squeeze(pred)
    # pred = np.argmax(pred, axis=1)
    print(f"labls:{labels}")
    print(f"pred:{pred}")
    a, e = PairwiseAccuracyGroup(
        labels, pred
    )  # metrics.mean_squared_error(y_true=labels, y_pred=pred)
    l1 = mean_absolute_error(labels, pred)
    return {'accuracy': a, 'equal': e, 'l1': l1}


### Trainer
args = TrainingArguments(
    output_dir="output2",
    evaluation_strategy="steps",
    eval_steps=5,
    per_device_train_batch_size=80,
    per_device_eval_batch_size=80,
    num_train_epochs=100,
    seed=0,
    load_best_model_at_end=True,
    learning_rate=5e-5,
    weight_decay=0.1,
)
trainer = RegressionTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train
trainer.train()
