import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.utils import shuffle

DEBUG = 0

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_f = nn.MSELoss()
        if DEBUG: print(f"logits:{logits}")
        if DEBUG: print(f"labels:{labels}")
        loss = loss_f(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        if DEBUG: print(f"loss: {loss}")
        return (loss, outputs) if return_outputs else loss


### Data
df = pd.read_csv('searched_graphs.tsv', sep = '\t',error_bad_lines=False, names=['text','label'],encoding='utf-8')
df = shuffle(df)

data = pd.DataFrame({
    'text' : df['text'],
    'label' : df['label']
})
data = data[:500]
data = shuffle(data)
data['label'] = data['label'].apply(lambda x:x/100.0)
#data=data.loc[data.index.repeat(100)]
print(data[:10])

### pretrained model
model_name = "google/bert_uncased_L-2_H-128_A-2"#"bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)


# Preprocess data
X = list(data["text"])
y = list(data["label"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
#y_train = y
#X_train = X

word_dict = {'[PAD]': 0, 'x': 1, 't': 2, 'tdg': 3, 'h':4, ';':5, 'cx':6}
for i in range(10):
    word_dict[str(i)] = i + 7
def QToken(data, padding=True, truncation=True, max_length=512 ):    
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
    #number_dict = {i: w for i, w in enumerate(word_dict)}
    print(len(input_ids[0]))
    return {'input_ids':input_ids}

X_train_tokenized = QToken(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = QToken(X_val, padding=True, truncation=True, max_length=512)
#print(X_train_tokenized)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        #print(f"encode: {encodings}")
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

def PairwiseAccuracy(A,B):
    assert(len(A)==len(B))
    n = len(A)
    c = 0.01
    p = 0
    equal = 0
    for i in range(n):
        for j in range(i):
            if (abs(A[i]-A[j]) < 0.0005) and (abs(B[i]-B[j])<0.0005):
                equal += 1
                continue
            c += 1
            if (A[i] > A[j])^(B[i] <= B[j]):
                p += 1
    return   p/c, equal/c

def PairwiseAccuracyGroup(A,B):
    assert(len(A)==len(B))
    n = len(A)
    c = 0.01
    p = 0
    equal = 0
    for i in range(n):
        for j in range(i):
            if (abs(A[i]-A[j]) < 0.01):
                
                if (abs(A[i]-A[j])<0.0005):
                    equal += 1
                    continue
                c += 1
                if (A[i] > A[j])^(B[i] <= B[j]):
                    p += 1
    return   p/c, equal/c

def compute_metrics(p):
    pred, labels = p
    pred = np.squeeze(pred)
    #pred = np.argmax(pred, axis=1)
    print(f"labls:{labels}")
    print(f"pred:{pred}")
    a, e = PairwiseAccuracyGroup(labels,pred) #metrics.mean_squared_error(y_true=labels, y_pred=pred)
    return {'accuracy': a, 'equal': e}

    
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
    learning_rate = 5e-5,
    weight_decay = 0.1
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


