import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('./dataset/requests.csv')

df['full_request'] = df['Method'] + " " + df['URL'] + " " + df['content'].fillna('')

# Mapping the 'classification' to a binary label
df['label'] = df['classification'].apply(lambda x: 1 if x == 'anomalous' else 0)

# Split dataset into training and testing sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['full_request'].values, df['label'].values, test_size=0.2, random_state=42)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128)

class HttpRequestDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HttpRequestDataset(train_encodings, train_labels)
val_dataset = HttpRequestDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model.to(device)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    model.train()
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
    print(f'Epoch {epoch+1}: Loss {loss.item()}')

model.eval()
predictions = []
references = []
for batch in val_loader:
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predictions.extend(logits.argmax(dim=1).tolist())
        references.extend(labels.tolist())

print(classification_report(references, predictions))

# Save the model in ./model directory
model.save_pretrained('./model')