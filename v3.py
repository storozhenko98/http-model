import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the CSV file
data = pd.read_csv('./dataset/requests.csv')

# Preprocessing
data['label'] = data['Class'].apply(lambda x: 1 if x == 'Valid' else 0)
data.drop(['Class'], axis=1, inplace=True)

# Handle missing or NaN values in 'POST-Data' column
data['POST-Data'] = data['POST-Data'].fillna('')

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(text):
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # Reduced max_length to 128
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

data['input_ids'], data['attention_mask'] = zip(*data['POST-Data'].apply(tokenize))

# Convert to tensors
labels = torch.tensor(data['label'].values)

# Split the data
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    data[['input_ids', 'attention_mask']].values.tolist(),
    labels,
    test_size=0.2,
    random_state=42
)

train_inputs = [(x[0].clone().detach(), x[1].clone().detach()) for x in train_inputs]
val_inputs = [(x[0].clone().detach(), x[1].clone().detach()) for x in val_inputs]

# Define the model
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_output = self.sigmoid(linear_output)
        return final_output

# Set up training
model = BertClassifier().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop
epochs = 5
batch_size = 8  # Reduced batch size to 8

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    for i in range(0, len(train_inputs), batch_size):
        batch_inputs = train_inputs[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        
        input_ids = torch.stack([x[0] for x in batch_inputs]).to(device)
        attention_mask = torch.stack([x[1] for x in batch_inputs]).to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), batch_labels.float())
        loss.backward()
        optimizer.step()
        
        if (i // batch_size) % 10 == 0:
            print(f"Batch {i // batch_size}/{len(train_inputs) // batch_size}, Loss: {loss.item():.4f}")
    
    # Validation
    val_batch_size = 8  # Reduced validation batch size to 8
    val_preds = []
    
    with torch.no_grad():
        for i in range(0, len(val_inputs), val_batch_size):
            batch_inputs = val_inputs[i:i+val_batch_size]
            
            val_input_ids = torch.stack([x[0] for x in batch_inputs]).to(device)
            val_attention_mask = torch.stack([x[1] for x in batch_inputs]).to(device)
            
            val_outputs = model(val_input_ids, val_attention_mask)
            val_preds.extend(torch.round(val_outputs.squeeze()).cpu().tolist())
        
        accuracy = accuracy_score(val_labels.cpu(), val_preds)
        precision = precision_score(val_labels.cpu(), val_preds)
        recall = recall_score(val_labels.cpu(), val_preds)
        f1 = f1_score(val_labels.cpu(), val_preds)
        
        print(f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Save the model
torch.save(model.state_dict(), 'bert_classifier.pth')
print("Model saved.")

# Load the saved model
loaded_model = BertClassifier().to(device)
loaded_model.load_state_dict(torch.load('bert_classifier.pth'))
loaded_model.eval()
print("Model loaded.")

# Inference
def predict(text):
    input_ids, attention_mask = tokenize(text)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    with torch.no_grad():
        output = loaded_model(input_ids, attention_mask)
        prediction = torch.round(output.squeeze())
    return prediction.item()

# Example usage
text = "GET /search?q=python HTTP/1.1"
prediction = predict(text)
print(f"Prediction: {'Valid' if prediction == 1 else 'Anomalous'}")