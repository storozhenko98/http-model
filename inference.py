import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
model_path = './model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Set the model to evaluation mode
model.eval()
# Load your dataset
df = pd.read_csv('./dataset/requests.csv')

# Prepare some samples
sample_data = df.sample(1000)  # Adjust the number as needed
sample_texts = sample_data['Method'] + " " + sample_data['URL'] + " " + sample_data['content'].fillna('')

encodings = tokenizer(sample_texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=1)

# Display the results
for text, prediction in zip(sample_texts, predictions):
    print("Request:", text)
    print("Is anomalous (1) or not (0):", prediction.item())
    print("----------")
