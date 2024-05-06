import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Set the device to MPS if available, otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the model architecture
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the saved model state dictionary
model_path = './model2/http_request_classifier.pt'
model.load_state_dict(torch.load(model_path, map_location=device))

# Set the model to evaluation mode
model.eval()

# Move the model to the MPS device
model.to(device)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the maximum sequence length
max_length = 512

# Function to preprocess and tokenize the input text
def preprocess_text(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    return input_ids, attention_mask

# Test cases
test_cases = [
    {
        "name": "Normal Request",
        "request": "GET /example/search?q=normal+query HTTP/1.1"
    },
    {
        "name": "SQL Injection",
        "request": "GET /example/search?q='+OR+1=1--' HTTP/1.1"
    },
    {
        "name": "Cross-Site Scripting (XSS)",
        "request": "POST /example/comment HTTP/1.1\r\nContent-Type: application/x-www-form-urlencoded\r\n\r\ncomment=<script>alert('XSS Attack!');</script>"
    },
    {
        "name": "Directory Traversal",
        "request": "GET /example/../../../etc/passwd HTTP/1.1"
    },
    {
        "name": "Command Injection",
        "request": "POST /example/ping HTTP/1.1\r\nContent-Type: application/x-www-form-urlencoded\r\n\r\nhost=127.0.0.1;+cat+/etc/passwd"
    },
    {
        "name": "Buffer Overflow",
        "request": "GET /example/vuln?param=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA HTTP/1.1"
    }
]

# Test the model with different cases
for case in test_cases:
    print(f"Testing: {case['name']}")
    input_ids, attention_mask = preprocess_text(case['request'])
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    
    print(f"Request: {case['request']}")
    print(f"Predicted Label: {'Malicious' if predicted_label == 1 else 'Benign'}")
    print()