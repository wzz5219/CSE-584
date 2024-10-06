import re
import pandas as pd
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification 
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time, random
from io import StringIO

label_mapping = {'meta': 0, 'gemini': 1, 'openai': 2, 'mistral': 3, 'distilgpt2': 4}  # Extend this mapping as needed
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def process_csv(file_path, label, n_rows):
    print(file_path)
    with open(file_path, 'r') as file:
        content = file.read()    
    content = content.replace('"', '')
    content = StringIO(content)
    df = pd.read_csv(content, sep='|', on_bad_lines='skip')
    print(f"Number of rows in the DataFrame after skipping bad lines: {len(df)}")
    df = df.head(n_rows)
    df['label'] = label
    df['label'] = df['label'].map(label_mapping)
    
    return df

n_rows = int(sys.argv[1])
csv_directory = './data/'
train_data = []
test_data = []
dfs = []
NUM_LABEL = 0

for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_directory, filename)
        NUM_LABEL = NUM_LABEL + 1
        label = filename.split('.')[0]
        df = process_csv(file_path, label, n_rows)
        dfs.append(df)
        ltrain_df, ltest_df = train_test_split(df, test_size=float(sys.argv[3]), random_state=100) 
        train_data.append(ltrain_df)  
        test_data.append(ltest_df)    

if int(sys.argv[4]) == 1 :
    combined_df = pd.concat(dfs, ignore_index=True)
    train_df, test_df = train_test_split(combined_df, test_size=float(sys.argv[3]), random_state=100)
else :
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

print("Training set LLM distribution:")
print(train_df['label'].value_counts())
print("Testing set LLM distribution:")
print(test_df['label'].value_counts())

train_df.to_csv('train.csv', sep='|',index=False)
test_df.to_csv('test.csv', sep='|',index=False)
print(f"number of LLMs {NUM_LABEL}")

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length    
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        inputs = self.tokenizer(
            str(row['Prefix']), str(row['Suffix']),
            padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        label = torch.tensor(row['label'], dtype=torch.long)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': label
        }

cache = "./cache/"


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',  cache_dir=cache, num_labels=NUM_LABEL, 
    hidden_dropout_prob=0.1)  

max_length = 150
train_dataset = TextDataset(train_df, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = TextDataset(test_df, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    torch.set_num_threads(4)
    print(f"Running on CPU with max threads: {torch.get_num_threads()}")
else:
    print("Running on GPU")
model.to(device)

# training
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0 
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples * 100

    return loss, accuracy
    
# testing
def evaluate(model, test_loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    
    preds = np.concatenate(preds)
    true_labels = np.concatenate(true_labels)
    
    overall_accuracy = accuracy_score(true_labels, preds) * 100
    
    # Calculate per-class accuracy
    LLMs = np.unique(true_labels)  # Get unique classes (labels)
    per_LLM_accuracy = {}

    for LLM in LLMs:
        total = np.sum(true_labels == LLM)  # Total instances of this class
        correct = np.sum((preds == LLM) & (true_labels == LLM))  # Correct predictions for this class

        if total > 0:  # To avoid division by zero
            per_LLM_accuracy[LLM] = (correct / total) * 100  # Accuracy as a percentage
        else:
            per_LLM_accuracy[LLM] = 0.0  # If no instances, accuracy is 0%

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')

    return overall_accuracy, per_LLM_accuracy, precision, recall, f1

train_start_time = time.time()
training_pass = int(sys.argv[2])
for i in range(training_pass):
    loss, accuracy = train(model, train_loader, optimizer, device)
    print(f"Pass {i+1}/{training_pass}, Train Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
train_end_time = time.time()
elapsed_time = train_end_time - train_start_time
print(f"Elapsed time in training : {elapsed_time:.6f} seconds")

test_start_time = time.time()
test_accuracy, per_LLM_accuracy, precision, recall, f1 = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision (weighted): {precision:.2f}")
print(f"Recall (weighted): {recall:.2f}")
print(f"F1 Score (weighted): {f1:.2f}")

for LLM, accuracy in per_LLM_accuracy.items():
    print(f'LLM {LLM} Accuracy: {accuracy:.2f}%')
    print(f'LLM-name: {reverse_label_mapping[LLM]} Accuracy: {accuracy:.2f}%')
    
test_end_time = time.time()
elapsed_time = test_end_time - test_start_time
print(f"Elapsed time in testing : {elapsed_time:.6f} seconds")
