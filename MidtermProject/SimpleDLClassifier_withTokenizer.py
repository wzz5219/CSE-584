import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time, random
from transformers import BertTokenizer
from io import StringIO

label_mapping = {'meta': 0, 'gemini': 1, 'openai': 2, 'mistral': 3, 'distilgpt2': 4}
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
cache = "./cache/"

for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_directory, filename)
        NUM_LABEL += 1
        label = filename.split('.')[0]
        df = process_csv(file_path, label, n_rows)
        dfs.append(df)
        ltrain_df, ltest_df = train_test_split(df, test_size=float(sys.argv[3]), random_state=100)
        train_data.append(ltrain_df)
        test_data.append(ltest_df)

if int(sys.argv[4]) == 1:
    combined_df = pd.concat(dfs, ignore_index=True)
    train_df, test_df = train_test_split(combined_df, test_size=float(sys.argv[3]), random_state=100)
else:
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

print("Training set LLM distribution:")
print(train_df['label'].value_counts())
print("Testing set LLM distribution:")
print(test_df['label'].value_counts())

train_df.to_csv('train.csv', sep='|', index=False)
test_df.to_csv('test.csv', sep='|', index=False)

print(f"number of LLMs {NUM_LABEL}")

class TextDataset(Dataset):
    def __init__(self, dataframe, max_length):
        self.dataframe = dataframe
        #pretrained tokennizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",cache_dir=cache)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        prefix = str(row['Prefix'])
        suffix = str(row['Suffix'])
        inputs = prefix + " " + suffix
        
        encoding = self.tokenizer.encode_plus(
            inputs,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)  
        attention_mask = encoding['attention_mask'].squeeze(0)  
        label = torch.tensor(row['label'], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

# Define model with 7 layers
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_LLMs):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, num_LLMs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        return x

max_length = 150  
input_size = max_length
train_dataset = TextDataset(train_df, max_length)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TextDataset(test_df, max_length)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = SimpleNN(input_size, NUM_LABEL)
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    torch.set_num_threads(4)
    print(f"Running on CPU with max threads: {torch.get_num_threads()}")
else:
    print("Running on GPU")

model.to(device)

# trianing
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids.float())
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    loss = total_loss / len(train_loader)
    #just to check current predictions on the same training set. Overfitting?
    accuracy = total_correct / total_samples * 100
    return loss, accuracy

# testing
def evaluate(model, test_loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids.float())
            preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    
    preds = np.concatenate(preds)
    true_labels = np.concatenate(true_labels)

    overall_accuracy = accuracy_score(true_labels, preds) * 100
    
    # Calculate per-class accuracy
    LLMs = np.unique(true_labels)  
    per_LLM_accuracy = {}

    for LLM in LLMs:
        total = np.sum(true_labels == LLM)  
        correct = np.sum((preds == LLM) & (true_labels == LLM))

        if total > 0:
            per_LLM_accuracy[LLM] = (correct / total) * 100
        else:
            per_LLM_accuracy[LLM] = 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')

    return overall_accuracy, per_LLM_accuracy, precision, recall, f1

# Training and evaluation
train_start_time = time.time()
training_pass = int(sys.argv[2])
for i in range(training_pass):
    loss, accuracy = train(model, train_loader, optimizer, device)
    print(f"Pass {i+1}/{training_pass}, Train Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
train_end_time = time.time()

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
