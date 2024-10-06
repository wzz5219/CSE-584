import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os

access_token = "" #put HF token

def maximize_cpu_cores():
    num_cores = os.cpu_count()
    torch.set_num_threads(num_cores)
    print(f"Using {num_cores} CPU cores for inference.")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU detected, using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU detected, switching to CPU.")
    maximize_cpu_cores()

cache_dir = "./bert/"
start_prompt = ""
tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

model = model.to(device)

input_csv_path = 'llama_prefixes.csv'
output_csv_path = 'completed_texts.csv'

df = pd.read_csv(input_csv_path)

def complete_text(prefix_text, max_length=50):
    prompt = f"{start_prompt} {prefix_text}"
    input_encodings = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=input_encodings['input_ids'],
            attention_mask=input_encodings['attention_mask'],  # Pass the attention mask
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    suffix_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    if suffix_text.startswith(prompt):
        suffix_text = suffix_text[len(prompt):].strip()
    
    suffix_text = suffix_text.replace("\n", " ").strip()
    stop_tokens = ['. ', '! ', '? ']
    stop_index = min((suffix_text.find(token) for token in stop_tokens if suffix_text.find(token) != -1), default=len(suffix_text))
    suffix_text = suffix_text[:stop_index + 1]
    print(prefix_text + '|' + suffix_text,  flush=True)
    return suffix_text

df['Suffix'] = df['Prefix'].apply(lambda x: complete_text(x))
df.to_csv(output_csv_path, sep='|', index=False)

print("done")
