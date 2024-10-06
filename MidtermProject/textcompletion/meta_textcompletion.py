import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

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

input_csv_path = 'llama_prefixes.csv'  
output_csv_path = 'completed_texts.csv'

df = pd.read_csv(input_csv_path)

#I used offline model.
cache_dirmodel="./models/models--meta-llama--Llama-2-7b-hf--0/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
cache_dirtokenizer="./models/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

tokenizer = AutoTokenizer.from_pretrained(cache_dirtokenizer)
model = AutoModelForCausalLM.from_pretrained(cache_dirmodel).to(device)

i = 0
def complete_text(prefix_text):
    inputs = tokenizer(prefix_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.3,
        )

    completed_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    suffix_text = completed_text[len(prefix_text):].strip()
    stop_tokens = ['\n', '.', '!', '?']
    stop_index = min((suffix_text.find(token) for token in stop_tokens if suffix_text.find(token) != -1), default=len(suffix_text))
    suffix_text = suffix_text[:stop_index + 1]
    print(prefix_text + '|' + suffix_text,  flush=True)
    return suffix_text

df['Suffix'] = df['Prefix'].apply(lambda x: complete_text(x))
df.to_csv(output_csv_path, index=False)
print("done")
